# -*- coding: utf-8 -*-
"""End-to-end harness: FA4-port prefill + FlashInfer-port decode inside HF
transformers, measuring exact (SDPA) vs windowed quality and speed over the
whole request (prefill + decode) on real benchmark prompts (shared/data/).

- Integration: AttentionInterface("atf_window") + an apply_rotary_pos_emb patch.
  Right after RoPE, the head-dim of q/k is permuted into descending-wavelength
  pair layout (pair j -> dims 2j,2j+1); the cache holds the permuted K.
  prefill = FA4 (BSHD view), decode = FlashInfer (HND view).
- Window tables: WIDTHS come from the raw (pre-scaling) RoPE base, i.e. the same
  zhi tables as the published kernel microbenchmarks. ORDER (the
  permute) comes from the model's real inv_freq argsort; if that argsort does
  not match the raw canonical order the run stops with SystemExit before
  measuring. Snap granularity: FA4 = 16 dims, FI = 8 dims.
- MODE: exact = F.sdpa(enable_gqa) / win = window ON / nowin = window OFF
  (anchor: window OFF matches exact numerically and is bit-identical to the
  stock kernel [gate B0]).
"""
import argparse
import datetime
import importlib
import json
import math
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("HF_HOME", "/home/USER/atflash/shared/data/hf_cache")
sys.path.insert(0, "/home/USER/atflash/shared/src")

import contextlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# --- FA4 (prefill) + FlashInfer (decode) ports -------------------------------
# window WIDTH = raw (pre-scaling) RoPE base zhi table == the published kernel
# microbenchmarks; window ORDER = the harness permute (real inv_freq).
# fiwin_common = kernels/flashinfer_port/fiwin_common.py and atflash_window =
# window_tables/atflash_window.py in this repository (the paths below are placeholders).
sys.path.insert(0, "/home/USER/atflash/agents/agentB/fiwin/scripts")
sys.path.insert(0, "/home/USER/atflash/agents/agentA/fa-b1/flash_attn/cute")
from flash_attn.cute.interface import _flash_attn_fwd
import flashinfer  # noqa: F401
from flashinfer.decode import single_decode_with_kv_cache_with_jit_module
from fiwin_common import build_atfwin_module, zhi_slice_vec8
from atflash_window import window_zhi_from_kret, wavelengths_sorted

DATA = "/home/USER/atflash/shared/data"
RESULTS = "/home/USER/atflash/shared/results"
WIN_ZHI_CAP = 1_100_000  # FA4 zhi n_ctx cap: must be >= the max causal diagonal.
# Covers the 1M series (Qwen 7B-1M max_pos 1,010,000). With cap<N the
# long-wavelength slices would be wrongly deactivated; for llama at 128K
# cap>>N so there is no effect (zhi clamps at 0).
LB_FILE = f"{DATA}/longbench_v2_sample/sample.jsonl"  # overridable via --lb-file

MODE = {"mode": "exact"}
PREFILL_CHUNK = {"v": None}  # chunked-prefill length (None = single shot); set via --prefill-chunk
NULL_PASS = dict(tokens=[], prefill_ms=0.0, step_ms=[], wall_s=0.0, logits=[],
                 peak_alloc=0, free_before=0, free_after=0, stopped_by_text=False)
STATE = {}  # perm, fa_zhi (tuple), fi_zhi8 (cuda int32), fi_mod, sm_scale, head_dim

MODELS = {
    "qwen": "Qwen/Qwen2.5-0.5B",
    "llama": "unsloth/Llama-3.2-3B",
    "qwen7b1m": "Qwen/Qwen2.5-7B-Instruct-1M",  # Phase C: base 1e7, GQA g7, d128
}
PREFILL_CFG = {  # fastest configuration of the tile sweep
    64: dict(block_m=128, block_n=64, block_d=16, num_warps=4, num_stages=2),
    128: dict(block_m=64, block_n=64, block_d=32, num_warps=4, num_stages=2),
}
DECODE_CFG = dict(block_n=128, block_d=32, num_warps=4, num_stages=2,
                  sm_target=256)


# ---------------------------------------------------------------- integration
def atflash_attention(module, query, key, value, attention_mask,
                      scaling=None, dropout=0.0, **kwargs):
    mode = MODE["mode"]
    b, hq, L, d = query.shape
    S = key.shape[-2]
    if mode.startswith("exact"):
        if mode == "exact_cudnn":
            ctx = sdpa_kernel([SDPBackend.CUDNN_ATTENTION])
        elif mode == "exact_flash":
            ctx = sdpa_kernel([SDPBackend.FLASH_ATTENTION])
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            out = F.scaled_dot_product_attention(
                query, key, value, is_causal=(L > 1), enable_gqa=True,
                scale=scaling)
        return out.transpose(1, 2).contiguous(), None
    windowed = (mode == "win")
    # query/key/value are HF BHSD: q=[b,hq,L,d], k/v=[b,hkv,S,d] (already permuted).
    if L == 1:
        # FlashInfer decode: q=[hq,d], KV=HND [hkv,S,d] (=key[0], zero-copy view).
        q_fi = query[0, :, 0, :]                    # [hq, d]
        k_fi = key[0]                               # [hkv, S, d] (HND)
        v_fi = value[0]
        zhi = STATE["fi_zhi8"] if windowed else None
        o = single_decode_with_kv_cache_with_jit_module(
            STATE["fi_mod"], q_fi, k_fi, v_fi,
            None, zhi, 0.0, float(STATE["sm_scale"]), 1.0, 1.0 / 1e4,
            kv_layout="HND")                        # [hq, d]
        return o.view(1, 1, o.shape[0], o.shape[1]), None
    assert L <= S, f"L>S invalid: L={L} S={S}"  # L<S = chunked/continued prefill
    # FA4 prefill: BSHD views (strided accepted, verified by gate B0). The q chunk
    # sees the whole KV under a bottom-right causal mask, which is bit-identical
    # to full prefill (verified by gate C0). out=[b,L,hq,d].
    q_fa = query.transpose(1, 2)                    # [b,L,hq,d]
    k_fa = key.transpose(1, 2)                      # [b,S,hkv,d]
    v_fa = value.transpose(1, 2)
    kw = {"atflash_window_zhi": STATE["fa_zhi"]} if windowed else {}
    o = _flash_attn_fwd(q_fa, k_fa, v_fa, causal=True, pack_gqa=True, **kw)[0]
    return o.contiguous(), None                     # already [b,L,hq,d]


def setup_model(model_key, k_ret_prefill, k_ret_decode):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import AttentionInterface

    # Caution: transformers v5 force-routes any attn implementation whose name
    # contains "flash" through the FA hub-kernel path (utils/generic.py,
    # is_flash_attention_requested: `"flash" in name`). Hence the key is
    # "atf_window", which does not contain "flash".
    AttentionInterface.register("atf_window", atflash_attention)
    mid = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(
        mid, dtype=torch.bfloat16, attn_implementation="atf_window")
    model = model.to("cuda").eval()

    # ORDER (permute): from the model's REAL inv_freq (rope_scaling included).
    inv = model.model.rotary_emb.inv_freq.detach().float().cpu()  # [P] real
    P = inv.shape[0]
    d = 2 * P
    lam = 2.0 * math.pi / inv                       # real wavelengths (incr in r)
    wl_sorted, order = torch.sort(lam, descending=True)  # j=0 longest
    # Width/order separation: assert that argsort(real) equals the canonical raw
    # order (descending r). If it does not, the j-th raw-base width would land on
    # a different pair, so stop before measuring anything.
    canonical = torch.arange(P - 1, -1, -1, dtype=order.dtype)
    if not torch.equal(order, canonical):
        raise SystemExit(
            "STOP: real inv_freq argsort != raw canonical order; the width/order "
            f"separation breaks down. order[:8]={order[:8].tolist()} "
            f"canonical[:8]={canonical[:8].tolist()}")
    perm = torch.empty(d, dtype=torch.long)
    perm[0::2] = order                              # cos component (HF dim r)
    perm[1::2] = order + P                          # sin component (HF dim r+P)
    STATE["perm"] = perm.cuda()
    STATE["prefill_cfg"] = PREFILL_CFG[d]           # metadata only (FA4 self-tiles)
    STATE["head_dim"] = d
    # WIDTH: from the raw (pre-scaling) RoPE base == the published benches' tables.
    raw_base = float(getattr(model.config, "rope_theta", None)
                     or (getattr(model.config, "rope_scaling", None) or {}
                         ).get("rope_theta"))
    STATE["raw_base"] = raw_base
    # Widths use the raw base; k is an independent dial for prefill vs decode.
    # The order (permute) is shared and comes from the real inv_freq.
    STATE["k_ret_prefill"] = float(k_ret_prefill)
    STATE["k_ret_decode"] = float(k_ret_decode)
    STATE["fa_zhi"] = window_zhi_from_kret(
        d, raw_base, float(k_ret_prefill), WIN_ZHI_CAP)          # prefill-k (FA4)
    STATE["fi_zhi8"] = zhi_slice_vec8(
        d, raw_base, float(k_ret_decode)).cuda()                 # decode-k (FI)
    STATE["fi_mod"] = build_atfwin_module(torch.bfloat16, d)
    STATE["wvec_cpu"] = torch.tensor(
        wavelengths_sorted(d, raw_base, float(k_ret_prefill)), dtype=torch.float32)
    scaling = model.model.layers[0].self_attn.scaling
    assert abs(scaling - d ** -0.5) < 1e-9
    STATE["sm_scale"] = float(scaling)

    # rope patch (module-global function referenced by Attention.forward)
    mod_name = {"qwen": "transformers.models.qwen2.modeling_qwen2",
                "qwen7b1m": "transformers.models.qwen2.modeling_qwen2",
                "llama": "transformers.models.llama.modeling_llama"}[model_key]
    mmod = importlib.import_module(mod_name)
    orig_rope = mmod.apply_rotary_pos_emb

    def patched_rope(q, k, cos, sin, unsqueeze_dim=1):
        q, k = orig_rope(q, k, cos, sin, unsqueeze_dim)
        if not MODE["mode"].startswith("exact"):
            pm = STATE["perm"]
            q = q.index_select(-1, pm)
            k = k.index_select(-1, pm)
        return q, k

    mmod.apply_rotary_pos_emb = patched_rope
    return tok, model


# ------------------------------------------------------------------ gen loop
@torch.inference_mode()
def run_pass(model, ids, mode, max_new, eos_ids, forced=None, stop_strs=None,
             tok=None, collect_logits=True, prefill_chunk=None):
    """One pass = prefill + greedy/forced decode. ids: (1, L) cuda long.
    Returns a dict: tokens, prefill_ms, step_ms[], wall_s, logits (list of cpu
    fp32), peak_alloc, free_before/after.
    collect_logits=False marks a timing run: the per-step full-vocab logits.cpu()
    sync is skipped (only greedy's argmax->int scalar sync remains), which keeps
    synchronization noise out of wall_s."""
    MODE["mode"] = mode
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free_before = torch.cuda.mem_get_info()[0]
    ev = lambda: torch.cuda.Event(enable_timing=True)
    e0, e1 = ev(), ev()
    wall0 = time.perf_counter()
    e0.record()
    Lp = ids.shape[1]
    ch = prefill_chunk or PREFILL_CHUNK["v"] or Lp  # the 1M series chunks to bound activations
    cache = None
    for cs in range(0, Lp, ch):
        out = model(input_ids=ids[:, cs:min(cs + ch, Lp)], use_cache=True,
                    past_key_values=cache, logits_to_keep=1)
        cache = out.past_key_values
    e1.record()
    torch.cuda.synchronize()
    prefill_ms = e0.elapsed_time(e1)
    logits = out.logits[:, -1, :].float()
    step_logits = [logits.cpu()] if collect_logits else []
    cur = int(forced[0]) if forced is not None else int(logits.argmax(-1))
    tokens = [cur]
    pend = []
    text_done = False
    n_steps = len(forced) if forced is not None else max_new
    for t in range(1, n_steps):
        inp = torch.tensor([[cur]], device="cuda")
        s0, s1 = ev(), ev()
        s0.record()
        out = model(input_ids=inp, use_cache=True, past_key_values=cache)
        s1.record()
        logits = out.logits[:, -1, :].float()
        if collect_logits:
            step_logits.append(logits.cpu())    # syncs this step (quality run)
        cache = out.past_key_values
        pend.append((s0, s1))
        if forced is not None:
            cur = int(forced[t])
            tokens.append(cur)
        else:
            cur = int(logits.argmax(-1))
            tokens.append(cur)
            if cur in eos_ids:
                break
            if stop_strs and (t % 8 == 0 or t == n_steps - 1):
                txt = tok.decode(tokens)
                if any(s in txt for s in stop_strs):
                    text_done = True
                    break
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - wall0
    step_ms = [a.elapsed_time(b) for a, b in pend]
    return dict(tokens=tokens, prefill_ms=prefill_ms, step_ms=step_ms,
                wall_s=wall_s, logits=step_logits,
                peak_alloc=torch.cuda.max_memory_allocated(),
                free_before=free_before,
                free_after=torch.cuda.mem_get_info()[0],
                stopped_by_text=text_done)


def kl_top1(logits_e, logits_w):
    """Per-step KL(P_e||P_w) (natural log, full vocab) and top-1. CPU fp32."""
    T = min(len(logits_e), len(logits_w))
    kls, t1e, t1w = [], [], []
    for i in range(T):
        le = logits_e[i].cuda()
        lw = logits_w[i].cuda()
        lpe = F.log_softmax(le, dim=-1)
        lpw = F.log_softmax(lw, dim=-1)
        pe = lpe.exp()
        kls.append(float((pe * (lpe - lpw)).sum()))
        t1e.append(int(le.argmax(-1)))
        t1w.append(int(lw.argmax(-1)))
    match = [int(a == b) for a, b in zip(t1e, t1w)]
    return np.array(kls), np.array(t1e), np.array(t1w), np.array(match)


def nll_of(logits, tokens):
    """Teacher-forced NLL: logits[i] is the distribution predicting tokens[i]."""
    T = min(len(logits), len(tokens))
    out = []
    for i in range(T):
        lp = F.log_softmax(logits[i].cuda(), dim=-1)
        out.append(float(-lp[0, tokens[i]]))
    return np.array(out)


# ------------------------------------------------------------------- benches
def load_gsm8k(n_shots=8, n_items=16):
    tr = [json.loads(l) for l in
          open(f"{DATA}/gsm8k/train.jsonl", encoding="utf-8")][:n_shots]
    te = [json.loads(l) for l in
          open(f"{DATA}/gsm8k/test.jsonl", encoding="utf-8")][:n_items]
    prefix = "".join(f"Question: {x['question']}\nAnswer: {x['answer']}\n\n"
                     for x in tr)
    items = []
    for i, x in enumerate(te):
        gold = x["answer"].split("####")[-1].strip().replace(",", "")
        items.append(dict(id=f"gsm8k_test{i}",
                          prompt=prefix + f"Question: {x['question']}\nAnswer:",
                          gold=gold))
    return items, dict(max_new=256, stop_strs=["\nQuestion:"], judge="gsm8k")


def gsm8k_judge(pred_text, gold):
    m = re.search(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)", pred_text)
    if not m:
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", pred_text)
        if not nums:
            return 0.0
        cand = nums[-1]
    else:
        cand = m.group(1)
    cand = cand.replace(",", "").rstrip(".")
    try:
        return float(abs(float(cand) - float(gold)) < 1e-6)
    except ValueError:
        return 0.0


def load_ruler(task, seqlen, n_items=8):
    path = f"{DATA}/ruler/qwen2.5-0.5b_tok/{seqlen}/{task}/validation.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")][:n_items]
    items = [dict(id=f"ruler_{task}_{seqlen}_{r['index']}",
                  prompt=r["input"] + r.get("answer_prefix", ""),
                  gold=r["outputs"]) for r in rows]
    judge = "ruler_part" if task.startswith("qa") else "ruler_all"
    return items, dict(max_new=64, stop_strs=None, judge=judge)


def ruler_judge(pred_text, gold, judge):
    p = pred_text.lower()
    hits = [1.0 if str(g).lower() in p else 0.0 for g in gold]
    return max(hits) if judge == "ruler_part" else sum(hits) / len(hits)


LB_TEMPLATE = (  # LongBench-v2 official 0-shot prompt structure (THUDM/LongBench-v2)
    "Please read the following text and answer the question below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\n\n"
    "Format your response as follows: \"The correct answer is (insert answer "
    "here)\".\nThe correct answer is ("
)


def load_longbench(tok, ctx_cap, n_items=6, skip=0, long_only=False,
                   max_new=32):
    rows = [json.loads(l) for l in
            open(LB_FILE, encoding="utf-8")]
    if long_only:  # 64K/128K bands: only genuine length=long docs (center-truncated to cap)
        rows = [r for r in rows if r.get("length") == "long"]
    rows = rows[skip:skip + n_items]
    items = []
    for r in rows:
        enc = tok(r["context"], add_special_tokens=False).input_ids
        if len(enc) > ctx_cap:  # center truncation (the LongBench convention)
            half = ctx_cap // 2
            ctx = (tok.decode(enc[:half]) + "\n...[truncated]...\n"
                   + tok.decode(enc[-half:]))
        else:
            ctx = r["context"]
        items.append(dict(
            id=f"lbv2_{r['_id']}",
            prompt=LB_TEMPLATE.format(context=ctx, question=r["question"],
                                      A=r["choice_A"], B=r["choice_B"],
                                      C=r["choice_C"], D=r["choice_D"]),
            gold=r["answer"]))
    return items, dict(max_new=max_new, stop_strs=[")"] if max_new <= 32
                       else None, judge="lbv2")


def lbv2_judge(pred_text, gold):
    m = re.search(r"([A-D])", pred_text.strip())
    return float(bool(m) and m.group(1) == gold)


# ---------------------------------------------------------------------- main
def smi(query):
    return subprocess.run(["nvidia-smi"] + query.split(),
                          capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["qwen", "llama", "qwen7b1m"])
    ap.add_argument("--bench", required=True,
                    choices=["gsm8k", "ruler", "longbench", "floor"])
    ap.add_argument("--tasks", default="niah_single_1,niah_multikey_1,qa_1")
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ctx-cap", type=int, default=8000)
    ap.add_argument("--lb-skip", type=int, default=0)
    ap.add_argument("--lb-file", default=None,
                    help="LongBench jsonl override (default: sample.jsonl)")
    ap.add_argument("--lb-long-only", action="store_true")
    ap.add_argument("--lb-max-new", type=int, default=32)
    ap.add_argument("--k-ret", type=float, default=2.0)
    ap.add_argument("--k-ret-prefill", type=float, default=None,
                    help="window k for prefill (FA4); default: --k-ret")
    ap.add_argument("--k-ret-decode", type=float, default=None,
                    help="window k for decode (FlashInfer); default: --k-ret")
    ap.add_argument("--timing-only", action="store_true",
                    help="timing run: skip per-step logits collection (keeps sync "
                         "noise out of wall time); quality (top1/ppl/KL) is not "
                         "collected")
    ap.add_argument("--prefill-chunk", type=int, default=None,
                    help="chunked-prefill length in tokens (None = single shot). "
                         "Used by the 1M series to bound activations. Leaves the "
                         "ports untouched and is bit-identical to full prefill "
                         "(verified by gate C0)")
    ap.add_argument("--no-cudnn", action="store_true",
                    help="skip the external-reference SDPA cuDNN pass (avoids OOM "
                         "and slowness at 1M); cudnn fields become 0/None")
    ap.add_argument("--smoke", action="store_true",
                    help="2 items + nowin anchor pass")
    ap.add_argument("--full-logits-items", type=int, default=1)
    ap.add_argument("--topic", default=None)
    args = ap.parse_args()
    if args.lb_file:
        global LB_FILE
        LB_FILE = args.lb_file
    PREFILL_CHUNK["v"] = args.prefill_chunk

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    topic = args.topic or f"e2e_{args.model}_{args.bench}" + (
        f"{args.seqlen // 1024}k" if args.bench == "ruler" else "") + (
        "_smoke" if args.smoke else "")
    out_dir = f"{RESULTS}/{ts}_{topic}"
    os.makedirs(out_dir, exist_ok=False)
    with open(f"{out_dir}/clocks_start.txt", "w") as f:
        f.write(smi("-q -d CLOCK"))

    kpf = args.k_ret_prefill if args.k_ret_prefill is not None else args.k_ret
    kdc = args.k_ret_decode if args.k_ret_decode is not None else args.k_ret
    tok, model = setup_model(args.model, kpf, kdc)
    eos_ids = {model.config.eos_token_id} if isinstance(
        model.config.eos_token_id, int) else set(model.config.eos_token_id)

    if args.bench == "floor":
        # KL floor calibration: on representative prompts, compare exact_cudnn
        # (re-run) / exact_flash / nowin / win teacher-forced against the
        # exact_cudnn baseline.
        g_it, g_cfg = load_gsm8k(n_items=1)
        g_cfg = dict(g_cfg, max_new=128)
        groups = [("floor_gsm8k0", g_it, g_cfg)]
        if args.model == "qwen":
            for t in ["niah_single_1", "qa_1"]:
                it, cfg2 = load_ruler(t, 32768, n_items=1)
                groups.append((f"floor_{t}_32k", it, cfg2))
        else:
            it, cfg2 = load_longbench(tok, args.ctx_cap, n_items=1)
            groups.append((f"floor_lbv2_cap{args.ctx_cap}", it, cfg2))
    elif args.bench == "gsm8k":
        items, cfg = load_gsm8k(n_items=(2 if args.smoke else args.n))
        groups = [("gsm8k", items, cfg)]
    elif args.bench == "ruler":
        groups = []
        for t in args.tasks.split(","):
            it, cfg = load_ruler(t, args.seqlen,
                                 n_items=(2 if args.smoke else args.n))
            groups.append((f"{t}_{args.seqlen}", it, cfg))
    else:
        items, cfg = load_longbench(tok, args.ctx_cap,
                                    n_items=(2 if args.smoke else args.n),
                                    skip=args.lb_skip,
                                    long_only=args.lb_long_only,
                                    max_new=args.lb_max_new)
        groups = [(f"lbv2_cap{args.ctx_cap}", items, cfg)]

    meta = dict(
        script="atflash_e2e.py", script_version="v1", agent="e2e",
        torch=torch.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        device=torch.cuda.get_device_name(0),
        driver_smi_at_start=smi(
            "--query-gpu=driver_version --format=csv,noheader").strip(),
        model=MODELS[args.model], dtype="bf16", batch=1,
        attn_impl="fafi(FA4 prefill / FlashInfer decode)",
        k_ret=args.k_ret, k_ret_prefill=STATE["k_ret_prefill"],
        k_ret_decode=STATE["k_ret_decode"], prefill_cfg=STATE["prefill_cfg"],
        decode_cfg=DECODE_CFG, raw_base=STATE["raw_base"],
        wvec_from="WIDTH=raw_base (same tables as published benches) / "
                  "ORDER=real inv_freq permute",
        wvec_head=STATE["wvec_cpu"][:4].tolist(),
        wvec_tail=STATE["wvec_cpu"][-4:].tolist(),
        head_dim=STATE["head_dim"],
        timing="CUDA events per forward + wall; greedy; single request",
        args=vars(args), started_utc=ts, hostname=os.uname().nodename)
    json.dump(meta, open(f"{out_dir}/meta_start.json", "w"), indent=1)

    stream = open(f"{out_dir}/results_stream.jsonl", "a")

    def log(rec):
        stream.write(json.dumps(rec) + "\n")
        stream.flush()

    # --- warmup (JIT compile + cudnn autotune; logged with a warmup tag) ---
    w_ids = tok(groups[0][1][0]["prompt"], return_tensors="pt"
                ).input_ids.cuda()
    warm_modes = (["nowin", "win"] if args.no_cudnn
                  else ["exact_cudnn", "nowin", "win"])
    if args.bench == "floor":
        warm_modes = ["exact_cudnn", "exact_flash", "nowin", "win"]
    for m in warm_modes:
        try:
            r = run_pass(model, w_ids, m, 8, eos_ids)
            log(dict(kind="warmup", mode=m, prefill_ms=r["prefill_ms"],
                     step_ms=r["step_ms"], prompt_len=int(w_ids.shape[1])))
        except torch.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            log(dict(kind="warmup", mode=m, status="oom_run",
                     prompt_len=int(w_ids.shape[1]), error=str(e)[:300]))

    if args.bench == "floor":
        summary = []
        for gname, items, cfg in groups:
            item = items[0]
            ids = tok(item["prompt"], return_tensors="pt").input_ids.cuda()
            plen = int(ids.shape[1])
            ref = run_pass(model, ids, "exact_cudnn", cfg["max_new"], eos_ids,
                           stop_strs=cfg["stop_strs"], tok=tok)
            rec = dict(kind="floor", group=gname, id=item["id"],
                       prompt_len=plen, gen_len=len(ref["tokens"]),
                       ref_mode="exact_cudnn")
            npz = dict(tokens_ref=np.array(ref["tokens"]),
                       prompt_len=np.array([plen]))
            for vname, vmode in [("exact_cudnn2", "exact_cudnn"),
                                 ("exact_flash", "exact_flash"),
                                 ("nowin", "nowin"), ("win", "win")]:
                r = run_pass(model, ids, vmode, cfg["max_new"], eos_ids,
                             forced=ref["tokens"])
                kls, _, _, match = kl_top1(ref["logits"], r["logits"])
                npz[f"kl_{vname}"] = kls
                npz[f"match_{vname}"] = match
                rec[f"{vname}_kl_mean"] = float(kls.mean())
                rec[f"{vname}_kl_p99"] = float(np.percentile(kls, 99))
                rec[f"{vname}_kl_max"] = float(kls.max())
                rec[f"{vname}_top1"] = float(match.mean())
            np.savez_compressed(f"{out_dir}/floor_{item['id']}.npz", **npz)
            log(rec)
            summary.append(rec)
        fin = dict(meta=meta, floor=summary,
                   finished_utc=datetime.datetime.now(
                       datetime.timezone.utc).strftime("%Y%m%d_%H%M%S"))
        json.dump(fin, open(f"{out_dir}/e2e_summary.json", "w"), indent=1)
        with open(f"{out_dir}/clocks_end.txt", "w") as f:
            f.write(smi("-q -d CLOCK"))
        stream.close()
        print("OUT_DIR", out_dir)
        for r in summary:
            print("FLOOR", json.dumps(r))
        return

    summary = []
    for gname, items, cfg in groups:
        g_sum = dict(group=gname, n=len(items), items=[])
        for ii, item in enumerate(items):
            ids = tok(item["prompt"], return_tensors="pt").input_ids.cuda()
            plen = int(ids.shape[1])
            rec = dict(kind="item", group=gname, id=item["id"],
                       prompt_len=plen)
            if args.timing_only:  # timing run: no logits collection = no sync noise in wall time
                try:
                    ex = (NULL_PASS if args.no_cudnn else
                          run_pass(model, ids, "exact_cudnn", cfg["max_new"],
                                   eos_ids, stop_strs=cfg["stop_strs"], tok=tok,
                                   collect_logits=False))
                    of = run_pass(model, ids, "nowin", cfg["max_new"], eos_ids,
                                  stop_strs=cfg["stop_strs"], tok=tok,
                                  collect_logits=False)
                    wg = run_pass(model, ids, "win", cfg["max_new"], eos_ids,
                                  stop_strs=cfg["stop_strs"], tok=tok,
                                  collect_logits=False)
                    dec_o = float(np.median(of["step_ms"])) if of["step_ms"] else 0
                    dec_w = float(np.median(wg["step_ms"])) if wg["step_ms"] else 0
                    dec_c = float(np.median(ex["step_ms"])) if ex["step_ms"] else 0
                    rec.update(
                        status="ok_timing", gen_len_off=len(of["tokens"]),
                        gen_len_win=len(wg["tokens"]),
                        prefill_ms_off=of["prefill_ms"],
                        prefill_ms_win=wg["prefill_ms"],
                        prefill_ms_cudnn=ex["prefill_ms"],
                        decode_ms_median_off=dec_o, decode_ms_median_win=dec_w,
                        decode_ms_median_cudnn=dec_c,
                        tokens_per_s_gpu_off=(1000.0 / dec_o if dec_o else None),
                        tokens_per_s_gpu_win=(1000.0 / dec_w if dec_w else None),
                        tokens_per_s_gpu_cudnn=(1000.0 / dec_c if dec_c else None),
                        req_time_s_off=of["wall_s"], req_time_s_win=wg["wall_s"],
                        req_time_s_cudnn=ex["wall_s"],
                        decode_speedup_win_vs_off=(dec_o / dec_w if dec_w else None),
                        prefill_speedup_win_vs_off=(
                            of["prefill_ms"] / wg["prefill_ms"]
                            if wg["prefill_ms"] else None),
                        req_speedup_win_vs_off=(
                            of["wall_s"] / wg["wall_s"] if wg["wall_s"] else None),
                        peak_alloc_mb=max(of["peak_alloc"], wg["peak_alloc"],
                                          ex["peak_alloc"]) / 1e6)
                except torch.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    rec.update(status="oom_run", error=str(e)[:300])
                log(rec)
                g_sum["items"].append(rec)
                continue
            try:
                # OFF/ON = same binary (FA4+FI, only the window toggles); cuDNN = external reference
                ex = (NULL_PASS if args.no_cudnn else
                      run_pass(model, ids, "exact_cudnn", cfg["max_new"], eos_ids,
                               stop_strs=cfg["stop_strs"], tok=tok))
                of = run_pass(model, ids, "nowin", cfg["max_new"], eos_ids,
                              stop_strs=cfg["stop_strs"], tok=tok)
                wg = run_pass(model, ids, "win", cfg["max_new"], eos_ids,
                              stop_strs=cfg["stop_strs"], tok=tok)
                wf = run_pass(model, ids, "win", cfg["max_new"], eos_ids,
                              forced=of["tokens"])
                kls, t1o, t1w, match = kl_top1(of["logits"], wf["logits"])
                nll_o = nll_of(of["logits"], of["tokens"])
                nll_w = nll_of(wf["logits"], of["tokens"])
                of_txt = tok.decode(of["tokens"])
                wg_txt = tok.decode(wg["tokens"])
                ex_txt = tok.decode(ex["tokens"])
                if cfg["judge"] == "gsm8k":
                    acc_o = gsm8k_judge(of_txt, item["gold"])
                    acc_w = gsm8k_judge(wg_txt, item["gold"])
                elif cfg["judge"] == "lbv2":
                    acc_o = lbv2_judge(of_txt, item["gold"])
                    acc_w = lbv2_judge(wg_txt, item["gold"])
                else:
                    acc_o = ruler_judge(of_txt, item["gold"], cfg["judge"])
                    acc_w = ruler_judge(wg_txt, item["gold"], cfg["judge"])

                npz = dict(
                    kl_win_vs_off=kls, top1_off=t1o, top1_win=t1w,
                    top1_match=match, nll_off=nll_o, nll_win_forced=nll_w,
                    tokens_off=np.array(of["tokens"]),
                    tokens_win_free=np.array(wg["tokens"]),
                    tokens_cudnn=np.array(ex["tokens"]),
                    step_ms_off=np.array(of["step_ms"]),
                    step_ms_win_free=np.array(wg["step_ms"]),
                    step_ms_cudnn=np.array(ex["step_ms"]),
                    prefill_ms=np.array([of["prefill_ms"], wg["prefill_ms"],
                                         ex["prefill_ms"]]),
                    wall_s=np.array([of["wall_s"], wg["wall_s"], ex["wall_s"]]),
                    prompt_len=np.array([plen]),
                    peak_alloc=np.array([of["peak_alloc"], wg["peak_alloc"],
                                         ex["peak_alloc"]]),
                    free_before=np.array([of["free_before"]]),
                    free_after=np.array([wg["free_after"]]))
                if ii < args.full_logits_items:
                    cap = 128
                    npz["full_logits_off_f16"] = torch.stack(
                        of["logits"][:cap]).squeeze(1).to(torch.float16).numpy()
                    npz["full_logits_winf_f16"] = torch.stack(
                        wf["logits"][:cap]).squeeze(1).to(torch.float16).numpy()
                np.savez_compressed(f"{out_dir}/{item['id']}.npz", **npz)

                dec_o = float(np.median(of["step_ms"])) if of["step_ms"] else 0
                dec_w = float(np.median(wg["step_ms"])) if wg["step_ms"] else 0
                dec_c = float(np.median(ex["step_ms"])) if ex["step_ms"] else 0
                rec.update(
                    status="ok", gen_len_off=len(of["tokens"]),
                    gen_len_win=len(wg["tokens"]),
                    prefill_ms_off=of["prefill_ms"],
                    prefill_ms_win=wg["prefill_ms"],
                    prefill_ms_cudnn=ex["prefill_ms"],
                    decode_ms_median_off=dec_o, decode_ms_median_win=dec_w,
                    decode_ms_median_cudnn=dec_c,
                    tokens_per_s_gpu_off=(1000.0 / dec_o if dec_o else None),
                    tokens_per_s_gpu_win=(1000.0 / dec_w if dec_w else None),
                    tokens_per_s_gpu_cudnn=(1000.0 / dec_c if dec_c else None),
                    req_time_s_off=of["wall_s"], req_time_s_win=wg["wall_s"],
                    req_time_s_cudnn=ex["wall_s"],
                    decode_speedup_win_vs_off=(dec_o / dec_w if dec_w else None),
                    prefill_speedup_win_vs_off=(
                        of["prefill_ms"] / wg["prefill_ms"]
                        if wg["prefill_ms"] else None),
                    req_speedup_win_vs_off=(
                        of["wall_s"] / wg["wall_s"] if wg["wall_s"] else None),
                    top1_agree_win_vs_off=float(match.mean()),
                    kl_mean=float(kls.mean()), kl_max=float(kls.max()),
                    ppl_off=float(np.exp(nll_o.mean())),
                    ppl_win_forced=float(np.exp(nll_w.mean())),
                    acc_off=acc_o, acc_win=acc_w,
                    peak_alloc_mb=max(of["peak_alloc"], wg["peak_alloc"],
                                      ex["peak_alloc"]) / 1e6,
                    smi_free_mb=int(torch.cuda.mem_get_info()[0] / 1e6),
                    pred_tail_off=of_txt[-160:], pred_tail_win=wg_txt[-160:])
            except torch.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                rec.update(status="oom_run", error=str(e)[:300])
            log(rec)
            g_sum["items"].append(rec)
        oks = [r for r in g_sum["items"]
               if r["status"] in ("ok", "ok_timing")]
        if oks:
            def md(k):
                vs = [r[k] for r in oks if r.get(k) is not None]
                return float(np.median(vs)) if vs else None

            def mnf(k):
                vs = [r[k] for r in oks if r.get(k) is not None]
                return float(np.mean(vs)) if vs else None
            g_sum.update(
                top1_agree_win_vs_off_mean=mnf("top1_agree_win_vs_off"),
                kl_mean=mnf("kl_mean"),
                kl_max=(float(np.max([r["kl_max"] for r in oks
                                      if r.get("kl_max") is not None]))
                        if any(r.get("kl_max") is not None for r in oks)
                        else None),
                acc_off=mnf("acc_off"), acc_win=mnf("acc_win"),
                ppl_off=mnf("ppl_off"), ppl_win=mnf("ppl_win_forced"),
                tok_s_gpu_off_med=md("tokens_per_s_gpu_off"),
                tok_s_gpu_win_med=md("tokens_per_s_gpu_win"),
                tok_s_gpu_cudnn_med=md("tokens_per_s_gpu_cudnn"),
                decode_speedup_win_vs_off_med=md("decode_speedup_win_vs_off"),
                prefill_speedup_win_vs_off_med=md("prefill_speedup_win_vs_off"),
                req_speedup_win_vs_off_med=md("req_speedup_win_vs_off"),
                req_time_s_off_med=md("req_time_s_off"),
                req_time_s_win_med=md("req_time_s_win"),
                req_time_s_cudnn_med=md("req_time_s_cudnn"),
                prefill_ms_off_med=md("prefill_ms_off"),
                prefill_ms_win_med=md("prefill_ms_win"),
                prefill_ms_cudnn_med=md("prefill_ms_cudnn"))
        summary.append(g_sum)

    # Theoretical window ratios (from the raw-base zhi): FlashInfer vec8 mean read
    # fraction s and the ceiling 2/(1+s).
    try:
        from fiwin_common import theory_s_vec8
        d = STATE["head_dim"]
        s_bar, ceiling = theory_s_vec8(d, STATE["raw_base"],
                                       STATE["k_ret_decode"], plen)
        mma = dict(fi_s_bar=s_bar, fi_ceiling_2_over_1ps=ceiling, at_len=plen,
                   raw_base=STATE["raw_base"], fa_zhi=list(STATE["fa_zhi"]),
                   fi_zhi8=STATE["fi_zhi8"].detach().cpu().tolist())
    except Exception as e:  # statistics only; never fail the run over this
        mma = dict(error=str(e)[:200])

    fin = dict(meta=meta, groups=summary, mma_slices=mma,
               finished_utc=datetime.datetime.now(
                   datetime.timezone.utc).strftime("%Y%m%d_%H%M%S"),
               smi_at_end=smi("--query-gpu=memory.used,memory.free "
                              "--format=csv,noheader").strip())
    json.dump(fin, open(f"{out_dir}/e2e_summary.json", "w"), indent=1)
    with open(f"{out_dir}/clocks_end.txt", "w") as f:
        f.write(smi("-q -d CLOCK"))
    stream.close()
    print("OUT_DIR", out_dir)
    for g in summary:
        print("GROUP", json.dumps({k: v for k, v in g.items()
                                   if k != "items"}))


if __name__ == "__main__":
    main()
