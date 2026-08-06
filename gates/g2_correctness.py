# -*- coding: utf-8 -*-
"""fiwin G2+G3: windowed correctness vs vec8-exact reference + real-data KL.

References (decode, M=1):
  ref_vec8  : exactly the kernel's window model -- slice tx (8 dims) active for
              age <= zhi_slice[tx]; fp32 ("ref32") and bf16-matmul ("ref16", anchor).
  ref_pair  : pair-exact canonical w01 window (zhi per pair = floor(w_j)), fp32.
Gate (W1 rule, agentA convention): max|out - ref32_vec8| <= 2*max|ref16-ref32| + 2e-3.
Reported extra: snap_cost = max|ref32_vec8 - ref32_pair| (vec8 snap-up inclusion),
real-data attention-weight KL (exact vs vec8 window, fp64) for the V2cap qwen case.
RoPE/permute helpers ported from agentA b1_correctness_w01.py (canonical layout).
"""
import json
import math
import os
import sys

import numpy as np
import torch
import flashinfer  # noqa: F401  (shadow package sanity)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fiwin_common import (VEC_SIZE, build_atfwin_module, make_case, run_atfwin,
                          zhi_slice_vec8)
from atflash_window import wavelengths_sorted  # via fiwin_common sys.path

K_RET = 2.0
V2CAP = "/home/USER/atflash/shared/results/20260705_162750_kernel_M4e_V2cap"


def _slice_masks(N, zhi, device):
    """bool [n_slices, N]: slice active per position (age = N-1-pos)."""
    age = (N - 1) - torch.arange(N, device=device)
    return age[None, :] <= zhi[:, None]


def ref_decode_sliced(q, k, v, zhi, slice_w, mode="fp32"):
    """q (hq,d) k,v (N,hkv,d) bf16; zhi int per slice of width slice_w. -> (hq,d) f32."""
    hq, d = q.shape
    N, hkv, _ = k.shape
    G = hq // hkv
    ns = d // slice_w
    dev = q.device
    act = _slice_masks(N, zhi.to(dev), dev)                       # (ns, N)
    head_map = torch.arange(hq, device=dev) // G
    kx = k.permute(1, 0, 2)[head_map]                             # (hq, N, d)
    vx = v.permute(1, 0, 2)[head_map]
    if mode == "fp32":
        qf, kf = q.float(), kx.float()
        S = torch.zeros(hq, N, device=dev)
        for t in range(ns):
            sl = slice(t * slice_w, (t + 1) * slice_w)
            part = torch.einsum("hd,hnd->hn", qf[:, sl], kf[:, :, sl])
            S += part * act[t][None, :].float()
    else:  # bf16 anchor: bf16 partial matmuls, fp32 accumulate (mirror of w01 ref16)
        S = torch.zeros(hq, N, device=dev)
        for t in range(ns):
            sl = slice(t * slice_w, (t + 1) * slice_w)
            part = torch.einsum("hd,hnd->hn", q[:, sl], kx[:, :, sl]).float()
            S += part * act[t][None, :].float()
    S = S / math.sqrt(d)
    W = torch.softmax(S, dim=-1)
    out = torch.einsum("hn,hnd->hd", W, vx.float())
    return out, S


def zhi_pair(d, base, k_ret, device):
    w = wavelengths_sorted(d, base, k_ret)
    return torch.tensor([min(2**31 - 1, int(math.floor(x))) if not math.isinf(x)
                         else 2**31 - 1 for x in w], dtype=torch.int64, device=device)


def _rope_hf(x, positions, inv_freq):
    P = x.shape[-1] // 2
    t = positions.float()[:, None] * inv_freq[None, :]
    cos, sin = torch.cos(t), torch.sin(t)
    x = x.float()
    x1, x2 = x[..., :P], x[..., P:]
    out = torch.empty_like(x)
    out[..., :P] = x1 * cos - x2 * sin
    out[..., P:] = x2 * cos + x1 * sin
    return out


def hf_to_sorted_pair_perm(d):
    P = d // 2
    src = torch.empty(d, dtype=torch.long)
    for j in range(P):
        r = P - 1 - j
        src[2 * j] = r
        src[2 * j + 1] = r + P
    return src


def gate_case(name, q, k, v, base, results, kl=False):
    d = q.shape[-1]
    zs = zhi_slice_vec8(d, base, K_RET)
    ref32, S32 = ref_decode_sliced(q, k, v, zs, VEC_SIZE, "fp32")
    ref16, _ = ref_decode_sliced(q, k, v, zs, VEC_SIZE, "bf16")
    refp, _ = ref_decode_sliced(q, k, v, zhi_pair(d, base, K_RET, q.device), 2, "fp32")
    anchor = (ref16 - ref32).abs().max().item()
    mod = build_atfwin_module(torch.bfloat16, d)
    out = run_atfwin(mod, q, k, v, zhi_slice=zs).float()
    err = (out - ref32).abs().max().item()
    tol = 2.0 * anchor + 2e-3
    rec = {"case": name, "d": d, "N": k.shape[0], "k_ret": K_RET,
           "zhi_slice": zs.cpu().tolist(), "max_abs_err": err,
           "bf16_anchor": anchor, "tol": tol, "PASS": bool(err <= tol),
           "snap_cost_vs_pair_exact": (ref32 - refp).abs().max().item()}
    if kl:  # G3: attention-weight KL exact-vs-window, fp64
        hq = q.shape[0]
        G = hq // k.shape[1]
        head_map = torch.arange(hq, device=q.device) // G
        kx = k.permute(1, 0, 2)[head_map].double()
        Sfull = torch.einsum("hd,hnd->hn", q.double(), kx) / math.sqrt(d)
        Pw = torch.softmax(Sfull, -1)
        Qw = torch.softmax(S32.double(), -1)
        klv = (Pw * (Pw.clamp_min(1e-300).log() - Qw.clamp_min(1e-300).log())).sum(-1)
        rec["kl_exact_vs_vec8_max"] = klv.max().item()
        rec["kl_exact_vs_vec8_mean"] = klv.mean().item()
    results.append(rec)
    print({k2: (round(v2, 6) if isinstance(v2, float) else v2)
           for k2, v2 in rec.items() if k2 != "zhi_slice"}, flush=True)
    return rec["PASS"]


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    results, ok = [], True
    for model, kvs in [("llama", [512, 2048, 8192, 32768]),
                       ("qwen", [512, 2048, 8192, 32768])]:
        for kv in kvs:
            q, k, v, base = make_case(model, kv)
            ok &= gate_case(f"{model}_rand_kv{kv}", q, k, v, base, results)
    # real qwen decode-form (V2cap, pre-RoPE bf16 bits)
    z = np.load(os.path.join(V2CAP, "qkv_qwen_L12.npz"))
    kf = torch.from_numpy(z["k_full"].astype(np.uint16)).view(torch.bfloat16).cuda()
    vf = torch.from_numpy(z["v_full"].astype(np.uint16)).view(torch.bfloat16).cuda()
    qt = torch.from_numpy(z["q_tail"].astype(np.uint16)).view(torch.bfloat16).cuda()
    hkv, N, d = kf.shape
    inv_freq = torch.tensor(json.load(open(os.path.join(V2CAP, "meta_qwen.json")))["inv_freq"],
                            device="cuda", dtype=torch.float32)
    k_r = _rope_hf(kf, torch.arange(N, device="cuda"), inv_freq)
    q_r = _rope_hf(qt[:, -1:, :], torch.tensor([N - 1], device="cuda"), inv_freq)
    perm = hf_to_sorted_pair_perm(d).cuda()
    k_s = k_r[..., perm].to(torch.bfloat16).permute(1, 0, 2).contiguous()  # (N,hkv,d)
    v_s = vf.permute(1, 0, 2).contiguous()
    q_s = q_r[..., perm].to(torch.bfloat16).squeeze(1).contiguous()        # (hq,d)
    ok &= gate_case("real_qwen_L12_decodeform", q_s, k_s, v_s, 1000000.0, results, kl=True)
    report = {"meta": {"purpose": "G2 vec8-window correctness + G3 real-data KL",
                       "gate": "max|out-ref32_vec8| <= 2*anchor+2e-3 (W1 rule)",
                       "torch": torch.__version__},
              "results": results, "pass_all": bool(ok)}
    with open(os.path.join(outdir, "g2_correctness.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("G2_ALL_PASS" if ok else "G2_FAIL")
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
