#!/usr/bin/env python
"""agentA B1 full prefill sweep (process A): cudnn + FA4 {off,on} x {packT,packF}.

7 official categories: llama 8k/32k/64k/160k + qwen 8k/32k/64k (qwen64k = OOD,
model ctx 32k — annotation inherited). BASELINE_PROTOCOL timing: probe -> warmup 10
-> n = clamp(1.5s/est, 10, 200) -> per-call CUDA events, median, all iters saved.
TF32 off, b=1, causal, bf16. Input-buffer rotation n_copies=2 for N>=65536 (new
categories; 8k/32k keep the legacy no-rotation convention for comparability with
the registered baseline/SOTA prefill numbers). Window: k_ret=2.0, zhi cap 262144
(>= max diag of all cases; level-equivalent to per-N tables).
Main metric: attainment of the 2/(1+s) ceiling, s = per-case
QK slice ratio (FA4 tile quantization) recorded as qk_slice_ratio_theory.
provisional (base PR#2656 head 4f1fde1; PR merged upstream as squash 5835c73 on
2026-07-07 but main has diverged -> rebase deferred).
"""
import argparse
import json
import os
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.atflash_window import window_zhi_from_kret

MODELS = [
    ("llama", 24, 8, 128, 500000.0, [8192, 32768, 65536, 163840]),
    ("qwen", 14, 2, 64, 1000000.0, [8192, 32768, 65536]),
]
K_RET = 2.0
N_CAP = 262144
ROT_MIN_N = 65536
ROT_COPIES = 2

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def fa4_tiles(d):
    return (128, 64) if d > 64 else (128, 128)


def fast_slice_stats(N, BM, BN, zhi_perN, d):
    """WT.mma_slice_stats (prefill) semantics via zhi table: deff(z)=16*#{i: zhi[i]>=z}."""
    full_slices = d // 16
    tw = tf = kw = kf = 0
    for q0 in range(0, N, BM):
        n_hi = min(q0 + BM, N)
        ntiles = (n_hi + BN - 1) // BN
        for t in range(ntiles):
            zmin = max(0, q0 - (t * BN + BN - 1))
            deff = 16 * sum(1 for z in zhi_perN if z >= zmin)
            tw += deff // 16
            tf += full_slices
            kw += deff
            kf += d
    return {"qk_slice_ratio": tw / max(1, tf), "kload_dim_ratio": kw / max(1, kf)}


def smi_probe():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate",
             "--format=csv,noheader"], text=True).strip()
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def bench_call(fn, warmup=10, target_s=1.5, max_iters=200, min_iters=10):
    fn()  # probe
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    est = time.perf_counter() - t0
    n = max(min_iters, min(max_iters, int(target_s / max(est, 1e-6))))
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = [starts[i].elapsed_time(ends[i]) for i in range(n)]
    return {
        "n_iters": n, "median_ms": statistics.median(times),
        "mean_ms": statistics.fmean(times), "min_ms": min(times),
        "max_ms": max(times), "std_ms": statistics.pstdev(times), "times_ms": times,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    stream_path = os.path.join(args.outdir, "results_stream_A.jsonl")

    with open(os.path.join(args.outdir, "clocks_start.txt"), "w") as f:
        f.write(subprocess.check_output(["nvidia-smi", "-q", "-d", "CLOCK"], text=True))

    meta = {
        "script": "b1_bench_protocol.py (process A, sweep v2)",
        "script_version": "v2",
        "agent": "agentA",
        "provisional": "base PR#2656 head 4f1fde1 + b1-window bc21b79 (unmerged-base; "
                       "upstream squash-merged as 5835c73 2026-07-07, main diverged)",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "dtype": "bfloat16", "batch": 1, "tf32": False,
        "k_ret": K_RET, "zhi_cap": N_CAP,
        "rotation": f"input buffers x{ROT_COPIES} cycled for N>={ROT_MIN_N} "
                     "(8k/32k legacy no-rotation for comparability)",
        "timing": "cuda events per-call, warmup 10, adaptive iters (target 1.5s, max 200)",
        "flops_formula": {"prefill_causal": "2*Hq*N^2*D (ideal; not window-adjusted)"},
        "ceiling_formula": "2/(1+s), s = qk_slice_ratio_theory (FA4 tile quantization)",
        "qwen_64k_note": "OOD: Qwen2.5-0.5B context window 32k; annotation inherited",
        "correctness_pairing": "same-day agentA w01 correctness runs (see ChangeLog)",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(args.outdir, "meta_start_A.json"), "w") as f:
        json.dump(meta, f, indent=1)

    results = []
    for (model, hq, hkv, d, base, Ns) in MODELS:
        zhi = window_zhi_from_kret(d, base, K_RET, N_CAP)
        BM, BN = fa4_tiles(d)
        for N in Ns:
            ncop = ROT_COPIES if N >= ROT_MIN_N else 1
            g = torch.Generator(device="cuda").manual_seed(2026)
            qs, ks, vs, qhs, khs, vhs = [], [], [], [], [], []
            for _ in range(ncop):
                q = torch.randn((1, N, hq, d), generator=g, device="cuda",
                                dtype=torch.float32).to(torch.bfloat16)
                k = torch.randn((1, N, hkv, d), generator=g, device="cuda",
                                dtype=torch.float32).to(torch.bfloat16)
                v = torch.randn((1, N, hkv, d), generator=g, device="cuda",
                                dtype=torch.float32).to(torch.bfloat16)
                qs.append(q); ks.append(k); vs.append(v)
                qhs.append(q.transpose(1, 2).contiguous())
                khs.append(k.transpose(1, 2).contiguous())
                vhs.append(v.transpose(1, 2).contiguous())
            zhi_perN = window_zhi_from_kret(d, base, K_RET, N)
            stats = fast_slice_stats(N, BM, BN, zhi_perN, d)
            s_ratio = stats["qk_slice_ratio"]
            flops = 2.0 * hq * N * N * d
            it = [0]

            def make_sdpa():
                def fn():
                    j = it[0] % ncop; it[0] += 1
                    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                        return F.scaled_dot_product_attention(
                            qhs[j], khs[j], vhs[j], is_causal=True, enable_gqa=True)
                return fn

            def make_fa4(pack, window):
                kw = {"atflash_window_zhi": zhi} if window else {}

                def fn():
                    j = it[0] % ncop; it[0] += 1
                    return _flash_attn_fwd(qs[j], ks[j], vs[j], causal=True,
                                           pack_gqa=pack, **kw)
                return fn

            impls = [("sdpa_cudnn", make_sdpa()),
                     ("fa4_off_packT", make_fa4(True, False)),
                     ("fa4_on_packT", make_fa4(True, True)),
                     ("fa4_off_packF", make_fa4(False, False)),
                     ("fa4_on_packF", make_fa4(False, True))]
            with torch.no_grad():
                o_off = _flash_attn_fwd(qs[0], ks[0], vs[0], causal=True, pack_gqa=False)[0]
                o_on = _flash_attn_fwd(qs[0], ks[0], vs[0], causal=True, pack_gqa=False,
                                       atflash_window_zhi=zhi)[0]
                d_onoff = (o_on.float() - o_off.float()).abs().max().item()
                del o_off, o_on
            torch.cuda.empty_cache()
            for (impl, fn) in impls:
                try:
                    r = bench_call(fn)
                    status = "ok"
                except Exception as e:  # noqa: BLE001
                    r = {}
                    status = f"error: {type(e).__name__}: {e}"[:300]
                rec = {
                    "phase": "prefill", "model": model, "impl": impl, "hq": hq,
                    "hkv": hkv, "d": d, "N": N, "causal": True, "status": status,
                    "n_input_copies": ncop,
                    "tflops_ideal": (flops / (r["median_ms"] * 1e-3) / 1e12) if r else None,
                    "qk_slice_ratio_theory": s_ratio,
                    "kload_dim_ratio_theory": stats["kload_dim_ratio"],
                    "ceiling_2_over_1ps": 2.0 / (1.0 + s_ratio),
                    "max_abs_on_vs_off_packF": d_onoff,
                    "ood": (model == "qwen" and N > 32768),
                    "smi": smi_probe(),
                    **r,
                }
                results.append(rec)
                with open(stream_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                med = r.get("median_ms")
                print(f"[A {model} N={N}] {impl}: "
                      f"{med:.4f} ms (n={r.get('n_iters')})" if med else
                      f"[A {model} N={N}] {impl}: {status}", flush=True)
            del qs, ks, vs, qhs, khs, vhs
            torch.cuda.empty_cache()

    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(args.outdir, "b1_bench_A.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)
    with open(os.path.join(args.outdir, "clocks_end.txt"), "w") as f:
        f.write(subprocess.check_output(["nvidia-smi", "-q", "-d", "CLOCK"], text=True))
    print("PROCESS A DONE")


if __name__ == "__main__":
    main()
