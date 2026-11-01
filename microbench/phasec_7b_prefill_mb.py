# -*- coding: utf-8 -*-
"""Prefill-kernel sweep on Qwen2.5-7B-Instruct-1M shapes (the Qwen panel of the
prefill figure).
Shape: n_q = 28 / n_kv = 4 (GQA group 7) / d = 128, RoPE base 1e7.
Single-forward prefill (L == S, no chunking), so cuDNN through SDPA is a valid
baseline (the top-left causal-mask issue only arises with chunking).
Series: stock FA4 / FA4 + window (k = 2, the window_tables/ table for base 1e7) /
SDPA cuDNN / stock FlashInfer.
Timing protocol as in b1_bench_protocol.py: warm-up, adaptive iterations (target
1.5 s, at most 200), CUDA-event median, input rotation for N >= 65536. Runs under
an exclusive GPU lock and writes timestamped raw results. bf16, causal, TF32 off."""
import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, "/home/USER/atflash/agents/agentA/fa-b1/flash_attn/cute")
from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.atflash_window import window_zhi_from_kret
import flashinfer

HQ, HKV, D, BASE = 28, 4, 128, 1e7
K_RET = 2.0
N_CAP = 1_100_000
ROT_MIN_N, ROT_COPIES = 65536, 2
BM, BN = 128, 64  # fa4_tiles(d=128)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def fast_slice_stats(N, zhi_perN, d):
    full = d // 16
    tw = tf = 0
    for q0 in range(0, N, BM):
        n_hi = min(q0 + BM, N)
        ntiles = (n_hi + BN - 1) // BN
        for t in range(ntiles):
            zmin = max(0, q0 - (t * BN + BN - 1))
            deff = 16 * sum(1 for z in zhi_perN if z >= zmin)
            tw += deff // 16
            tf += full
    return tw / max(1, tf)


def bench(fn, warmup=10, target_s=1.5, max_iters=200, min_iters=10):
    fn(); torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
    est = time.perf_counter() - t0
    n = max(min_iters, min(max_iters, int(target_s / max(est, 1e-6))))
    ss = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        ss[i].record(); fn(); ee[i].record()
    torch.cuda.synchronize()
    return statistics.median([ss[i].elapsed_time(ee[i]) for i in range(n)]), n


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    Ns = [8192, 32768, 65536, 131072, 262144, 524288]
    zhi = window_zhi_from_kret(D, BASE, K_RET, N_CAP)
    meta = dict(script="phasec_7b_prefill_mb.py", model="Qwen2.5-7B-Instruct-1M",
                hq=HQ, hkv=HKV, d=D, group=HQ // HKV, base=BASE, k_ret=K_RET,
                torch=torch.__version__, device=torch.cuda.get_device_name(0),
                flashinfer=flashinfer.__version__, dtype="bf16", tf32=False,
                mode="single-forward L==S (cuDNN valid)",
                started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    results = []
    for N in Ns:
        ncop = ROT_COPIES if N >= ROT_MIN_N else 1
        g = torch.Generator(device="cuda").manual_seed(2026)
        qs, ks, vs, qhs, khs, vhs, qns, kns, vns = ([] for _ in range(9))
        try:
            for _ in range(ncop):
                q = torch.randn((1, N, HQ, D), generator=g, device="cuda", dtype=torch.bfloat16)
                k = torch.randn((1, N, HKV, D), generator=g, device="cuda", dtype=torch.bfloat16)
                v = torch.randn((1, N, HKV, D), generator=g, device="cuda", dtype=torch.bfloat16)
                qs.append(q); ks.append(k); vs.append(v)
                qhs.append(q.transpose(1, 2).contiguous()); khs.append(k.transpose(1, 2).contiguous())
                vhs.append(v.transpose(1, 2).contiguous())
                qns.append(q[0].contiguous()); kns.append(k[0].contiguous()); vns.append(v[0].contiguous())
        except torch.OutOfMemoryError:
            print(f"N={N}: OOM building inputs (ncop={ncop}) -> skip", flush=True)
            torch.cuda.empty_cache(); continue
        it = [0]
        def nxt():
            j = it[0] % ncop; it[0] += 1; return j
        def f_stock():
            j = nxt(); return _flash_attn_fwd(qs[j], ks[j], vs[j], causal=True, pack_gqa=True)[0]
        def f_win():
            j = nxt(); return _flash_attn_fwd(qs[j], ks[j], vs[j], causal=True, pack_gqa=True,
                                              atflash_window_zhi=zhi)[0]
        def f_cudnn():
            j = nxt()
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return F.scaled_dot_product_attention(qhs[j], khs[j], vhs[j], is_causal=True, enable_gqa=True)
        def f_fi():
            j = nxt(); return flashinfer.single_prefill_with_kv_cache(qns[j], kns[j], vns[j], causal=True)
        row = dict(N=N, n_copies=ncop)
        for name, fn in [("stock_fa4", f_stock), ("fa4_win", f_win),
                         ("sdpa_cudnn", f_cudnn), ("stock_fi", f_fi)]:
            try:
                ms, n = bench(fn); row[name] = ms; row[name + "_iters"] = n
            except Exception as e:
                row[name] = None; row[name + "_err"] = f"{type(e).__name__}:{str(e)[:80]}"
            torch.cuda.synchronize()
        s = fast_slice_stats(N, window_zhi_from_kret(D, BASE, K_RET, N), D)
        row["qk_slice_ratio_s"] = s
        row["ceiling_2_over_1ps"] = 2.0 / (1.0 + s)
        vals = [row[k] for k in ("stock_fa4", "fa4_win", "sdpa_cudnn", "stock_fi") if row.get(k)]
        mn = min(vals) if vals else None
        if row.get("stock_fa4") and row.get("fa4_win"):
            row["win_vs_stock"] = row["stock_fa4"] / row["fa4_win"]
            row["win_recovery"] = (row["stock_fa4"] / row["fa4_win"] - 1) / max(1e-9, row["ceiling_2_over_1ps"] - 1)
        for k in ("stock_fa4", "fa4_win", "sdpa_cudnn", "stock_fi"):
            if row.get(k) and mn:
                row[k + "_vs_min"] = row[k] / mn
        results.append(row)
        print(f"N={N:>7} n{ncop}: stock={row.get('stock_fa4')} win={row.get('fa4_win')} "
              f"cudnn={row.get('sdpa_cudnn')} fi={row.get('stock_fi')} "
              f"| win/stock={row.get('win_vs_stock')} s={s:.3f} ceil={row['ceiling_2_over_1ps']:.3f}",
              flush=True)
        for lst in (qs, ks, vs, qhs, khs, vhs, qns, kns, vns):
            lst.clear()
        torch.cuda.empty_cache()
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(dict(meta=meta, results=results), open(os.path.join(outdir, "mb.json"), "w"), indent=1)
    print("OUTDIR", outdir); print("MB_DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", required=True)
    main(ap.parse_args().outdir)
