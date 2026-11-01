# -*- coding: utf-8 -*-
"""Reproducibility check: re-measure 256K and 512K (fresh seeds and allocations)
to confirm that stock FA4 slows down faster than quadratically between them
(×4.57, against ×3.96 with the window). 7B shapes, single forward, bf16."""
import statistics
import sys
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, "/home/USER/atflash/agents/agentA/fa-b1/flash_attn/cute")
from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.atflash_window import window_zhi_from_kret
import flashinfer

HQ, HKV, D, BASE = 28, 4, 128, 1e7
zhi = window_zhi_from_kret(D, BASE, 2.0, 1_100_000)


def bench(fn, warmup=8, iters=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


res = {}
for N in (262144, 524288):
    g = torch.Generator(device="cuda").manual_seed(7)
    q = torch.randn(1, N, HQ, D, generator=g, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, N, HKV, D, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, N, HKV, D, generator=g, device="cuda", dtype=torch.bfloat16)
    qh, kh, vh = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    qn, kn, vn = q[0].contiguous(), k[0].contiguous(), v[0].contiguous()
    r = {}
    r["stock"] = bench(lambda: _flash_attn_fwd(q, k, v, causal=True, pack_gqa=True)[0])
    r["win"] = bench(lambda: _flash_attn_fwd(q, k, v, causal=True, pack_gqa=True, atflash_window_zhi=zhi)[0])
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        r["cudnn"] = bench(lambda: F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True))
    r["fi"] = bench(lambda: flashinfer.single_prefill_with_kv_cache(qn, kn, vn, causal=True))
    res[N] = r
    print(f"N={N}: " + " ".join(f"{k}={v:.1f}" for k, v in r.items()) +
          f" | win/stock={r['stock']/r['win']:.3f}", flush=True)
    del q, k, v, qh, kh, vh, qn, kn, vn
    torch.cuda.empty_cache()
print("--- 256k->512k scaling (2x N) ---")
for s in ("stock", "win", "cudnn", "fi"):
    print(f"  {s:6s} ×{res[524288][s]/res[262144][s]:.2f}")
print("REMEASURE_DONE")
