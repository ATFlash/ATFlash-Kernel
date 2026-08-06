# -*- coding: utf-8 -*-
"""Phase C task#14: isolate the prefill ATTENTION kernel cost (7B GQA dims) —
FA4(single vs chunked) / SDPA-cuDNN / FlashInfer stock single_prefill — at
256K/512K. Decides FI prefill port. bf16, causal, hq28/hkv4/d128. GPU(gpu_lock)."""
import statistics
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, "/home/USER/atflash/agents/agentA/fa-b1/flash_attn/cute")
from flash_attn.cute.interface import _flash_attn_fwd
import flashinfer

hq, hkv, d = 28, 4, 128
torch.manual_seed(0)


def timed(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


for N in (262144, 524288):
    q = torch.randn(1, N, hq, d, dtype=torch.bfloat16, device="cuda")  # BSHD
    k = torch.randn(1, N, hkv, d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, N, hkv, d, dtype=torch.bfloat16, device="cuda")
    qh, kh, vh = (t.transpose(1, 2).contiguous() for t in (q, k, v))  # BHSD for SDPA
    qn, kn, vn = q[0], k[0], v[0]                                     # NHD for FI

    def fa4_single():
        return _flash_attn_fwd(q, k, v, causal=True, pack_gqa=True)[0]

    def fa4_chunked(C):
        def f():
            for s in range(0, N, C):
                e = min(s + C, N)
                _flash_attn_fwd(q[:, s:e], k[:, :e], v[:, :e], causal=True, pack_gqa=True)
        return f

    def sdpa_cudnn():
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True)

    def fi_prefill():
        return flashinfer.single_prefill_with_kv_cache(qn, kn, vn, causal=True)

    res = {}
    for name, fn in [("FA4_single", fa4_single),
                     ("FA4_chunk64k", fa4_chunked(65536)),
                     ("FA4_chunk128k", fa4_chunked(131072)),
                     ("sdpa_cudnn", sdpa_cudnn),
                     ("FI_prefill", fi_prefill)]:
        try:
            res[name] = timed(fn)
        except Exception as ex:
            res[name] = f"ERR:{type(ex).__name__}:{str(ex)[:80]}"
    print(f"N={N}:")
    for nm, ms in res.items():
        print(f"  {nm:16s} {ms if isinstance(ms,str) else round(ms,1):>10} ms")
    # correctness FA4 vs SDPA vs FI (bf16 level)
    try:
        of = fa4_single().float()
        os_ = sdpa_cudnn().transpose(1, 2)[0].float()
        oi = fi_prefill().float()
        print(f"  max|FA4-SDPA|={(of[0]-os_).abs().max().item():.3e} "
              f"max|FI-SDPA|={(oi-os_).abs().max().item():.3e}")
    except Exception as ex:
        print("  correctness err:", str(ex)[:100])
    del q, k, v, qh, kh, vh, qn, kn, vn
    torch.cuda.empty_cache()
print("PREFILL_PROBE_DONE")
