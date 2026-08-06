# -*- coding: utf-8 -*-
"""Phase C task#14 root-cause v2: measure the ACTUAL C2 shape = chunked prefill
(q_chunk=64k attending growing KV, causal) for FA4 / SDPA-cuDNN / FI, at 256K.
Single-forward misled (all ~equal); C2 is chunked. Also permute (index_select)
cost. Decides FI prefill port. 7B dims, bf16. GPU."""
import statistics
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, "/home/USER/atflash/agents/agentA/fa-b1/flash_attn/cute")
from flash_attn.cute.interface import _flash_attn_fwd
import flashinfer

hq, hkv, d, N, C = 28, 4, 128, 262144, 65536
torch.manual_seed(0)


def timed(fn, warmup=2, iters=4):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


# BSHD for FA4, BHSD for SDPA, NHD for FI (all same values, contiguous)
q = torch.randn(1, N, hq, d, dtype=torch.bfloat16, device="cuda")
k = torch.randn(1, N, hkv, d, dtype=torch.bfloat16, device="cuda")
v = torch.randn(1, N, hkv, d, dtype=torch.bfloat16, device="cuda")
qh, kh, vh = (t.transpose(1, 2).contiguous() for t in (q, k, v))
qn, kn, vn = q[0].contiguous(), k[0].contiguous(), v[0].contiguous()
perm = torch.randperm(d, device="cuda")


def fa4_chunked():
    for s in range(0, N, C):
        e = min(s + C, N)
        _flash_attn_fwd(q[:, s:e], k[:, :e], v[:, :e], causal=True, pack_gqa=True)


def cudnn_chunked():
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        for s in range(0, N, C):
            e = min(s + C, N)
            F.scaled_dot_product_attention(qh[:, :, s:e], kh[:, :, :e], vh[:, :, :e],
                                           is_causal=True, enable_gqa=True)


def fi_chunked():
    for s in range(0, N, C):
        e = min(s + C, N)
        flashinfer.single_prefill_with_kv_cache(qn[s:e], kn[:e], vn[:e], causal=True)


def permute_all():
    for s in range(0, N, C):
        e = min(s + C, N)
        qh[:, :, s:e].index_select(-1, perm)
        kh[:, :, s:e].index_select(-1, perm)


for name, fn in [("FA4_chunked", fa4_chunked), ("cuDNN_chunked", cudnn_chunked),
                 ("FI_chunked", fi_chunked), ("permute(idx_select)", permute_all)]:
    try:
        print(f"{name:22s} {timed(fn):8.1f} ms")
    except Exception as ex:
        print(f"{name:22s} ERR {type(ex).__name__}: {str(ex)[:90]}")
print("CHUNKED_ATTN_PROBE_DONE")
