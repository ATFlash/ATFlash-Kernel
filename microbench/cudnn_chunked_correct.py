# -*- coding: utf-8 -*-
"""CRITICAL: is cuDNN chunked prefill (SDPA is_causal, q_len<kv_len) CORRECT
(bottom-right) or WRONG (top-left, under-masked = artificially fast)? Compare vs
FA4 full-causal reference. If wrong, the '4x faster' + C2 cuDNN ref are invalid.
Small N for exact check. 7B dims, bf16."""
import sys
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, "/home/USER/atflash/agents/agentA/fa-b1/flash_attn/cute")
from flash_attn.cute.interface import _flash_attn_fwd

hq, hkv, d, N, C = 28, 4, 128, 4096, 1024
torch.manual_seed(0)
q = torch.randn(1, N, hq, d, dtype=torch.bfloat16, device="cuda")
k = torch.randn(1, N, hkv, d, dtype=torch.bfloat16, device="cuda")
v = torch.randn(1, N, hkv, d, dtype=torch.bfloat16, device="cuda")
qh, kh, vh = (t.transpose(1, 2).contiguous() for t in (q, k, v))

ref = _flash_attn_fwd(q, k, v, causal=True, pack_gqa=True)[0].float()  # [1,N,hq,d] full causal


def cudnn_single():
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True)
    return o.transpose(1, 2).float()


def cudnn_chunked():
    outs = []
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        for s in range(0, N, C):
            e = min(s + C, N)
            o = F.scaled_dot_product_attention(qh[:, :, s:e], kh[:, :, :e], vh[:, :, :e],
                                               is_causal=True, enable_gqa=True)
            outs.append(o.transpose(1, 2))
    return torch.cat(outs, dim=1).float()


cs = cudnn_single()
cc = cudnn_chunked()
print(f"N={N} chunk={C} 7B dims")
print(f"cuDNN single  vs FA4-full-causal: max|Δ|={(cs-ref).abs().max().item():.4e}  "
      f"mean|Δ|={(cs-ref).abs().mean().item():.4e}")
print(f"cuDNN chunked vs FA4-full-causal: max|Δ|={(cc-ref).abs().max().item():.4e}  "
      f"mean|Δ|={(cc-ref).abs().mean().item():.4e}")
# per-chunk: is chunk>0 wrong? check last chunk rows vs ref
last = slice(N - C, N)
print(f"last-chunk block max|Δ| (chunked vs ref)={(cc[:,last]-ref[:,last]).abs().max().item():.4e}")
print("CORRECT_CHECK_DONE (small=correct/bottom-right; large=wrong/top-left)")
