# -*- coding: utf-8 -*-
"""Phase C task#14 root-cause: is the C2 FA4-OFF 2.5x slowness a HARNESS strided-
input artifact? Harness feeds FA4 query.transpose(1,2) (non-contig BSHD from BHSD).
Compare FA4 speed on contiguous BSHD vs strided BSHD vs strided+.contiguous().
7B dims, 256K, causal, bf16. GPU."""
import statistics
import sys

import torch

sys.path.insert(0, "/home/USER/atflash/agents/agentA/fa-b1/flash_attn/cute")
from flash_attn.cute.interface import _flash_attn_fwd

hq, hkv, d, N = 28, 4, 128, 262144
torch.manual_seed(0)


def timed(fn, warmup=3, iters=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


# contiguous BSHD (what the probe used = fast)
qc = torch.randn(1, N, hq, d, dtype=torch.bfloat16, device="cuda")
kc = torch.randn(1, N, hkv, d, dtype=torch.bfloat16, device="cuda")
vc = torch.randn(1, N, hkv, d, dtype=torch.bfloat16, device="cuda")
# BHSD contiguous (what HF cache/attention gives), then transpose -> strided BSHD (harness path)
qb = torch.randn(1, hq, N, d, dtype=torch.bfloat16, device="cuda")
kb = torch.randn(1, hkv, N, d, dtype=torch.bfloat16, device="cuda")
vb = torch.randn(1, hkv, N, d, dtype=torch.bfloat16, device="cuda")
qs, ks, vs = qb.transpose(1, 2), kb.transpose(1, 2), vb.transpose(1, 2)

print(f"N={N} 7B dims. contig={qc.is_contiguous()} strided_view_contig={qs.is_contiguous()}")
t_contig = timed(lambda: _flash_attn_fwd(qc, kc, vc, causal=True, pack_gqa=True)[0])
t_strided = timed(lambda: _flash_attn_fwd(qs, ks, vs, causal=True, pack_gqa=True)[0])
t_fixed = timed(lambda: _flash_attn_fwd(qs.contiguous(), ks.contiguous(), vs.contiguous(),
                                        causal=True, pack_gqa=True)[0])
print(f"FA4 contiguous BSHD   : {t_contig:8.1f} ms")
print(f"FA4 strided BSHD(harness): {t_strided:8.1f} ms  ({t_strided/t_contig:.2f}x contig)")
print(f"FA4 strided+.contiguous(): {t_fixed:8.1f} ms  (incl copy)")
# cost of the .contiguous() copy alone
t_copy = timed(lambda: (qs.contiguous(), ks.contiguous(), vs.contiguous()))
print(f".contiguous() copy alone : {t_copy:8.1f} ms")
print("STRIDED_TEST_DONE")
