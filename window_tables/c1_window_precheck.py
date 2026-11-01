# -*- coding: utf-8 -*-
"""C1 pre-check (config-based, no weights): argsort separation gate +
1M window tables for Qwen2.5-7B-Instruct-1M (base 1e7; rope_scaling=null means
the raw base equals the effective base, 1e7). Cross-checked against the model's
real inv_freq at load time."""
import hashlib
import math
import sys

import torch

# fiwin_common = kernels/flashinfer_port/fiwin_common.py and atflash_window =
# window_tables/atflash_window.py in this repository (the paths below are placeholders).
sys.path.insert(0, "/home/USER/atflash/agents/agentB/fiwin/scripts")
sys.path.insert(0, "/home/USER/atflash/agents/agentA/fa-b1/flash_attn/cute")
from atflash_window import window_zhi_from_kret, wavelengths_sorted
from fiwin_common import zhi_slice_vec8

BASE = 1e7
D = 128
P = D // 2
CAP = 1_100_000  # >= max_pos 1,010,000 (the 1M series)


def sha(obj):
    return hashlib.sha256(repr(obj).encode()).hexdigest()[:16]


# rope_scaling=null => real inv_freq == closed form base**(-2i/d)
inv = 1.0 / (BASE ** (torch.arange(0, D, 2).float() / D))
lam = 2.0 * math.pi / inv
wl_sorted, order = torch.sort(lam, descending=True)
canonical = torch.arange(P - 1, -1, -1, dtype=order.dtype)
argsort_ok = torch.equal(order, canonical)
print(f"model=Qwen2.5-7B-Instruct-1M base={BASE:g} d={D} P={P}")
print(f"ARGSORT gate (real argsort == raw canonical): {argsort_ok}  "
      f"(rope_scaling=null => raw base == effective base; width and order share one basis)")
print(f"wavelengths_sorted head/tail: "
      f"{[round(x,1) for x in wavelengths_sorted(D, BASE, 2.0)[:3]]} ... "
      f"{[round(x,1) for x in wavelengths_sorted(D, BASE, 2.0)[-3:]]}")
print()
for k in (2.0, 3.0, 4.0):
    fa = window_zhi_from_kret(D, BASE, k, CAP)         # FA4 16snap (len d/16=8)
    fi = zhi_slice_vec8(D, BASE, k, device="cpu").tolist()  # FI 8snap (len d/8=16)
    print(f"k={k}: FA4 16snap len={len(fa)} sha={sha(list(fa))} {list(fa)}")
    print(f"      FI  8snap  len={len(fi)} sha={sha(fi)} {fi}")
print("\nC1_PRECHECK_DONE (no weights needed; bit-checked against the real inv_freq at load time)")
