# -*- coding: utf-8 -*-
"""Host-side window table for the FA4 port (pure Python, no torch).

Scripts in this repository import it as `atflash_window`; the FA4 companion
repository ships the same code as `flash_attn/cute/atflash_window.py`. It is the
torch-free twin of atflash/window_table.py:

  - window widths (the rule):
        RoPE pair r in standard order (r = 0 is the highest frequency):
            theta_r  = rope_base ** (-2*r / head_dim)
            lambda_r = 2*pi / theta_r
            w_r      = k_ret * lambda_r
        The kernel stores pairs in wavelength-sorted order: sorted pair j is standard
        pair r = P-1-j (j = 0 is the lowest frequency, i.e. the longest wavelength) and
        occupies dims (2j, 2j+1). Hence
            wvec[j] = k_ret * 2*pi * rope_base ** (2*(P-1-j) / head_dim), descending.
        A pair is read at causal distance z  <=>  z <= w_r  (fp32 comparison).

  - D_eff, the length of the dim prefix that is read (16-dim snap of the MMA):
        D_eff(z) = clamp(round_up_16(2 * #{j : wvec[j] >= z}), 16, head_dim)

  - zhi ("z high"), the largest distance at which each 16-dim level is still read:
        zhi[i] = max{ z >= 0 integer : D_eff(z) >= 16*(i+1) }   for i in 0..head_dim//16-1
        (capped at n_ctx; any cap >= the largest causal diagonal gives the same level
        boundaries in the kernel, because boundaries clamp at 0).

The tuple returned by zhi_table() / window_zhi_from_kret() is what the FA4 forward
kernels accept as the compile-time `window_zhi` argument.

Precondition: Q and K must already be permuted into the wavelength-sorted pair
layout above, so that the set of pairs read at any distance is a dim prefix.

fp32 note: the kernel compares fp32 widths against fp32 distances. Here the widths
are rounded to fp32 through struct packing, and z is an integer < 2**24, hence
exactly representable in fp32, so the Python (fp64) comparison `wvec_f32 >= z` is
bit-equivalent to the kernel's fp32 comparison.

Running this file performs structural self-checks only. The equality of these
tables with the ones the kernels and the torch reference use is checked by
tools/cpu_checks.py (CPU) and by the gates (GPU).
"""
import math
import struct
from typing import Sequence, Tuple


def _f32(x: float) -> float:
    """Round a Python float to fp32 precision (value preserved exactly in fp64)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def wavelengths_sorted(head_dim: int, rope_base: float, k_ret: float) -> Tuple[float, ...]:
    """fp32 [P]: sorted pair j (lowest frequency first) window width, descending.

    Torch-free port of atflash/window_table.py::wavelengths_sorted:
        w'_j = k_ret * 2*pi * rope_base ** (2*(P-1-j) / head_dim)
    computed in fp64 and rounded to fp32 (same as torch float64 -> float32 cast).
    k_ret=inf means "no window" (all pairs always active).
    """
    P = head_dim // 2
    if math.isinf(k_ret):
        return tuple(float("inf") for _ in range(P))
    out = []
    for j in range(P):
        r_std = float(P - 1 - j)
        w = k_ret * 2.0 * math.pi * (float(rope_base) ** (2.0 * r_std / float(head_dim)))
        out.append(_f32(w))
    return tuple(out)


def n_active(z_min: int, wvec: Sequence[float]) -> int:
    """Number of pairs read at distance z_min (fp32 comparison, as in the kernel)."""
    z = float(z_min)
    return sum(1 for w in wvec if w >= z)


def deff_of_zmin(z_min: int, wvec: Sequence[float], head_dim: int) -> int:
    """Dim-prefix length: 2*n_active, rounded up to a multiple of 16, clamped to
    [16, head_dim]. Same rule as atflash/window_table.py::deff_of_zmin."""
    deff = 2 * n_active(z_min, wvec)
    deff = ((deff + 15) // 16) * 16
    return max(16, min(head_dim, deff))


def zhi_table(wvec: Sequence[float], head_dim: int, n_ctx: int) -> Tuple[int, ...]:
    """zhi[i] = max{ z >= 0 int : D_eff(z) >= 16*(i+1) }, capped at n_ctx.

    Binary search per level; valid because D_eff(0) = head_dim >= e always holds
    and D_eff is non-increasing in z.
    """
    assert head_dim % 16 == 0, "head_dim must be a multiple of 16"
    assert len(wvec) == head_dim // 2, "wvec must have head_dim//2 entries"
    assert 0 < n_ctx < 2 ** 24, "n_ctx must be exactly representable in fp32"
    out = []
    for i in range(head_dim // 16):
        e = 16 * (i + 1)
        if deff_of_zmin(n_ctx, wvec, head_dim) >= e:
            out.append(n_ctx)
            continue
        lo, hi = 0, n_ctx  # deff(0) = head_dim >= e is always true
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if deff_of_zmin(mid, wvec, head_dim) >= e:
                lo = mid
            else:
                hi = mid - 1
        out.append(lo)
    return tuple(out)


def window_zhi_from_kret(
    head_dim: int, rope_base: float, k_ret: float, n_ctx: int
) -> Tuple[int, ...]:
    """Window rule (k_ret) -> the `window_zhi` tuple of the FA4 forward kernels, i.e.
    the `atflash_window_zhi` argument of flash_attn.cute.interface._flash_attn_fwd."""
    return zhi_table(wavelengths_sorted(head_dim, rope_base, k_ret), head_dim, n_ctx)


if __name__ == "__main__":
    # Structural self-check only; no reference numbers are asserted here (see
    # tools/cpu_checks.py for the equality checks against the recorded tables).
    for d, base in ((64, 10000.0), (128, 500000.0), (128, 1000000.0)):
        for n_ctx in (8192, 32768):
            wv = wavelengths_sorted(d, base, 2.0)
            assert all(wv[j] >= wv[j + 1] for j in range(len(wv) - 1)), "wvec not descending"
            t = zhi_table(wv, d, n_ctx)
            assert len(t) == d // 16
            assert all(t[i] >= t[i + 1] for i in range(len(t) - 1)), "zhi not non-increasing"
            assert all(0 <= z <= n_ctx for z in t)
            # snap rule invariants at both ends
            assert deff_of_zmin(0, wv, d) == d
            assert deff_of_zmin(n_ctx, wv, d) >= 16
            print(f"d={d} base={base:g} n_ctx={n_ctx}: zhi={t}")
    print("atflash_window self-check OK")
