# -*- coding: utf-8 -*-
"""ATFlash B1: host-side window table for the distance-banded dim-prefix QK window.

This module is a pure-Python port (no torch dependency) of the canonical ATFlash
route-(i) window table:

  - window vector ("wavelength-proportional", verify_restructure.py section (4) /
    kernel_window/w01_torch_ref/window_table.py::wavelengths_sorted):
        RoPE pair r (standard order, r=0 highest frequency):
            theta_r  = rope_base ** (-2*r / head_dim)
            lambda_r = 2*pi / theta_r
            w_r      = k_ret * lambda_r
        kernel layout is wavelength-SORTED: sorted pair j = standard pair r = P-1-j
        (j=0 = lowest frequency = longest wavelength, occupying dims (2j, 2j+1)).
        Hence wvec[j] = k_ret * 2*pi * rope_base ** (2*(P-1-j) / head_dim), descending.
        A pair is active at causal distance z  <=>  z <= w_r  (fp32 comparison).

  - D_eff snap rule (w01 deff_of_zmin / w06 _deff_host, identical semantics):
        D_eff(z) = clamp(round_up_16(2 * #{j : wvec[j] >= z}), 16, head_dim)

  - zhi table (w06 _zhi_tensor, identical binary search):
        zhi[i] = max{ z >= 0 integer : D_eff(z) >= 16*(i+1) }   for i in 0..head_dim//16-1
        (capped at n_ctx; any cap >= the largest causal diagonal produces identical
        level boundaries in the kernel, because boundaries clamp at 0).

The tuple returned by zhi_table()/window_zhi_from_kret() is what
FlashAttentionForwardSm80/Sm120 accept as the compile-time `window_zhi` argument.

PRECONDITION (same as the Triton w0x kernels): Q and K must be pre-permuted to the
wavelength-sorted pair layout described above, so that the active set at any distance
is a dim-PREFIX.

fp32 semantics note: the Triton reference compares fp32 wvec against fp32 z inside the
kernel and uses the same comparison on the host (w06 "kernel と同一の fp32 比較
semantics"). Here wvec entries are rounded to fp32 via struct packing, and z is an
integer < 2**24, hence exactly representable in fp32 -- the Python float (fp64)
comparison `wvec_f32 >= z` is then bit-equivalent to the fp32 comparison.

VERIFICATION CHECKLIST (RTX, GPU box; do not skip):
  [ ] zhi_table(wavelengths_sorted(...), ...) must match w06 _zhi_tensor output with
      0 diff for the shapes under test (llama d=128 base=500000.0, qwen d=128 base=
      1000000.0, N in {8k, 32k}), and the induced per-tile D_eff must match
      w01_torch_ref window_table.deff_segments / verify_restructure.py section (4)
      conventions (active = z <= w_r) with 0 diff.
"""
import math
import struct
from typing import Sequence, Tuple


def _f32(x: float) -> float:
    """Round a Python float to fp32 precision (value preserved exactly in fp64)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def wavelengths_sorted(head_dim: int, rope_base: float, k_ret: float) -> Tuple[float, ...]:
    """fp32 [P]: sorted pair j (lowest frequency first) window width, descending.

    Port of kernel_window/w01_torch_ref/window_table.py::wavelengths_sorted:
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
    """#pairs active at distance z_min (fp32 comparison; identical to kernel/w01)."""
    z = float(z_min)
    return sum(1 for w in wvec if w >= z)


def deff_of_zmin(z_min: int, wvec: Sequence[float], head_dim: int) -> int:
    """dim-prefix length: 2*n_active, rounded UP to a multiple of 16, clamped to
    [16, head_dim]. Identical to w01 deff_of_zmin / w06 _deff_host."""
    deff = 2 * n_active(z_min, wvec)
    deff = ((deff + 15) // 16) * 16
    return max(16, min(head_dim, deff))


def zhi_table(wvec: Sequence[float], head_dim: int, n_ctx: int) -> Tuple[int, ...]:
    """zhi[i] = max{ z >= 0 int : D_eff(z) >= 16*(i+1) }, capped at n_ctx.

    Port of w06 _zhi_tensor (identical binary search; D_eff(0) = head_dim >= e always
    holds, D_eff is non-increasing in z).
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
    """Convenience: wavelength-proportional window (k_ret) -> `window_zhi` tuple for
    FlashAttentionForwardSm80/Sm120 (and the `atflash_window_zhi` argument of
    interface._flash_attn_fwd)."""
    return zhi_table(wavelengths_sorted(head_dim, rope_base, k_ret), head_dim, n_ctx)


if __name__ == "__main__":
    # Structural self-check only (no reference numbers are asserted here; the
    # 0-diff check against w06 _zhi_tensor / w01 deff_segments runs on the GPU box).
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
