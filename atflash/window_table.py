# -*- coding: utf-8 -*-
"""Canonical dim-prefix window table for ATFlash.

Definition. For RoPE pair r in standard order (r=0 is the highest frequency):

    theta_r = base^(-2r/d),   lambda_r = 2*pi/theta_r
    window width of pair r:   w_r = k_ret * lambda_r
    pair r is active at key distance z   <=>   z <= w_r

Capping the width at the context length, w_r = min(k_ret*lambda_r, N), does not
change the activity test, because z <= N-1 < N.

Kernel layout (wavelength order). Sorted pair j corresponds to standard index
r = P-1-j, so j=0 is the lowest frequency and therefore the longest wavelength.
The two components of pair j occupy the contiguous dims (2j, 2j+1). The active
set is then a prefix along the head dimension:

    j < n_act(z),   n_act(z) = #{r : w_r >= z}

which is what makes the window expressible as a shortening of the query-key
contraction rather than as a mask.

Multiple-of-16 constraint (the K-direction atom of the MMA instruction):

    D_eff(z) = clamp(ceil(2*n_act(z)/16)*16, 16, d)

Rounding up widens the window, i.e. it errs toward keeping terms. Tiles are
quantized the same way: D_eff is decided from the smallest distance z_min inside
a (BLOCK_M, BLOCK_N) tile, again widening rather than narrowing. The reference
implementation and the kernel must use the same fp32 wvec and the same
comparison (wvec >= z); this is what makes them bit-comparable.
"""
import math

import torch


def wavelengths_sorted(head_dim: int, rope_base: float, k_ret: float,
                       device="cpu") -> torch.Tensor:
    """fp32 [P]: window width w'_j = k_ret * 2pi * base^(2(P-1-j)/d) per sorted pair j.

    Descending, so j=0 is the widest (lowest frequency). k_ret=inf disables the
    window and every pair stays active at all distances. Both the kernel and the
    reference compare against these fp32 values and an fp32 z.
    """
    P = head_dim // 2
    r_std = torch.arange(P - 1, -1, -1, dtype=torch.float64)  # j=0 -> r=P-1
    if math.isinf(k_ret):
        w = torch.full((P,), float("inf"), dtype=torch.float64)
    else:
        w = (k_ret * 2.0 * math.pi
             * torch.pow(torch.tensor(float(rope_base), dtype=torch.float64),
                         2.0 * r_std / head_dim))
    return w.to(torch.float32).to(device)


def n_active(z_min, wvec: torch.Tensor) -> int:
    """Number of pairs active at distance z_min (fp32 comparison, as in the kernel)."""
    return int((wvec >= float(z_min)).sum().item())


def deff_of_zmin(z_min, wvec: torch.Tensor, head_dim: int) -> int:
    """Prefix length along the head dim, rounded up to a multiple of 16 and
    clamped to [16, head_dim]."""
    deff = 2 * n_active(z_min, wvec)
    deff = ((deff + 15) // 16) * 16
    return max(16, min(head_dim, deff))


def zmin_tile(phase: str, q0: int, n0: int, block_m: int, block_n: int,
              n_ctx: int) -> int:
    """Smallest key distance inside a tile (same formula as the kernel; partial
    tiles are still evaluated with a full block_n)."""
    if phase == "decode":
        return max(0, (n_ctx - 1) - (n0 + block_n - 1))
    return max(0, q0 - (n0 + block_n - 1))


def deff_segments(phase: str, q0: int, n_hi: int, block_m: int, block_n: int,
                  n_ctx: int, wvec: torch.Tensor, head_dim: int):
    """Segments (n_start, n_end, deff) covering the key range [0, n_hi) for query tile q0.

    deff is constant within a key tile and monotone in distance, so it collapses
    into contiguous runs. Used by the reference's banded matmul and by the MMA
    slice statistics below.
    """
    segs = []
    n0 = 0
    while n0 < n_hi:
        deff = deff_of_zmin(zmin_tile(phase, q0, n0, block_m, block_n, n_ctx),
                            wvec, head_dim)
        n1 = n0 + block_n
        while n1 < n_hi:
            d2 = deff_of_zmin(zmin_tile(phase, q0, n1, block_m, block_n, n_ctx),
                              wvec, head_dim)
            if d2 != deff:
                break
            n1 += block_n
        segs.append((n0, min(n1, n_hi), deff))
        n0 = n1
    return segs


def mma_slice_stats(phase: str, m_ctx: int, n_ctx: int, block_m: int,
                    block_n: int, wvec: torch.Tensor, head_dim: int):
    """Theoretical MMA k-slice counts (16 dims per slice), windowed vs full,
    together with the ratio of K bytes loaded."""
    total_win = 0
    total_full = 0
    kload_win = 0
    kload_full = 0
    full_slices = head_dim // 16
    for q0 in range(0, m_ctx, block_m):
        n_hi = min(q0 + block_m, n_ctx) if phase == "prefill" else n_ctx
        for (n0, n1, deff) in deff_segments(phase, q0, n_hi, block_m, block_n,
                                            n_ctx, wvec, head_dim):
            ntile = (n1 - n0 + block_n - 1) // block_n
            total_win += ntile * ((deff + 15) // 16)
            total_full += ntile * full_slices
            kload_win += ntile * deff
            kload_full += ntile * head_dim
    return {
        "qk_mma_slices_windowed": total_win,
        "qk_mma_slices_full": total_full,
        "qk_slice_ratio": total_win / max(1, total_full),
        "kload_dim_ratio": kload_win / max(1, kload_full),
    }
