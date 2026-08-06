# -*- coding: utf-8 -*-
"""ATFlash 路線(i) 窓のみ — canonical dim-prefix window table (w01).

定義 (shared/src/verify_restructure.py §(4) / atflash_r_outer.html E10 に整合):
  RoPE pair r (標準順, r=0 最高周波): theta_r = base^(-2r/d), lambda_r = 2*pi/theta_r
  pair r の窓幅 w_r = k_ret * lambda_r
    (verify_restructure.py は w_r = min(k_ret*lam_r, N) だが min(.,N) は
     活性判定 z <= w_r に影響しない: z <= N-1 < N)
  key 距離 z で pair r が active <=> z <= w_r   (verify: active = zv <= w_r)

カーネル layout (波長順): sorted pair j = 標準 r = P-1-j (j=0 が最低周波=最長波長)。
  pair j の成分は dim (2j, 2j+1) に連続配置。
  active set は dim-prefix: j < n_act(z),  n_act(z) = #{r : w_r >= z}。
  (html E10 の r_lo は標準順 suffix。同一集合の並べ替え。
   タイルの z は z_min を使う = html の z_hi 案より広い = 品質安全側)

16 倍数制約 (MMA K-atom):
  D_eff(z) = clamp(ceil(2*n_act(z)/16)*16, 16, d)  # 切上げ = 窓拡大(品質安全側)
  タイル量子化: (BLOCK_M, BLOCK_N) タイル内の最小距離 z_min で D_eff を決定 (拡大方向)。
  reference と kernel は同一の fp32 wvec と同一の比較 (wvec >= z) を用いる。
"""
import math

import torch


def wavelengths_sorted(head_dim: int, rope_base: float, k_ret: float,
                       device="cpu") -> torch.Tensor:
    """fp32 [P]: sorted pair j (低周波先頭) の窓幅 w'_j = k_ret * 2pi * base^(2(P-1-j)/d).

    降順 (j=0 最大)。k_ret=inf で窓なし (全 pair 常時 active)。
    kernel と reference は必ずこの fp32 値と z (fp32 化) を比較する。
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
    """#pairs active at distance z_min (fp32 comparison; kernel と同一)."""
    return int((wvec >= float(z_min)).sum().item())


def deff_of_zmin(z_min, wvec: torch.Tensor, head_dim: int) -> int:
    """dim-prefix length, 16 の倍数へ切上げ, [16, head_dim] に clamp."""
    deff = 2 * n_active(z_min, wvec)
    deff = ((deff + 15) // 16) * 16
    return max(16, min(head_dim, deff))


def zmin_tile(phase: str, q0: int, n0: int, block_m: int, block_n: int,
              n_ctx: int) -> int:
    """タイル内最小距離 (kernel と同一式; 部分タイルも full block_n で計算)."""
    if phase == "decode":
        return max(0, (n_ctx - 1) - (n0 + block_n - 1))
    return max(0, q0 - (n0 + block_n - 1))


def deff_segments(phase: str, q0: int, n_hi: int, block_m: int, block_n: int,
                  n_ctx: int, wvec: torch.Tensor, head_dim: int):
    """q タイル q0 に対する key 範囲 [0, n_hi) の (n_start, n_end, deff) 区間列.

    deff は key タイル毎に一定・距離単調なので連続区間に纏まる。
    reference の banded matmul と MMA スライス統計に使う。
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
    """理論 MMA k スライス数 (16dim/slice): windowed vs full と K-load bytes 比."""
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
