# -*- coding: utf-8 -*-
"""w06_level_loops: prefill を deff レベル毎の分岐なしサブループへ分割.

変異点 (w05_static_slice 比・これのみ):
  - w05 の「k タイルループ内 runtime deff 分岐」を排除。同一 q タイル行では
    zmin が start_n に対し単調減少 = deff は単調非減少なので、各離散レベル
    e ∈ {16,32,...,HEAD_DIM} の k タイル列は連続区間。ホストで
    zhi_e = max{z: deff(z) >= e} を kernel と同一の fp32 比較semanticsで
    二分探索し (呼出間 cache)、kernel はレベル毎に固定形 dot の
    「分岐なしサブループ」を順に回す (SW pipelining / load 先読みが効く形)。
    diag/far ループ分割 (full-D 近傍帯 = 最終サブループ) を包含。
  - qk の縮約分割は w05 と同一の二進スライス (64/32/16, 次元昇順, §13 16 倍数)。
    q スライスは各サブループ直前に 1 回 global ロード (タイル毎再ロードなし)。
    tl.split 木は使わない (w05 の register エイリアス圧を回避)。
  - タイル処理順 = w03/w05 と同一 (start_n 昇順)・online softmax 同一
    → w05 が w03 と bit 一致だったのと同根拠で数値等価を期待 (w01 ref で検証)。
  - decode は w04_merge_par をそのまま import (変異 1 点のみ)。
"""
import math

import torch
import triton
import triton.language as tl

from kernel_window.w04_merge_par.kernel import window_attn_decode  # noqa: F401

NEG = tl.constexpr(-1.0e9)
MINIT = tl.constexpr(-1.0e30)
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _osm_tile(qk, m_i, l_i, acc, offs_m, offs_n, nmask, qk_scale,
              vb, svn, svd, HEAD_DIM: tl.constexpr):
    """online softmax 1 タイル更新 (w03 と同一算術・同一順)."""
    offs_d = tl.arange(0, HEAD_DIM)
    s2 = qk * qk_scale
    keep = nmask[None, :] & (offs_m[:, None] >= offs_n[None, :])
    s2 = tl.where(keep, s2, NEG)
    m_new = tl.maximum(m_i, tl.max(s2, 1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(s2 - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None]
    vblk = tl.load(vb + offs_n[:, None] * svn + offs_d[None, :] * svd,
                   mask=nmask[:, None], other=0.0)
    acc = tl.dot(p.to(vblk.dtype), vblk, acc)
    return m_new, l_i, acc


@triton.jit
def _qslice(qb, offs_m, sqm, sqd, mmask, BASE: tl.constexpr, W: tl.constexpr):
    offs_c = BASE + tl.arange(0, W)
    return tl.load(qb + offs_m[:, None] * sqm + offs_c[None, :] * sqd,
                   mask=mmask[:, None], other=0.0)


@triton.jit
def _kslice(kb, offs_n, skn, skd, nmask, BASE: tl.constexpr, W: tl.constexpr):
    offs_c = BASE + tl.arange(0, W)
    return tl.load(kb + offs_n[:, None] * skn + offs_c[None, :] * skd,
                   mask=nmask[:, None], other=0.0)


@triton.jit
def _prefill_fwd(Q, K, V, O, ZHI, sm_scale,
                 sqb, sqh, sqm, sqd,
                 skb, skh, skn, skd,
                 svb, svh, svn, svd,
                 sob, soh, som, sod,
                 HQ: tl.constexpr, GROUP: tl.constexpr,
                 N_CTX,
                 P: tl.constexpr, HEAD_DIM: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_D: tl.constexpr, WINDOWED: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    off_kh = off_h // GROUP
    qb = Q + off_b * sqb + off_h * sqh
    kb = K + off_b * skb + off_kh * skh
    vb = V + off_b * svb + off_kh * svh
    ob = O + off_b * sob + off_h * soh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_nb = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mmask = offs_m < N_CTX

    m_i = tl.full([BLOCK_M], MINIT, tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    qk_scale = sm_scale * LOG2E
    qfull = tl.load(qb + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
                    mask=mmask[:, None], other=0.0)

    hi = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX)

    if WINDOWED:
        base_m = pid_m * BLOCK_M
        if HEAD_DIM == 64:
            # レベル境界 (start_n 単位, 昇順): lvl16 [0,sn1) lvl32 [sn1,sn2)
            # lvl48 [sn2,sn3) full [sn3,hi)
            z32 = tl.load(ZHI + 1)
            z48 = tl.load(ZHI + 2)
            z64 = tl.load(ZHI + 3)
            sn1 = (tl.maximum(base_m - z32, 0) // BLOCK_N) * BLOCK_N
            sn2 = (tl.maximum(base_m - z48, 0) // BLOCK_N) * BLOCK_N
            sn3 = (tl.maximum(base_m - z64, 0) // BLOCK_N) * BLOCK_N
            sn1 = tl.minimum(sn1, hi)
            sn2 = tl.minimum(sn2, hi)
            sn3 = tl.minimum(sn3, hi)
            # --- lvl16 ---
            q16 = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 16)
            for start_n in range(0, sn1, BLOCK_N):
                offs_n = start_n + offs_nb
                nmask = offs_n < N_CTX
                kc = _kslice(kb, offs_n, skn, skd, nmask, 0, 16)
                qk = tl.dot(q16, tl.trans(kc))
                m_i, l_i, acc = _osm_tile(qk, m_i, l_i, acc, offs_m, offs_n,
                                          nmask, qk_scale, vb, svn, svd,
                                          HEAD_DIM)
            # --- lvl32 ---
            q32 = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 32)
            for start_n in range(sn1, sn2, BLOCK_N):
                offs_n = start_n + offs_nb
                nmask = offs_n < N_CTX
                kc = _kslice(kb, offs_n, skn, skd, nmask, 0, 32)
                qk = tl.dot(q32, tl.trans(kc))
                m_i, l_i, acc = _osm_tile(qk, m_i, l_i, acc, offs_m, offs_n,
                                          nmask, qk_scale, vb, svn, svd,
                                          HEAD_DIM)
            # --- lvl48 = (0,32)+(32,16) ---
            q32a = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 32)
            q48b = _qslice(qb, offs_m, sqm, sqd, mmask, 32, 16)
            for start_n in range(sn2, sn3, BLOCK_N):
                offs_n = start_n + offs_nb
                nmask = offs_n < N_CTX
                ka = _kslice(kb, offs_n, skn, skd, nmask, 0, 32)
                kb2 = _kslice(kb, offs_n, skn, skd, nmask, 32, 16)
                qk = tl.dot(q32a, tl.trans(ka))
                qk = tl.dot(q48b, tl.trans(kb2), qk)
                m_i, l_i, acc = _osm_tile(qk, m_i, l_i, acc, offs_m, offs_n,
                                          nmask, qk_scale, vb, svn, svd,
                                          HEAD_DIM)
            # --- full-D 近傍帯 (diag 含む) ---
            for start_n in range(sn3, hi, BLOCK_N):
                offs_n = start_n + offs_nb
                nmask = offs_n < N_CTX
                kfull = tl.load(kb + offs_n[:, None] * skn
                                + offs_d[None, :] * skd,
                                mask=nmask[:, None], other=0.0)
                qk = tl.dot(qfull, tl.trans(kfull))
                m_i, l_i, acc = _osm_tile(qk, m_i, l_i, acc, offs_m, offs_n,
                                          nmask, qk_scale, vb, svn, svd,
                                          HEAD_DIM)
        else:  # HEAD_DIM == 128
            z32 = tl.load(ZHI + 1)
            z48 = tl.load(ZHI + 2)
            z64 = tl.load(ZHI + 3)
            z80 = tl.load(ZHI + 4)
            z96 = tl.load(ZHI + 5)
            z112 = tl.load(ZHI + 6)
            z128 = tl.load(ZHI + 7)
            sn1 = (tl.maximum(base_m - z32, 0) // BLOCK_N) * BLOCK_N
            sn2 = (tl.maximum(base_m - z48, 0) // BLOCK_N) * BLOCK_N
            sn3 = (tl.maximum(base_m - z64, 0) // BLOCK_N) * BLOCK_N
            sn4 = (tl.maximum(base_m - z80, 0) // BLOCK_N) * BLOCK_N
            sn5 = (tl.maximum(base_m - z96, 0) // BLOCK_N) * BLOCK_N
            sn6 = (tl.maximum(base_m - z112, 0) // BLOCK_N) * BLOCK_N
            sn7 = (tl.maximum(base_m - z128, 0) // BLOCK_N) * BLOCK_N
            sn1 = tl.minimum(sn1, hi)
            sn2 = tl.minimum(sn2, hi)
            sn3 = tl.minimum(sn3, hi)
            sn4 = tl.minimum(sn4, hi)
            sn5 = tl.minimum(sn5, hi)
            sn6 = tl.minimum(sn6, hi)
            sn7 = tl.minimum(sn7, hi)
            # --- lvl16 ---
            q16 = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 16)
            for start_n in range(0, sn1, BLOCK_N):
                offs_n = start_n + offs_nb
                nmask = offs_n < N_CTX
                kc = _kslice(kb, offs_n, skn, skd, nmask, 0, 16)
                qk = tl.dot(q16, tl.trans(kc))
                m_i, l_i, acc = _osm_tile(qk, m_i, l_i, acc, offs_m, offs_n,
                                          nmask, qk_scale, vb, svn, svd,
                                          HEAD_DIM)
            # --- lvl32 ---
            q32 = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 32)
            for start_n in range(sn1, sn2, BLOCK_N):
                offs_n = start_n + offs_nb
                nmask = offs_n < N_CTX
                kc = _kslice(kb, offs_n, skn, skd, nmask, 0, 32)
                qk = tl.dot(q32, tl.trans(kc))
                m_i, l_i, acc = _osm_tile(qk, m_i, l_i, acc, offs_m, offs_n,
                                          nmask, qk_scale, vb, svn, svd,
                                          HEAD_DIM)
            # --- lvl48 = (0,32)+(32,16) ---
            q32a = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 32)
            q48b = _qslice(qb, offs_m, sqm, sqd, mmask, 32, 16)
            for start_n in range(sn2, sn3, BLOCK_N):
                offs_n = start_n + offs_nb
                nmask = offs_n < N_CTX
                ka = _kslice(kb, offs_n, skn, skd, nmask, 0, 32)
                kb2 = _kslice(kb, offs_n, skn, skd, nmask, 32, 16)
                qk = tl.dot(q32a, tl.trans(ka))
                qk = tl.dot(q48b, tl.trans(kb2), qk)
                m_i, l_i, acc = _osm_tile(qk, m_i, l_i, acc, offs_m, offs_n,
                                          nmask, qk_scale, vb, svn, svd,
                                          HEAD_DIM)
            # --- lvl64 = (0,64) ---
            q64 = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 64)
            for start_n in range(sn3, sn4, BLOCK_N):
                offs_n = start_n + offs_nb
                nmask = offs_n < N_CTX
                kc = _kslice(kb, offs_n, skn, skd, nmask, 0, 64)
                qk = tl.dot(q64, tl.trans(kc))
                m_i, l_i, acc = _osm_tile(qk, m_i, l_i, acc, offs_m, offs_n,
                                          nmask, qk_scale, vb, svn, svd,
                                          HEAD_DIM)
            # --- lvl80 = (0,64)+(64,16) ---
            q64a = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 64)
            q80b = _qslice(qb, offs_m, sqm, sqd, mmask, 64, 16)
            for start_n in range(sn4, sn5, BLOCK_N):
                offs_n = start_n + offs_nb
                nmask = offs_n < N_CTX
                ka = _kslice(kb, offs_n, skn, skd, nmask, 0, 64)
                kb2 = _kslice(kb, offs_n, skn, skd, nmask, 64, 16)
                qk = tl.dot(q64a, tl.trans(ka))
                qk = tl.dot(q80b, tl.trans(kb2), qk)
                m_i, l_i, acc = _osm_tile(qk, m_i, l_i, acc, offs_m, offs_n,
                                          nmask, qk_scale, vb, svn, svd,
                                          HEAD_DIM)
            # --- lvl96 = (0,64)+(64,32) ---
            q64b = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 64)
            q96b = _qslice(qb, offs_m, sqm, sqd, mmask, 64, 32)
            for start_n in range(sn5, sn6, BLOCK_N):
                offs_n = start_n + offs_nb
                nmask = offs_n < N_CTX
                ka = _kslice(kb, offs_n, skn, skd, nmask, 0, 64)
                kb2 = _kslice(kb, offs_n, skn, skd, nmask, 64, 32)
                qk = tl.dot(q64b, tl.trans(ka))
                qk = tl.dot(q96b, tl.trans(kb2), qk)
                m_i, l_i, acc = _osm_tile(qk, m_i, l_i, acc, offs_m, offs_n,
                                          nmask, qk_scale, vb, svn, svd,
                                          HEAD_DIM)
            # --- lvl112 = (0,64)+(64,32)+(96,16) ---
            q64c = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 64)
            q112b = _qslice(qb, offs_m, sqm, sqd, mmask, 64, 32)
            q112c = _qslice(qb, offs_m, sqm, sqd, mmask, 96, 16)
            for start_n in range(sn6, sn7, BLOCK_N):
                offs_n = start_n + offs_nb
                nmask = offs_n < N_CTX
                ka = _kslice(kb, offs_n, skn, skd, nmask, 0, 64)
                kb2 = _kslice(kb, offs_n, skn, skd, nmask, 64, 32)
                kc2 = _kslice(kb, offs_n, skn, skd, nmask, 96, 16)
                qk = tl.dot(q64c, tl.trans(ka))
                qk = tl.dot(q112b, tl.trans(kb2), qk)
                qk = tl.dot(q112c, tl.trans(kc2), qk)
                m_i, l_i, acc = _osm_tile(qk, m_i, l_i, acc, offs_m, offs_n,
                                          nmask, qk_scale, vb, svn, svd,
                                          HEAD_DIM)
            # --- full-D 近傍帯 (diag 含む) ---
            for start_n in range(sn7, hi, BLOCK_N):
                offs_n = start_n + offs_nb
                nmask = offs_n < N_CTX
                kfull = tl.load(kb + offs_n[:, None] * skn
                                + offs_d[None, :] * skd,
                                mask=nmask[:, None], other=0.0)
                qk = tl.dot(qfull, tl.trans(kfull))
                m_i, l_i, acc = _osm_tile(qk, m_i, l_i, acc, offs_m, offs_n,
                                          nmask, qk_scale, vb, svn, svd,
                                          HEAD_DIM)
    else:
        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + offs_nb
            nmask = offs_n < N_CTX
            kfull = tl.load(kb + offs_n[:, None] * skn
                            + offs_d[None, :] * skd,
                            mask=nmask[:, None], other=0.0)
            qk = tl.dot(qfull, tl.trans(kfull))
            m_i, l_i, acc = _osm_tile(qk, m_i, l_i, acc, offs_m, offs_n,
                                      nmask, qk_scale, vb, svn, svd,
                                      HEAD_DIM)

    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    o = acc / l_i[:, None]
    tl.store(ob + offs_m[:, None] * som + offs_d[None, :] * sod,
             o.to(O.dtype.element_ty), mask=mmask[:, None])


_ZHI_CACHE = {}


def _deff_host(wv_cpu_f32, z, head_dim):
    """kernel の _tile_deff と同一 semantics (fp32 wv >= fp32 z)."""
    na = int((wv_cpu_f32 >= float(z)).sum().item())
    d = 2 * na
    d = ((d + 15) // 16) * 16
    return max(16, min(d, head_dim))


def _zhi_tensor(wvec, head_dim, n_ctx):
    """レベル e=16*(i+1) 毎の zhi_e = max{z>=0 int: deff(z)>=e} (二分探索)."""
    key = (wvec.data_ptr(), wvec.numel(), head_dim, n_ctx)
    t = _ZHI_CACHE.get(key)
    if t is not None:
        return t
    wv = wvec.detach().float().cpu()
    out = []
    for i in range(head_dim // 16):
        e = 16 * (i + 1)
        if _deff_host(wv, n_ctx, head_dim) >= e:
            out.append(n_ctx)
            continue
        lo, hi = 0, n_ctx  # deff(0)=head_dim >= e は常に真
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _deff_host(wv, mid, head_dim) >= e:
                lo = mid
            else:
                hi = mid - 1
        out.append(lo)
    t = torch.tensor(out, dtype=torch.int32, device=wvec.device)
    _ZHI_CACHE[key] = t
    return t


def window_attn_prefill(q, k, v, wvec, *, windowed=True, block_m=64,
                        block_n=64, block_d=32, num_warps=4, num_stages=2):
    b, hq, M, d = q.shape
    _, hkv, N, _ = k.shape
    assert M == N and d in (64, 128)  # 実モデル head_dim; static 展開対象
    zhi = _zhi_tensor(wvec, d, N) if windowed else \
        torch.zeros(d // 16, dtype=torch.int32, device=q.device)
    o = torch.empty_like(q)
    grid = (triton.cdiv(M, block_m), b * hq)
    _prefill_fwd[grid](
        q, k, v, o, zhi, 1.0 / math.sqrt(d),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        HQ=hq, GROUP=hq // hkv, N_CTX=N, P=d // 2, HEAD_DIM=d,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d,
        WINDOWED=windowed, num_warps=num_warps, num_stages=num_stages)
    return o
