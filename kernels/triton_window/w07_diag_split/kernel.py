# -*- coding: utf-8 -*-
"""w07_diag_split: causal/nmask の mask 特化 (far=無mask, diag のみ mask).

変異点 (w06_level_loops 比・これのみ):
  - far 帯 (deff<HEAD_DIM の全レベル区間) のタイルは partition の性質から
    zmin >= 1 ⟺ start_n + BLOCK_N <= base_m が保証される
    = (a) 全 k 位置 < 全 q 位置 (causal keep 全真) かつ (b) offs_n < base_m <= N_CTX
    (k 行 mask 全真)。よって far 帯では tl.where(causal) と k/v load の
    predicate を完全除去 (全真 predicate の除去 = bit 等価)。
  - full-D 帯と非窓ループは sn_mask = (base_m // BLOCK_N) * BLOCK_N で
    pre-diag (無 mask; sn_mask 以下は同様に全真) / diag (従来 mask) に分割。
    sn_full <= sn_mask <= hi は sn_mask タイルの zmin<=0 ⇒ deff=HEAD_DIM から保証。
  - レベル構造・zhi ホスト前計算・二進スライス dot・online softmax は w06 と同一。
    decode は w04_merge_par をそのまま import (変異 1 点のみ)。
数値: 全真 predicate の除去のみ → w06 と bit 等価を期待 (w01 ref + w06 同一入力
      bit 比較で検証)。
"""
import math

import torch
import triton
import triton.language as tl

from kernel_window.w04_merge_par.kernel import window_attn_decode  # noqa: F401
from kernel_window.w06_level_loops.kernel import _zhi_tensor       # noqa: F401

NEG = tl.constexpr(-1.0e9)
MINIT = tl.constexpr(-1.0e30)
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _osm_far(qk, m_i, l_i, acc, offs_n, qk_scale,
             vb, svn, svd, HEAD_DIM: tl.constexpr):
    """far 帯タイル: causal keep 全真・k 行 mask 全真 → mask なし更新."""
    offs_d = tl.arange(0, HEAD_DIM)
    s2 = qk * qk_scale
    m_new = tl.maximum(m_i, tl.max(s2, 1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(s2 - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None]
    vblk = tl.load(vb + offs_n[:, None] * svn + offs_d[None, :] * svd)
    acc = tl.dot(p.to(vblk.dtype), vblk, acc)
    return m_new, l_i, acc


@triton.jit
def _osm_diag(qk, m_i, l_i, acc, offs_m, offs_n, nmask, qk_scale,
              vb, svn, svd, HEAD_DIM: tl.constexpr):
    """diag 帯タイル: w06 の _osm_tile と同一 (nmask + causal mask)."""
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
def _kslice_nm(kb, offs_n, skn, skd, BASE: tl.constexpr, W: tl.constexpr):
    """far 帯用: 行 mask 全真が保証されるため predicate なし load."""
    offs_c = BASE + tl.arange(0, W)
    return tl.load(kb + offs_n[:, None] * skn + offs_c[None, :] * skd)


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
    base_m = pid_m * BLOCK_M
    sn_mask = (base_m // BLOCK_N) * BLOCK_N  # 最初の diag(要mask) タイル
    sn_mask = tl.minimum(sn_mask, hi)

    if WINDOWED:
        if HEAD_DIM == 64:
            z32 = tl.load(ZHI + 1)
            z48 = tl.load(ZHI + 2)
            z64 = tl.load(ZHI + 3)
            sn1 = (tl.maximum(base_m - z32, 0) // BLOCK_N) * BLOCK_N
            sn2 = (tl.maximum(base_m - z48, 0) // BLOCK_N) * BLOCK_N
            sn3 = (tl.maximum(base_m - z64, 0) // BLOCK_N) * BLOCK_N
            sn1 = tl.minimum(sn1, hi)
            sn2 = tl.minimum(sn2, hi)
            sn3 = tl.minimum(sn3, hi)
            # --- lvl16 (far, 無mask) ---
            q16 = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 16)
            for start_n in range(0, sn1, BLOCK_N):
                offs_n = start_n + offs_nb
                kc = _kslice_nm(kb, offs_n, skn, skd, 0, 16)
                qk = tl.dot(q16, tl.trans(kc))
                m_i, l_i, acc = _osm_far(qk, m_i, l_i, acc, offs_n, qk_scale,
                                         vb, svn, svd, HEAD_DIM)
            # --- lvl32 (far) ---
            q32 = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 32)
            for start_n in range(sn1, sn2, BLOCK_N):
                offs_n = start_n + offs_nb
                kc = _kslice_nm(kb, offs_n, skn, skd, 0, 32)
                qk = tl.dot(q32, tl.trans(kc))
                m_i, l_i, acc = _osm_far(qk, m_i, l_i, acc, offs_n, qk_scale,
                                         vb, svn, svd, HEAD_DIM)
            # --- lvl48 (far) ---
            q32a = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 32)
            q48b = _qslice(qb, offs_m, sqm, sqd, mmask, 32, 16)
            for start_n in range(sn2, sn3, BLOCK_N):
                offs_n = start_n + offs_nb
                ka = _kslice_nm(kb, offs_n, skn, skd, 0, 32)
                kb2 = _kslice_nm(kb, offs_n, skn, skd, 32, 16)
                qk = tl.dot(q32a, tl.trans(ka))
                qk = tl.dot(q48b, tl.trans(kb2), qk)
                m_i, l_i, acc = _osm_far(qk, m_i, l_i, acc, offs_n, qk_scale,
                                         vb, svn, svd, HEAD_DIM)
            # --- full-D pre-diag (無mask) ---
            snf = tl.maximum(sn3, 0)
            for start_n in range(snf, sn_mask, BLOCK_N):
                offs_n = start_n + offs_nb
                kfull = tl.load(kb + offs_n[:, None] * skn
                                + offs_d[None, :] * skd)
                qk = tl.dot(qfull, tl.trans(kfull))
                m_i, l_i, acc = _osm_far(qk, m_i, l_i, acc, offs_n, qk_scale,
                                         vb, svn, svd, HEAD_DIM)
            # --- full-D diag (mask) ---
            snd = tl.maximum(sn_mask, sn3)
            for start_n in range(snd, hi, BLOCK_N):
                offs_n = start_n + offs_nb
                nmask = offs_n < N_CTX
                kfull = tl.load(kb + offs_n[:, None] * skn
                                + offs_d[None, :] * skd,
                                mask=nmask[:, None], other=0.0)
                qk = tl.dot(qfull, tl.trans(kfull))
                m_i, l_i, acc = _osm_diag(qk, m_i, l_i, acc, offs_m, offs_n,
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
                kc = _kslice_nm(kb, offs_n, skn, skd, 0, 16)
                qk = tl.dot(q16, tl.trans(kc))
                m_i, l_i, acc = _osm_far(qk, m_i, l_i, acc, offs_n, qk_scale,
                                         vb, svn, svd, HEAD_DIM)
            # --- lvl32 ---
            q32 = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 32)
            for start_n in range(sn1, sn2, BLOCK_N):
                offs_n = start_n + offs_nb
                kc = _kslice_nm(kb, offs_n, skn, skd, 0, 32)
                qk = tl.dot(q32, tl.trans(kc))
                m_i, l_i, acc = _osm_far(qk, m_i, l_i, acc, offs_n, qk_scale,
                                         vb, svn, svd, HEAD_DIM)
            # --- lvl48 ---
            q32a = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 32)
            q48b = _qslice(qb, offs_m, sqm, sqd, mmask, 32, 16)
            for start_n in range(sn2, sn3, BLOCK_N):
                offs_n = start_n + offs_nb
                ka = _kslice_nm(kb, offs_n, skn, skd, 0, 32)
                kb2 = _kslice_nm(kb, offs_n, skn, skd, 32, 16)
                qk = tl.dot(q32a, tl.trans(ka))
                qk = tl.dot(q48b, tl.trans(kb2), qk)
                m_i, l_i, acc = _osm_far(qk, m_i, l_i, acc, offs_n, qk_scale,
                                         vb, svn, svd, HEAD_DIM)
            # --- lvl64 ---
            q64 = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 64)
            for start_n in range(sn3, sn4, BLOCK_N):
                offs_n = start_n + offs_nb
                kc = _kslice_nm(kb, offs_n, skn, skd, 0, 64)
                qk = tl.dot(q64, tl.trans(kc))
                m_i, l_i, acc = _osm_far(qk, m_i, l_i, acc, offs_n, qk_scale,
                                         vb, svn, svd, HEAD_DIM)
            # --- lvl80 ---
            q64a = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 64)
            q80b = _qslice(qb, offs_m, sqm, sqd, mmask, 64, 16)
            for start_n in range(sn4, sn5, BLOCK_N):
                offs_n = start_n + offs_nb
                ka = _kslice_nm(kb, offs_n, skn, skd, 0, 64)
                kb2 = _kslice_nm(kb, offs_n, skn, skd, 64, 16)
                qk = tl.dot(q64a, tl.trans(ka))
                qk = tl.dot(q80b, tl.trans(kb2), qk)
                m_i, l_i, acc = _osm_far(qk, m_i, l_i, acc, offs_n, qk_scale,
                                         vb, svn, svd, HEAD_DIM)
            # --- lvl96 ---
            q64b = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 64)
            q96b = _qslice(qb, offs_m, sqm, sqd, mmask, 64, 32)
            for start_n in range(sn5, sn6, BLOCK_N):
                offs_n = start_n + offs_nb
                ka = _kslice_nm(kb, offs_n, skn, skd, 0, 64)
                kb2 = _kslice_nm(kb, offs_n, skn, skd, 64, 32)
                qk = tl.dot(q64b, tl.trans(ka))
                qk = tl.dot(q96b, tl.trans(kb2), qk)
                m_i, l_i, acc = _osm_far(qk, m_i, l_i, acc, offs_n, qk_scale,
                                         vb, svn, svd, HEAD_DIM)
            # --- lvl112 ---
            q64c = _qslice(qb, offs_m, sqm, sqd, mmask, 0, 64)
            q112b = _qslice(qb, offs_m, sqm, sqd, mmask, 64, 32)
            q112c = _qslice(qb, offs_m, sqm, sqd, mmask, 96, 16)
            for start_n in range(sn6, sn7, BLOCK_N):
                offs_n = start_n + offs_nb
                ka = _kslice_nm(kb, offs_n, skn, skd, 0, 64)
                kb2 = _kslice_nm(kb, offs_n, skn, skd, 64, 32)
                kc2 = _kslice_nm(kb, offs_n, skn, skd, 96, 16)
                qk = tl.dot(q64c, tl.trans(ka))
                qk = tl.dot(q112b, tl.trans(kb2), qk)
                qk = tl.dot(q112c, tl.trans(kc2), qk)
                m_i, l_i, acc = _osm_far(qk, m_i, l_i, acc, offs_n, qk_scale,
                                         vb, svn, svd, HEAD_DIM)
            # --- full-D pre-diag (無mask) ---
            for start_n in range(sn7, sn_mask, BLOCK_N):
                offs_n = start_n + offs_nb
                kfull = tl.load(kb + offs_n[:, None] * skn
                                + offs_d[None, :] * skd)
                qk = tl.dot(qfull, tl.trans(kfull))
                m_i, l_i, acc = _osm_far(qk, m_i, l_i, acc, offs_n, qk_scale,
                                         vb, svn, svd, HEAD_DIM)
            # --- full-D diag (mask) ---
            snd = tl.maximum(sn_mask, sn7)
            for start_n in range(snd, hi, BLOCK_N):
                offs_n = start_n + offs_nb
                nmask = offs_n < N_CTX
                kfull = tl.load(kb + offs_n[:, None] * skn
                                + offs_d[None, :] * skd,
                                mask=nmask[:, None], other=0.0)
                qk = tl.dot(qfull, tl.trans(kfull))
                m_i, l_i, acc = _osm_diag(qk, m_i, l_i, acc, offs_m, offs_n,
                                          nmask, qk_scale, vb, svn, svd,
                                          HEAD_DIM)
    else:
        # 非窓: pre-diag (無mask) / diag (mask) の 2 分割
        for start_n in range(0, sn_mask, BLOCK_N):
            offs_n = start_n + offs_nb
            kfull = tl.load(kb + offs_n[:, None] * skn
                            + offs_d[None, :] * skd)
            qk = tl.dot(qfull, tl.trans(kfull))
            m_i, l_i, acc = _osm_far(qk, m_i, l_i, acc, offs_n, qk_scale,
                                     vb, svn, svd, HEAD_DIM)
        for start_n in range(sn_mask, hi, BLOCK_N):
            offs_n = start_n + offs_nb
            nmask = offs_n < N_CTX
            kfull = tl.load(kb + offs_n[:, None] * skn
                            + offs_d[None, :] * skd,
                            mask=nmask[:, None], other=0.0)
            qk = tl.dot(qfull, tl.trans(kfull))
            m_i, l_i, acc = _osm_diag(qk, m_i, l_i, acc, offs_m, offs_n,
                                      nmask, qk_scale, vb, svn, svd,
                                      HEAD_DIM)

    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    o = acc / l_i[:, None]
    tl.store(ob + offs_m[:, None] * som + offs_d[None, :] * sod,
             o.to(O.dtype.element_ty), mask=mmask[:, None])


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
