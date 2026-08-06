# -*- coding: utf-8 -*-
"""w03_fulld_fastpath: w02 + 「deff == HEAD_DIM タイルは full-D 単発 tl.dot」fast path.

変異点 (w02 比・これのみ):
  - q タイルを常時レジスタ常駐でロードし、窓ありでも deff==HEAD_DIM の近傍タイルは
    chunk ループ/再ロードなしの単発 tl.dot(qfull, k^T)。遠帯のみ BLOCK_D chunk。
  - prefill / decode split 両方に適用。数値経路は同一 (dot 縮約順のみ; bf16->fp32 accum)。
窓定義・16 倍数制約・online softmax は w02 と同一。
"""
import math

import torch
import triton
import triton.language as tl

NEG = tl.constexpr(-1.0e9)
MINIT = tl.constexpr(-1.0e30)
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _tile_deff(zmin_f, wv, HEAD_DIM: tl.constexpr):
    na = tl.sum((wv >= zmin_f).to(tl.int32), axis=0)
    deff = 2 * na
    deff = ((deff + 15) // 16) * 16
    deff = tl.maximum(deff, 16)
    deff = tl.minimum(deff, HEAD_DIM)
    return deff


@triton.jit
def _prefill_fwd(Q, K, V, O, Wv, sm_scale,
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
    wv = tl.load(Wv + tl.arange(0, P))
    qfull = tl.load(qb + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
                    mask=mmask[:, None], other=0.0)

    hi = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + offs_nb
        nmask = offs_n < N_CTX
        if WINDOWED:
            zmin = pid_m * BLOCK_M - (start_n + BLOCK_N - 1)
            zmin = tl.maximum(zmin, 0)
            deff = _tile_deff(zmin.to(tl.float32), wv, HEAD_DIM)
            if deff == HEAD_DIM:
                kfull = tl.load(kb + offs_n[:, None] * skn
                                + offs_d[None, :] * skd,
                                mask=nmask[:, None], other=0.0)
                qk = tl.dot(qfull, tl.trans(kfull))
            else:
                qk = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
                for d0 in range(0, deff, BLOCK_D):
                    offs_dd = d0 + tl.arange(0, BLOCK_D)
                    dmask = offs_dd < deff
                    qc = tl.load(qb + offs_m[:, None] * sqm
                                 + offs_dd[None, :] * sqd,
                                 mask=mmask[:, None] & dmask[None, :],
                                 other=0.0)
                    kc = tl.load(kb + offs_n[:, None] * skn
                                 + offs_dd[None, :] * skd,
                                 mask=nmask[:, None] & dmask[None, :],
                                 other=0.0)
                    qk = tl.dot(qc, tl.trans(kc), qk)
        else:
            kfull = tl.load(kb + offs_n[:, None] * skn + offs_d[None, :] * skd,
                            mask=nmask[:, None], other=0.0)
            qk = tl.dot(qfull, tl.trans(kfull))

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
        acc = tl.dot(p.to(V.dtype.element_ty), vblk, acc)
        m_i = m_new

    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    o = acc / l_i[:, None]
    tl.store(ob + offs_m[:, None] * som + offs_d[None, :] * sod,
             o.to(O.dtype.element_ty), mask=mmask[:, None])


@triton.jit
def _decode_split(Q, K, V, Mp, Lp, Op, Wv, sm_scale,
                  sqb, sqh, sqd,
                  skb, skh, skn, skd,
                  svb, svh, svn, svd,
                  smp_bh, smp_s, sop_bh, sop_s, sop_g,
                  HKV: tl.constexpr, GROUP: tl.constexpr,
                  N_CTX, TILES_PER_SPLIT,
                  P: tl.constexpr, HEAD_DIM: tl.constexpr,
                  BLOCK_G: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_D: tl.constexpr, WINDOWED: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    off_b = pid_bh // HKV
    off_kh = pid_bh % HKV
    offs_g = tl.arange(0, BLOCK_G)
    gmask = offs_g < GROUP
    qrow = Q + off_b * sqb + (off_kh * GROUP + offs_g) * sqh
    kb = K + off_b * skb + off_kh * skh
    vb = V + off_b * svb + off_kh * svh
    offs_nb = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    m_i = tl.full([BLOCK_G], MINIT, tl.float32)
    l_i = tl.zeros([BLOCK_G], tl.float32)
    acc = tl.zeros([BLOCK_G, HEAD_DIM], tl.float32)
    qk_scale = sm_scale * LOG2E
    wv = tl.load(Wv + tl.arange(0, P))
    qfull = tl.load(qrow[:, None] + offs_d[None, :] * sqd,
                    mask=gmask[:, None], other=0.0)

    ntiles = tl.cdiv(N_CTX, BLOCK_N)
    t0 = pid_s * TILES_PER_SPLIT
    t1 = tl.minimum(t0 + TILES_PER_SPLIT, ntiles)

    for t in range(t0, t1):
        start_n = t * BLOCK_N
        offs_n = start_n + offs_nb
        nmask = offs_n < N_CTX
        if WINDOWED:
            zmin = (N_CTX - 1) - (start_n + BLOCK_N - 1)
            zmin = tl.maximum(zmin, 0)
            deff = _tile_deff(zmin.to(tl.float32), wv, HEAD_DIM)
            if deff == HEAD_DIM:
                kfull = tl.load(kb + offs_n[:, None] * skn
                                + offs_d[None, :] * skd,
                                mask=nmask[:, None], other=0.0)
                qk = tl.dot(qfull, tl.trans(kfull))
            else:
                qk = tl.zeros([BLOCK_G, BLOCK_N], tl.float32)
                for d0 in range(0, deff, BLOCK_D):
                    offs_dd = d0 + tl.arange(0, BLOCK_D)
                    dmask = offs_dd < deff
                    qc = tl.load(qrow[:, None] + offs_dd[None, :] * sqd,
                                 mask=gmask[:, None] & dmask[None, :],
                                 other=0.0)
                    kc = tl.load(kb + offs_n[:, None] * skn
                                 + offs_dd[None, :] * skd,
                                 mask=nmask[:, None] & dmask[None, :],
                                 other=0.0)
                    qk = tl.dot(qc, tl.trans(kc), qk)
        else:
            kfull = tl.load(kb + offs_n[:, None] * skn + offs_d[None, :] * skd,
                            mask=nmask[:, None], other=0.0)
            qk = tl.dot(qfull, tl.trans(kfull))

        s2 = tl.where(nmask[None, :], qk * qk_scale, NEG)
        m_new = tl.maximum(m_i, tl.max(s2, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s2 - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        vblk = tl.load(vb + offs_n[:, None] * svn + offs_d[None, :] * svd,
                       mask=nmask[:, None], other=0.0)
        acc = tl.dot(p.to(V.dtype.element_ty), vblk, acc)
        m_i = m_new

    tl.store(Mp + pid_bh * smp_bh + pid_s * smp_s + offs_g, m_i)
    tl.store(Lp + pid_bh * smp_bh + pid_s * smp_s + offs_g, l_i)
    tl.store(Op + pid_bh * sop_bh + pid_s * sop_s + offs_g[:, None] * sop_g
             + offs_d[None, :], acc)


@triton.jit
def _decode_merge(Mp, Lp, Op, O,
                  smp_bh, smp_s, sop_bh, sop_s, sop_g,
                  sob, soh, sod,
                  HKV: tl.constexpr, GROUP: tl.constexpr, S,
                  HEAD_DIM: tl.constexpr, BLOCK_G: tl.constexpr):
    pid_bh = tl.program_id(0)
    off_b = pid_bh // HKV
    off_kh = pid_bh % HKV
    offs_g = tl.arange(0, BLOCK_G)
    offs_d = tl.arange(0, HEAD_DIM)
    gmask = offs_g < GROUP

    m = tl.full([BLOCK_G], MINIT, tl.float32)
    l = tl.zeros([BLOCK_G], tl.float32)
    acc = tl.zeros([BLOCK_G, HEAD_DIM], tl.float32)
    for s in range(0, S):
        mp = tl.load(Mp + pid_bh * smp_bh + s * smp_s + offs_g)
        lp = tl.load(Lp + pid_bh * smp_bh + s * smp_s + offs_g)
        op = tl.load(Op + pid_bh * sop_bh + s * sop_s
                     + offs_g[:, None] * sop_g + offs_d[None, :])
        m_new = tl.maximum(m, mp)
        a1 = tl.exp2(m - m_new)
        a2 = tl.exp2(mp - m_new)
        l = l * a1 + lp * a2
        acc = acc * a1[:, None] + op * a2[:, None]
        m = m_new
    l = tl.where(l == 0.0, 1.0, l)
    o = acc / l[:, None]
    tl.store(O + off_b * sob + (off_kh * GROUP + offs_g[:, None]) * soh
             + offs_d[None, :] * sod, o.to(O.dtype.element_ty),
             mask=gmask[:, None])


def window_attn_prefill(q, k, v, wvec, *, windowed=True, block_m=64,
                        block_n=64, block_d=32, num_warps=4, num_stages=2):
    b, hq, M, d = q.shape
    _, hkv, N, _ = k.shape
    assert M == N and d in (32, 64, 128, 256)
    o = torch.empty_like(q)
    grid = (triton.cdiv(M, block_m), b * hq)
    _prefill_fwd[grid](
        q, k, v, o, wvec, 1.0 / math.sqrt(d),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        HQ=hq, GROUP=hq // hkv, N_CTX=N, P=d // 2, HEAD_DIM=d,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d,
        WINDOWED=windowed, num_warps=num_warps, num_stages=num_stages)
    return o


def window_attn_decode(q, k, v, wvec, *, windowed=True, splits=None,
                       block_n=128, block_d=32, num_warps=4, num_stages=2,
                       sm_target=512):
    b, hq, _, d = q.shape
    _, hkv, N, _ = k.shape
    group = hq // hkv
    assert group <= 16
    block_g = 16
    ntiles = triton.cdiv(N, block_n)
    if splits is None:
        splits = min(ntiles, max(1, (sm_target + b * hkv - 1) // (b * hkv)))
    tiles_per_split = triton.cdiv(ntiles, splits)
    splits = triton.cdiv(ntiles, tiles_per_split)

    o = torch.empty_like(q)
    mp = torch.empty((b * hkv, splits, block_g), dtype=torch.float32,
                     device=q.device)
    lp = torch.empty_like(mp)
    op = torch.empty((b * hkv, splits, block_g, d), dtype=torch.float32,
                     device=q.device)
    q3 = q.view(b, hq, d)
    grid1 = (b * hkv, splits)
    _decode_split[grid1](
        q3, k, v, mp, lp, op, wvec, 1.0 / math.sqrt(d),
        q3.stride(0), q3.stride(1), q3.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        mp.stride(0), mp.stride(1), op.stride(0), op.stride(1), op.stride(2),
        HKV=hkv, GROUP=group, N_CTX=N, TILES_PER_SPLIT=tiles_per_split,
        P=d // 2, HEAD_DIM=d, BLOCK_G=block_g, BLOCK_N=block_n,
        BLOCK_D=block_d, WINDOWED=windowed,
        num_warps=num_warps, num_stages=num_stages)
    grid2 = (b * hkv,)
    _decode_merge[grid2](
        mp, lp, op, o.view(b, hq, d),
        mp.stride(0), mp.stride(1), op.stride(0), op.stride(1), op.stride(2),
        o.view(b, hq, d).stride(0), o.view(b, hq, d).stride(1),
        o.view(b, hq, d).stride(2),
        HKV=hkv, GROUP=group, S=splits, HEAD_DIM=d, BLOCK_G=block_g,
        num_warps=4)
    return o
