# -*- coding: utf-8 -*-
"""w04_merge_par: w03 + decode merge の並列化 (これのみが変異点).

- 旧 merge (w03/_decode_merge): 1 program / (b*hkv), S を直列 online-merge
  -> hkv 小 (qwen=2) だと直列律速。
- 新 merge (_decode_merge_v2): grid=(b*hkv, HEAD_DIM/DCHUNK)。
  pass A: m_g = max_s mp (軽い [G] ロードのみ), pass B: l_g = sum lp*exp2(mp-m_g),
  pass C: acc = sum op*exp2(mp-m_g) — 直列依存なしの独立和で load を pipeline 可能。
  数値: 固定順の和 (決定的)。オーバーフロー安全 (m_g global max)。
- split カーネル / prefill は w03 をそのまま import (変異 1 点のみ)。
"""
import math

import torch
import triton
import triton.language as tl

from kernel_window.w03_fulld_fastpath.kernel import (  # noqa: F401
    MINIT, _decode_split, _prefill_fwd, window_attn_prefill)


@triton.jit
def _decode_merge_v2(Mp, Lp, Op, O,
                     smp_bh, smp_s, sop_bh, sop_s, sop_g,
                     sob, soh, sod,
                     HKV: tl.constexpr, GROUP: tl.constexpr, S,
                     HEAD_DIM: tl.constexpr, BLOCK_G: tl.constexpr,
                     DCHUNK: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)
    off_b = pid_bh // HKV
    off_kh = pid_bh % HKV
    offs_g = tl.arange(0, BLOCK_G)
    offs_d = pid_d * DCHUNK + tl.arange(0, DCHUNK)
    gmask = offs_g < GROUP

    m_g = tl.full([BLOCK_G], MINIT, tl.float32)
    for s in range(0, S):
        mp = tl.load(Mp + pid_bh * smp_bh + s * smp_s + offs_g)
        m_g = tl.maximum(m_g, mp)
    l_g = tl.zeros([BLOCK_G], tl.float32)
    for s in range(0, S):
        mp = tl.load(Mp + pid_bh * smp_bh + s * smp_s + offs_g)
        lp = tl.load(Lp + pid_bh * smp_bh + s * smp_s + offs_g)
        l_g += lp * tl.exp2(mp - m_g)
    acc = tl.zeros([BLOCK_G, DCHUNK], tl.float32)
    for s in range(0, S):
        mp = tl.load(Mp + pid_bh * smp_bh + s * smp_s + offs_g)
        op = tl.load(Op + pid_bh * sop_bh + s * sop_s
                     + offs_g[:, None] * sop_g + offs_d[None, :])
        acc += op * tl.exp2(mp - m_g)[:, None]
    l_g = tl.where(l_g == 0.0, 1.0, l_g)
    o = acc / l_g[:, None]
    tl.store(O + off_b * sob + (off_kh * GROUP + offs_g[:, None]) * soh
             + offs_d[None, :] * sod, o.to(O.dtype.element_ty),
             mask=gmask[:, None])


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
    dchunk = min(64, d)
    grid2 = (b * hkv, d // dchunk)
    _decode_merge_v2[grid2](
        mp, lp, op, o.view(b, hq, d),
        mp.stride(0), mp.stride(1), op.stride(0), op.stride(1), op.stride(2),
        o.view(b, hq, d).stride(0), o.view(b, hq, d).stride(1),
        o.view(b, hq, d).stride(2),
        HKV=hkv, GROUP=group, S=splits, HEAD_DIM=d, BLOCK_G=block_g,
        DCHUNK=dchunk, num_warps=2)
    return o
