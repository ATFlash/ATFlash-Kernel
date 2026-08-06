# -*- coding: utf-8 -*-
"""w05_static_slice: prefill 遠帯 chunk ループ → deff 離散値毎 static-slice dot.

変異点 (w03_fulld_fastpath 比・これのみ):
  - prefill 窓あり遠帯 (deff < HEAD_DIM) の qk 計算を、動的 `range(0, deff, BLOCK_D)`
    ループ + dmask 付き q/k 再ロードから、deff の離散値 (16 の倍数; _tile_deff が保証)
    への runtime 分岐 + 二進スライス分解 (64/32/16) の固定形 tl.dot 連鎖へ置換。
    * q スライス = register 常駐 qfull の reshape/permute/tl.split 木
      (k タイルループ外で 1 回構築、グローバル再ロードなし)。
    * k ロード = constexpr 幅 (16/32/64) の非 mask 列 load (deff は 16 の倍数なので
      端数 mask 不要)。
    * dot 縮約幅は全て 16 の倍数 (MMA K-atom 制約)。qk への fp32 累積は次元昇順。
  - decode は w04_merge_par をそのまま import (変異 1 点のみ)。
数値: 遠帯 qk の fp32 累積分割が BLOCK_D 均等 chunk → 二進スライス (昇順) に変わる
      (w03 と bit 等価は保証しない → w01 ref + bf16 mirror anchor で正しさ測定)。
      窓定義 (_tile_deff)・deff==HEAD_DIM full-D fast path・online softmax・PV dot は
      w03 と同一。HEAD_DIM ∈ {64, 128} (実モデル qwen/llama) を static 展開。
"""
import math

import torch
import triton
import triton.language as tl

from kernel_window.w03_fulld_fastpath.kernel import _tile_deff  # noqa: F401
from kernel_window.w04_merge_par.kernel import window_attn_decode  # noqa: F401

NEG = tl.constexpr(-1.0e9)
MINIT = tl.constexpr(-1.0e30)
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _half(x, BM: tl.constexpr, W: tl.constexpr):
    """[BM, W] -> 連続する前半/後半 [BM, W//2] (register 再解釈のみ)."""
    xr = tl.reshape(x, (BM, 2, W // 2))
    xp = tl.permute(xr, (0, 2, 1))
    a, b = tl.split(xp)
    return a, b


@triton.jit
def _kslice(kb, offs_n, skn, skd, nmask, BASE: tl.constexpr, W: tl.constexpr):
    """k の列 [BASE, BASE+W) を非 mask (行 mask のみ) でロード -> [BLOCK_N, W]."""
    offs_c = BASE + tl.arange(0, W)
    return tl.load(kb + offs_n[:, None] * skn + offs_c[None, :] * skd,
                   mask=nmask[:, None], other=0.0)


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

    # --- q の二進スライス木 (register 再解釈; k タイルループ外で 1 回) ---
    if WINDOWED:
        if HEAD_DIM == 64:
            q32a, q32b = _half(qfull, BLOCK_M, 64)      # [0:32) [32:64)
            q16a, _q16x = _half(q32a, BLOCK_M, 32)      # [0:16)
            q16c, _q16y = _half(q32b, BLOCK_M, 32)      # [32:48)
        if HEAD_DIM == 128:
            q64a, q64b = _half(qfull, BLOCK_M, 128)     # [0:64) [64:128)
            q32_0, q32_1 = _half(q64a, BLOCK_M, 64)     # [0:32) [32:64)
            q32_2, q32_3 = _half(q64b, BLOCK_M, 64)     # [64:96) [96:128)
            q16_0, _a = _half(q32_0, BLOCK_M, 32)       # [0:16)
            q16_2, _b = _half(q32_1, BLOCK_M, 32)       # [32:48)
            q16_4, _c = _half(q32_2, BLOCK_M, 32)       # [64:80)
            q16_6, _d = _half(q32_3, BLOCK_M, 32)       # [96:112)

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
                # deff ∈ {16,32,...,HEAD_DIM-16}: 二進スライスの固定形 dot
                # 注: Triton の runtime 分岐は then/else で同名変数の型一致を
                # 要求するため、k ロードは branch 毎に一意な名前にする。
                if HEAD_DIM == 64:
                    if deff == 48:
                        ka48 = _kslice(kb, offs_n, skn, skd, nmask, 0, 32)
                        kb48 = _kslice(kb, offs_n, skn, skd, nmask, 32, 16)
                        qk = tl.dot(q32a, tl.trans(ka48))
                        qk = tl.dot(q16c, tl.trans(kb48), qk)
                    elif deff == 32:
                        ka32 = _kslice(kb, offs_n, skn, skd, nmask, 0, 32)
                        qk = tl.dot(q32a, tl.trans(ka32))
                    else:  # 16
                        ka16 = _kslice(kb, offs_n, skn, skd, nmask, 0, 16)
                        qk = tl.dot(q16a, tl.trans(ka16))
                else:  # HEAD_DIM == 128
                    if deff == 112:
                        ka112 = _kslice(kb, offs_n, skn, skd, nmask, 0, 64)
                        kb112 = _kslice(kb, offs_n, skn, skd, nmask, 64, 32)
                        kc112 = _kslice(kb, offs_n, skn, skd, nmask, 96, 16)
                        qk = tl.dot(q64a, tl.trans(ka112))
                        qk = tl.dot(q32_2, tl.trans(kb112), qk)
                        qk = tl.dot(q16_6, tl.trans(kc112), qk)
                    elif deff == 96:
                        ka96 = _kslice(kb, offs_n, skn, skd, nmask, 0, 64)
                        kb96 = _kslice(kb, offs_n, skn, skd, nmask, 64, 32)
                        qk = tl.dot(q64a, tl.trans(ka96))
                        qk = tl.dot(q32_2, tl.trans(kb96), qk)
                    elif deff == 80:
                        ka80 = _kslice(kb, offs_n, skn, skd, nmask, 0, 64)
                        kb80 = _kslice(kb, offs_n, skn, skd, nmask, 64, 16)
                        qk = tl.dot(q64a, tl.trans(ka80))
                        qk = tl.dot(q16_4, tl.trans(kb80), qk)
                    elif deff == 64:
                        ka64 = _kslice(kb, offs_n, skn, skd, nmask, 0, 64)
                        qk = tl.dot(q64a, tl.trans(ka64))
                    elif deff == 48:
                        ka48 = _kslice(kb, offs_n, skn, skd, nmask, 0, 32)
                        kb48 = _kslice(kb, offs_n, skn, skd, nmask, 32, 16)
                        qk = tl.dot(q32_0, tl.trans(ka48))
                        qk = tl.dot(q16_2, tl.trans(kb48), qk)
                    elif deff == 32:
                        ka32 = _kslice(kb, offs_n, skn, skd, nmask, 0, 32)
                        qk = tl.dot(q32_0, tl.trans(ka32))
                    else:  # 16
                        ka16 = _kslice(kb, offs_n, skn, skd, nmask, 0, 16)
                        qk = tl.dot(q16_0, tl.trans(ka16))
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


def window_attn_prefill(q, k, v, wvec, *, windowed=True, block_m=64,
                        block_n=64, block_d=32, num_warps=4, num_stages=2):
    b, hq, M, d = q.shape
    _, hkv, N, _ = k.shape
    assert M == N and d in (64, 128)  # 実モデル head_dim; static 展開対象
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
