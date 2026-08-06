# -*- coding: utf-8 -*-
"""ATFlash 路線(i) — torch 参照実装 (w01): dim-prefix マスクの素直な実装.

banded full-softmax reference:
  q タイル毎に S 行ブロック [hq, BM, n_hi] を deff 区間の banded matmul で構築し
  (prefix 次元のみの内積 = dim-prefix マスク), 通常の softmax -> P @ V。
  mode="fp32": 全演算 fp32 (正しさの基準)。
  mode="bf16": matmul を bf16 で行い (fp32 accum, 出力 bf16 丸め), softmax は fp32,
               P を bf16 に落として PV。Triton kernel の算術を模倣する誤差アンカー。

窓は (block_m, block_n) タイル量子化 — kernel と同一定義 (window_table 参照)。
phase="prefill": M==N, causal。 phase="decode": M==1, query 位置 = N-1, 全 key 参照。
"""
import math

import torch

from . import window_table as WT


def ref_window_attn(q, k, v, wvec, *, phase: str, block_m: int, block_n: int,
                    mode: str = "fp32"):
    """q:(b,hq,M,d) k,v:(b,hkv,N,d) bf16. 戻り値 (b,hq,M,d) fp32 (mode=fp32) / bf16."""
    assert mode in ("fp32", "bf16")
    assert phase in ("prefill", "decode")
    b, hq, M, d = q.shape
    _, hkv, N, _ = k.shape
    G = hq // hkv
    scale = 1.0 / math.sqrt(d)
    head_map = torch.arange(hq, device=q.device) // G  # q head -> kv head
    out_dtype = torch.float32 if mode == "fp32" else torch.bfloat16
    out = torch.empty((b, hq, M, d), dtype=out_dtype, device=q.device)

    for bi in range(b):
        kf = k[bi].float() if mode == "fp32" else k[bi]
        vf = v[bi].float() if mode == "fp32" else v[bi]
        kx = kf[head_map]                       # (hq, N, d) GQA expand (gather copy)
        vx = vf[head_map]
        qf = q[bi].float() if mode == "fp32" else q[bi]
        for q0 in range(0, M, block_m):
            qm = min(block_m, M - q0)
            n_hi = min(q0 + block_m, N) if phase == "prefill" else N
            qt = qf[:, q0:q0 + qm, :]           # (hq, qm, d)
            S = torch.empty((hq, qm, n_hi), dtype=torch.float32, device=q.device)
            for (n0, n1, deff) in WT.deff_segments(
                    phase, q0, n_hi, block_m, block_n, N, wvec, d):
                s = torch.matmul(qt[:, :, :deff],
                                 kx[:, n0:n1, :deff].transpose(-1, -2))
                S[:, :, n0:n1] = s.float() * scale
            if phase == "prefill":
                rows = torch.arange(q0, q0 + qm, device=q.device)
                cols = torch.arange(0, n_hi, device=q.device)
                S.masked_fill_((cols[None, :] > rows[:, None]).unsqueeze(0),
                               float("-inf"))
            P = torch.softmax(S, dim=-1)
            if mode == "bf16":
                o = torch.matmul(P.to(torch.bfloat16), vx[:, :n_hi, :])
            else:
                o = torch.matmul(P, vx[:, :n_hi, :])
            out[bi, :, q0:q0 + qm, :] = o.to(out_dtype)
    return out
