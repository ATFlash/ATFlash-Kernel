# -*- coding: utf-8 -*-
"""Torch reference implementation of the dim-prefix window.

A banded full-softmax reference, written for clarity rather than speed: for each
query tile it builds the score block [hq, BM, n_hi] from the banded matmuls of
the deff segments (an inner product over the prefix dimensions only, which is
the window), then applies an ordinary softmax and P @ V.

    mode="fp32"  every operation in fp32 -- this is the correctness baseline.
    mode="bf16"  matmuls in bf16 (fp32 accumulate, bf16 rounded output), softmax
                 in fp32, P cast to bf16 before PV. This mirrors the arithmetic
                 of a real kernel and serves as the error anchor for one.

The window is tile-quantized by (block_m, block_n) using exactly the definition
in window_table, so this reference and a kernel can be compared directly.

    phase="prefill"  M == N, causal.
    phase="decode"   M == 1, the query sits at position N-1 and sees every key.
"""
import math

import torch

from . import window_table as WT


def ref_window_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    wvec: torch.Tensor, *, phase: str, block_m: int, block_n: int,
                    mode: str = "fp32") -> torch.Tensor:
    """Windowed attention, computed tile by tile from the window table.

    Args:
        q: (b, hq, M, d) bf16 queries, already in the wavelength-sorted pair layout.
        k, v: (b, hkv, N, d) bf16 keys and values (hq must be a multiple of hkv).
        wvec: per-pair window widths from window_table.wavelengths_sorted.
        phase: "prefill" (M == N, causal) or "decode" (M == 1, sees all N keys).
        block_m, block_n: the tile shape whose quantization the kernel uses.
        mode: "fp32" (correctness baseline) or "bf16" (kernel-like error anchor).

    Returns:
        (b, hq, M, d), fp32 for mode="fp32" and bf16 otherwise.
    """
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
