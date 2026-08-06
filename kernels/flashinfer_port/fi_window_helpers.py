# -*- coding: utf-8 -*-
"""fiwin (ATFLASH-B2): FlashInfer single-decode per-frequency prefix window.

Shared helpers: zhi table (vec8 snap), custom JIT module builder, runner.
flashinfer must be the patched shadow copy (PYTHONPATH=agents/agentB/fiwin/pkgs,
FLASHINFER_WORKSPACE_BASE=agents/agentB/fiwin/cache).
zhi semantics = canonical atflash_window (agentA fa-b1, read-only import):
  sorted pair j (desc wavelength) occupies dims (2j, 2j+1); pair active at age z
  iff z <= w_j (fp32). vec8 snap: thread tx owns 8 dims = 4 pairs
  [4*tx, 4*tx+4); slice active iff z <= floor(w[4*tx]) (max w in slice
  = snap-UP inclusion, finer than FA4-B1's 16-elem snap).
"""
import math
import os
import sys

import torch

AGENTA_CUTE = "/home/USER/atflash/agents/agentA/fa-b1/flash_attn/cute"
if AGENTA_CUTE not in sys.path:
    sys.path.insert(0, AGENTA_CUTE)  # read-only import of canonical window table
from atflash_window import wavelengths_sorted  # noqa: E402

INT32_MAX = 2**31 - 1
VEC_SIZE = 8  # bf16: 16B / 2B


def zhi_slice_vec8(head_dim: int, rope_base: float, k_ret: float, device="cuda:0"):
    """int32 [bdx] table: max age for which thread tx's 8-dim slice is active."""
    wvec = wavelengths_sorted(head_dim, rope_base, k_ret)  # [P] desc, fp32 values
    bdx = head_dim // VEC_SIZE
    pairs_per_thread = VEC_SIZE // 2
    out = []
    for tx in range(bdx):
        w = wvec[tx * pairs_per_thread]  # max wavelength in slice (desc order)
        out.append(INT32_MAX if math.isinf(w) else min(INT32_MAX, int(math.floor(w))))
    return torch.tensor(out, dtype=torch.int32, device=device)


def theory_s_vec8(head_dim: int, rope_base: float, k_ret: float, n_ctx: int):
    """(s_bar, ceiling): mean K-read fraction under vec8 snap and 2/(1+s) ceiling."""
    zhi = zhi_slice_vec8(head_dim, rope_base, k_ret, device="cpu").tolist()
    bdx = head_dim // VEC_SIZE
    total = 0
    for tx in range(bdx):
        # slice tx is read for ages 0..min(zhi, n_ctx-1) -> count of positions
        total += min(zhi[tx] + 1, n_ctx)
    s_bar = total / float(bdx * n_ctx)
    return s_bar, 2.0 / (1.0 + s_bar)


_MOD_CACHE = {}


def build_atfwin_module(dtype: torch.dtype, head_dim: int):
    key = (dtype, head_dim)
    if key in _MOD_CACHE:
        return _MOD_CACHE[key]
    from flashinfer.jit.attention import gen_customize_single_decode_module

    uri = f"single_decode_atfwin_dtype_{str(dtype).split('.')[-1]}_hd{head_dim}"
    mod = gen_customize_single_decode_module(
        uri,
        dtype,  # dtype_q
        dtype,  # dtype_kv
        dtype,  # dtype_o
        head_dim,
        head_dim,
        ["maybe_alibi_slopes", "maybe_zhi_slice"],  # additional_tensor_names
        ["float", "int32_t"],  # additional_tensor_dtypes
        ["logits_soft_cap", "sm_scale", "rope_rcp_scale", "rope_rcp_theta"],
        ["double", "double", "double", "double"],
        "DefaultAttention<false, false, false, false>",
        "#include<flashinfer/attention/variants.cuh>",
        pos_encoding_mode=0,
        use_sliding_window=False,
        use_logits_soft_cap=False,
    ).build_and_load()
    _MOD_CACHE[key] = mod
    return mod


def run_atfwin(mod, q, k, v, zhi_slice=None, sm_scale=None):
    """Custom-module decode. zhi_slice=None => window OFF (stock-identical path)."""
    from flashinfer.decode import single_decode_with_kv_cache_with_jit_module

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    return single_decode_with_kv_cache_with_jit_module(
        mod, q, k, v,
        None,          # maybe_alibi_slopes
        zhi_slice,     # maybe_zhi_slice (int32 [bdx] or None)
        0.0,           # logits_soft_cap
        float(sm_scale),
        1.0,           # rope_rcp_scale (unused, kNone)
        1.0 / 1e4,     # rope_rcp_theta (unused, kNone)
    )


def make_case(model: str, kv: int, device="cuda:0", seed=1234):
    cfg = {"llama": (24, 8, 128, 500000.0), "qwen": (14, 2, 64, 1000000.0)}[model]
    hq, hkv, d, base = cfg
    g = torch.Generator(device="cpu").manual_seed(seed + kv + d)
    q = torch.randn(hq, d, generator=g, dtype=torch.float32).to(torch.bfloat16).to(device)
    k = torch.randn(kv, hkv, d, generator=g, dtype=torch.float32).to(torch.bfloat16).to(device)
    v = torch.randn(kv, hkv, d, generator=g, dtype=torch.float32).to(torch.bfloat16).to(device)
    return q, k, v, base
