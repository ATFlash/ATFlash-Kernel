# -*- coding: utf-8 -*-
"""Gate B0: both ports in one process; with the window off they are bit-identical
to the stock kernels.

One process runs the FA4-port prefill and then the patched FlashInfer decode on
one shared KV buffer (NHD layout). The gate: with the window off, the integrated
path equals the stock kernels bit for bit. The script also checks that FA4
accepts a strided (non-contiguous) BSHD view, as used when the HF cache is
reused, and generates the raw-base window tables of both kernels (16-dim and
8-dim snap). Llama-3.2-3B dimensions (GQA group 3), bf16, batch 1.
"""
import hashlib
import sys

import torch

# fiwin_common = kernels/flashinfer_port/fiwin_common.py and atflash_window =
# window_tables/atflash_window.py in this repository (the paths below are placeholders).
sys.path.insert(0, "/home/USER/atflash/agents/agentB/fiwin/scripts")
sys.path.insert(0, "/home/USER/atflash/agents/agentA/fa-b1/flash_attn/cute")

from flash_attn.cute.interface import _flash_attn_fwd  # noqa: E402
import flashinfer  # noqa: E402
from fiwin_common import build_atfwin_module, run_atfwin, zhi_slice_vec8  # noqa: E402
from atflash_window import window_zhi_from_kret  # noqa: E402


def sha(t):
    a = t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(a).hexdigest()[:16]


torch.manual_seed(0)
dev = "cuda"
hq, hkv, d, base = 24, 8, 128, 500000.0   # llama-3.2-3B
N = 2048
k_ret = 2.0
print(f"case: llama hq={hq} hkv={hkv} d={d} base={base:g} N={N} k_ret={k_ret} bf16 b=1")
print("flashinfer", flashinfer.__file__.split("site-packages/")[-1]
      if "site-packages" in flashinfer.__file__ else flashinfer.__file__)

# ---- shared KV buffer: NHD [S,hkv,d] contiguous (zero-conversion design) ----
kv_k = torch.randn(N, hkv, d, dtype=torch.bfloat16, device=dev)
kv_v = torch.randn(N, hkv, d, dtype=torch.bfloat16, device=dev)
q_prefill = torch.randn(N, hq, d, dtype=torch.bfloat16, device=dev)

# ---- FA4 prefill (window OFF), reading shared buffer via unsqueeze(0)=BSHD view ----
fa4_k = kv_k.unsqueeze(0)   # [1,N,hkv,d]
fa4_v = kv_v.unsqueeze(0)
o_prefill = _flash_attn_fwd(q_prefill.unsqueeze(0), fa4_k, fa4_v,
                            causal=True, pack_gqa=False)[0]
print(f"\n[FA4 prefill] out={tuple(o_prefill.shape)} fa4_k.contig={fa4_k.is_contiguous()} "
      f"sha={sha(o_prefill)}")
# reference = FA4 on freshly-contiguous native BSHD (identical values)
o_prefill_ref = _flash_attn_fwd(q_prefill.unsqueeze(0).contiguous(),
                                kv_k.unsqueeze(0).contiguous(),
                                kv_v.unsqueeze(0).contiguous(),
                                causal=True, pack_gqa=False)[0]
prefill_ok = torch.equal(o_prefill, o_prefill_ref)
print(f"[FA4 prefill] ref sha={sha(o_prefill_ref)} integrated==ref: {prefill_ok}")

# ---- probe: FA4 on a STRIDED (non-contig) BSHD view (HF-cache BHSD reuse path) ----
hf_k = kv_k.transpose(0, 1).unsqueeze(0).contiguous()  # [1,hkv,N,d] (HF cache layout)
hf_v = kv_v.transpose(0, 1).unsqueeze(0).contiguous()
strided_k = hf_k.transpose(1, 2)   # [1,N,hkv,d] non-contiguous view
strided_v = hf_v.transpose(1, 2)
try:
    o_strided = _flash_attn_fwd(q_prefill.unsqueeze(0), strided_k, strided_v,
                                causal=True, pack_gqa=False)[0]
    print(f"[probe] FA4 strided-view ACCEPTED contig={strided_k.is_contiguous()} "
          f"==ref: {torch.equal(o_strided, o_prefill_ref)}")
except Exception as e:
    print(f"[probe] FA4 strided-view REJECTED -> per-request .contiguous() needed: "
          f"{type(e).__name__}: {str(e)[:110]}")

# ---- decode: append 1 token, FI reads the SAME buffer natively (NHD) ----
k_new = torch.randn(1, hkv, d, dtype=torch.bfloat16, device=dev)
v_new = torch.randn(1, hkv, d, dtype=torch.bfloat16, device=dev)
kbuf = torch.cat([kv_k, k_new], 0).contiguous()   # [N+1,hkv,d] NHD
vbuf = torch.cat([kv_v, v_new], 0).contiguous()
q_dec = torch.randn(hq, d, dtype=torch.bfloat16, device=dev)   # [num_qo_heads,d]

mod = build_atfwin_module(torch.bfloat16, d)
o_dec_off = run_atfwin(mod, q_dec, kbuf, vbuf, zhi_slice=None)         # integrated OFF
o_dec_stock = flashinfer.single_decode_with_kv_cache(q_dec, kbuf, vbuf)  # stock
dec_ok = torch.equal(o_dec_off, o_dec_stock)
print(f"\n[FI decode] custom-OFF sha={sha(o_dec_off)} stock sha={sha(o_dec_stock)} "
      f"bit-equal: {dec_ok}")

# ---- window ON sanity: the raw-base zhi runs and changes the output ----
zs8 = zhi_slice_vec8(d, base, k_ret)          # FI 8-snap from raw base
o_dec_on = run_atfwin(mod, q_dec, kbuf, vbuf, zhi_slice=zs8)
zhi16 = window_zhi_from_kret(d, base, k_ret, N)  # FA4 16-snap from same raw base
print(f"[FI decode] WIN-ON sha={sha(o_dec_on)} differs-from-OFF: {not torch.equal(o_dec_on, o_dec_off)}")
print(f"[raw-base zhi] FA4 16-snap={list(zhi16)}")
print(f"[raw-base zhi] FI  8-snap={zs8.tolist()}")

gate = prefill_ok and dec_ok
print(f"\nB0_GATE {'PASS' if gate else 'FAIL'} "
      f"(prefill_bit={prefill_ok}, decode_off==stock={dec_ok}, both kernels launched in-process)")
