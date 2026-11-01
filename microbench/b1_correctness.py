#!/usr/bin/env python
"""Correctness of the FA4 prefill window against the banded fp32 reference
(atflash/reference.py).

FA4 with the window port (PYTHONPATH must expose the built port), window on with
k_ret = 2.0, pack_gqa = False so that the tile quantization matches the
reference's block_m / block_n. Tiles are the FA4 sm120 defaults:
d = 128 -> (BM, BN) = (128, 64); d <= 64 -> (128, 128).
Tolerance rule: max|out - ref32| <= 2 * max|ref16 - ref32| + 2e-3.
Grid: Llama (24/8/128, base 5e5) and Qwen (14/2/64, base 1e6), N in
{512, 2048, 8192, 32768}. --real <npz> adds one case on real model q/k (the
pair permutation is applied inside).
FA4 base: PR #2656 head 4f1fde1.
"""
import argparse
import json
import os
import sys

import torch

from atflash import window_table as WT
from atflash.reference import ref_window_attn
from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.atflash_window import window_zhi_from_kret, zhi_table

GRID = [
    ("llama", 24, 8, 128, 500000.0, [512, 2048, 8192, 32768]),
    ("qwen", 14, 2, 64, 1000000.0, [512, 2048, 8192, 32768]),
]
GRID_EXT = [  # sweep extension shapes (Phase 1); N_CAP raised accordingly
    ("llama", 24, 8, 128, 500000.0, [65536, 163840]),
    ("qwen", 14, 2, 64, 1000000.0, [65536]),
]
K_RET = 2.0
N_CAP = 262144  # fixed zhi cap: entries >= n_idx_diag all clamp bnd to 0, so level
# assignment is identical to a per-N table for any N <= N_CAP (one compile per model)


def fast_slice_stats(N, BM, BN, zhi_perN, d):
    """WT.mma_slice_stats (prefill) semantics via the zhi table (O(1) deff lookup).

    deff(z) = 16 * #{i: zhi[i] >= z}  (exact: zhi[i] = max{z: deff(z) >= 16(i+1)}).
    WT.deff_of_zmin per call does a torch reduction -> too slow at 160k (3M+ calls).
    """
    full_slices = d // 16
    tw = tf = kw = kf = 0
    for q0 in range(0, N, BM):
        n_hi = min(q0 + BM, N)
        ntiles = (n_hi + BN - 1) // BN
        for t in range(ntiles):
            zmin = max(0, q0 - (t * BN + BN - 1))
            deff = 16 * sum(1 for z in zhi_perN if z >= zmin)
            tw += deff // 16
            tf += full_slices
            kw += deff
            kf += d
    return {"qk_slice_ratio": tw / max(1, tf), "kload_dim_ratio": kw / max(1, kf)}


def fa4_tiles(d):
    return (128, 64) if d > 64 else (128, 128)


def run_case(name, hq, hkv, d, base, N, q_bhsd, k_bhsd, v_bhsd, results):
    BM, BN = fa4_tiles(d)
    dev = q_bhsd.device
    wvec = WT.wavelengths_sorted(d, base, K_RET, device=dev)
    zhi = window_zhi_from_kret(d, base, K_RET, N_CAP)
    # cross-check vs torch-side table (must be 0-diff; verified pre-hoc as well)
    zhi_t = zhi_table([float(x) for x in wvec.cpu()], d, N_CAP)
    assert tuple(zhi) == tuple(zhi_t), (zhi, zhi_t)

    ref32 = ref_window_attn(q_bhsd, k_bhsd, v_bhsd, wvec, phase="prefill",
                            block_m=BM, block_n=BN, mode="fp32")
    ref16 = ref_window_attn(q_bhsd, k_bhsd, v_bhsd, wvec, phase="prefill",
                            block_m=BM, block_n=BN, mode="bf16")
    anchor = (ref16.float() - ref32).abs().max().item()

    # FA4 layout (B,S,H,D)
    q4 = q_bhsd.transpose(1, 2).contiguous()
    k4 = k_bhsd.transpose(1, 2).contiguous()
    v4 = v_bhsd.transpose(1, 2).contiguous()
    out, *_ = _flash_attn_fwd(q4, k4, v4, causal=True, pack_gqa=False,
                              atflash_window_zhi=zhi)
    torch.cuda.synchronize()
    out_bhsd = out.transpose(1, 2).float()

    err = (out_bhsd - ref32).abs().max().item()
    tol = 2.0 * anchor + 2e-3
    ok = err <= tol
    zhi_perN = window_zhi_from_kret(d, base, K_RET, N)
    stats = fast_slice_stats(N, BM, BN, zhi_perN, d)
    if N <= 8192:  # self-check fast stats vs WT authority on small shapes
        ref_stats = WT.mma_slice_stats("prefill", N, N, BM, BN, wvec, d)
        assert abs(stats["qk_slice_ratio"] - ref_stats["qk_slice_ratio"]) < 1e-12,             (stats, ref_stats)
        assert abs(stats["kload_dim_ratio"] - ref_stats["kload_dim_ratio"]) < 1e-12
    rec = {
        "case": name, "hq": hq, "hkv": hkv, "d": d, "N": N, "rope_base": base,
        "k_ret": K_RET, "block_m": BM, "block_n": BN, "pack_gqa": False,
        "zhi": list(zhi), "max_abs_err": err, "bf16_anchor": anchor, "tol": tol,
        "pass": bool(ok),
        "qk_slice_ratio_theory": stats["qk_slice_ratio"],
        "kload_dim_ratio_theory": stats["kload_dim_ratio"],
    }
    results.append(rec)
    print(f"[{name} N={N}] err={err:.4e} anchor={anchor:.4e} tol={tol:.4e} "
          f"{'PASS' if ok else 'FAIL'} qk_slice_ratio={stats['qk_slice_ratio']:.3f}",
          flush=True)
    return ok


def _rope_hf(x, positions, inv_freq):
    """HF rotate-half RoPE on std layout (..., seq, d): pair r = dims (r, r+P)."""
    P = x.shape[-1] // 2
    t = positions.float()[:, None] * inv_freq[None, :]          # (seq, P)
    cos, sin = torch.cos(t), torch.sin(t)
    x = x.float()
    x1, x2 = x[..., :P], x[..., P:]
    out = torch.empty_like(x)
    out[..., :P] = x1 * cos - x2 * sin
    out[..., P:] = x2 * cos + x1 * sin
    return out


def hf_to_sorted_pair_perm(d):
    """dim map: sorted-pair layout dim <- HF rotate-half layout dim.
    HF pair r = (r, r+P); sorted pair j = std pair r = P-1-j -> dims (2j, 2j+1)."""
    P = d // 2
    src = torch.empty(d, dtype=torch.long)
    for j in range(P):
        r = P - 1 - j
        src[2 * j] = r
        src[2 * j + 1] = r + P
    return src


def run_real_decode_case(npz_path, results):
    """Real-model q/k (V2 capture, pre-RoPE bf16-bits): decode-form M=1 case.

    FA4 causal is bottom-right aligned, so q_len=1 causal=True attends all keys
    = decode semantics; window distance n_idx_diag = seqlen_k - 1 matches the w01
    decode zmin exactly (also exercises the seqlen_q != seqlen_k offset path).
    """
    import numpy as np
    z = np.load(npz_path)
    kf = torch.from_numpy(z["k_full"].astype(np.uint16)).view(torch.bfloat16).cuda()
    vf = torch.from_numpy(z["v_full"].astype(np.uint16)).view(torch.bfloat16).cuda()
    qt = torch.from_numpy(z["q_tail"].astype(np.uint16)).view(torch.bfloat16).cuda()
    hkv, N, d = kf.shape
    hq = qt.shape[0]
    base = 1000000.0 if d == 64 else 500000.0
    inv_freq = torch.tensor(
        json.load(open(os.path.join(os.path.dirname(npz_path),
                                    f"meta_{'qwen' if d == 64 else 'llama'}.json")))["inv_freq"],
        device="cuda", dtype=torch.float32)
    # RoPE (fp32) at true positions, then permute to wavelength-sorted pair layout
    pos_k = torch.arange(N, device="cuda")
    pos_q = torch.tensor([N - 1], device="cuda")
    k_r = _rope_hf(kf, pos_k, inv_freq)
    q_r = _rope_hf(qt[:, -1:, :], pos_q, inv_freq)
    perm = hf_to_sorted_pair_perm(d).cuda()
    k_s = k_r[..., perm].to(torch.bfloat16).contiguous()          # (hkv, N, d)
    q_s = q_r[..., perm].to(torch.bfloat16).contiguous()          # (hq, 1, d)
    del kf, qt, k_r, q_r

    BM, BN = fa4_tiles(d)
    wvec = WT.wavelengths_sorted(d, base, K_RET, device="cuda")
    zhi = window_zhi_from_kret(d, base, K_RET, N_CAP)

    q_b = q_s.unsqueeze(0)                                        # (1,hq,1,d)
    k_b = k_s.unsqueeze(0)
    v_b = vf.unsqueeze(0)
    ref32 = ref_window_attn(q_b, k_b, v_b, wvec, phase="decode",
                            block_m=BM, block_n=BN, mode="fp32")
    ref16 = ref_window_attn(q_b, k_b, v_b, wvec, phase="decode",
                            block_m=BM, block_n=BN, mode="bf16")
    anchor = (ref16.float() - ref32).abs().max().item()

    out, *_ = _flash_attn_fwd(q_b.transpose(1, 2).contiguous(),
                              k_b.transpose(1, 2).contiguous(),
                              v_b.transpose(1, 2).contiguous(),
                              causal=True, pack_gqa=False,
                              atflash_window_zhi=zhi)
    torch.cuda.synchronize()
    err = (out.transpose(1, 2).float() - ref32).abs().max().item()
    tol = 2.0 * anchor + 2e-3
    ok = err <= tol
    rec = {
        "case": "real_qwen_decodeform", "hq": hq, "hkv": hkv, "d": d, "N": N,
        "rope_base": base, "k_ret": K_RET, "block_m": BM, "block_n": BN,
        "pack_gqa": False, "zhi": list(zhi), "max_abs_err": err,
        "bf16_anchor": anchor, "tol": tol, "pass": bool(ok),
        "source": npz_path,
        "note": "V2 capture pre-RoPE; RoPE+wavelength-sort applied in-script; "
                "M=1 causal(bottom-right)=decode; w01 phase=decode ref",
    }
    results.append(rec)
    print(f"[real_qwen N={N}] err={err:.4e} anchor={anchor:.4e} tol={tol:.4e} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--real", default=None, help="V2 capture npz (qkv_qwen_LNN.npz)")
    ap.add_argument("--ext", action="store_true", help="run GRID_EXT (64k/160k shapes)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    results = []
    n_fail = 0
    for (name, hq, hkv, d, base, Ns) in (GRID_EXT if args.ext else GRID):
        for N in Ns:
            g = torch.Generator(device="cuda").manual_seed(777)
            q = torch.randn((1, hq, N, d), generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)
            k = torch.randn((1, hkv, N, d), generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)
            v = torch.randn((1, hkv, N, d), generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)
            if not run_case(f"{name}_rand", hq, hkv, d, base, N, q, k, v, results):
                n_fail += 1
            del q, k, v
            torch.cuda.empty_cache()

    if args.real:
        ok = run_real_decode_case(args.real, results)
        if not ok:
            n_fail += 1

    with open(os.path.join(args.outdir, "w01_correctness.json"), "w") as f:
        json.dump({"tol_rule": "max|out-ref32| <= 2*max|ref16-ref32| + 2e-3",
                   "results": results}, f, indent=1)
    n = len(results)
    print(f"RESULT: {n - n_fail}/{n} PASS")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
