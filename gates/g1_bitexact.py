# -*- coding: utf-8 -*-
"""fiwin G1: window-OFF bit-identity gate.

Chain per case: shadow_stock (stock module path, patched header, has_member=false)
  == custom_off (atfwin module, zhi=None)
  == custom_inf (atfwin module, zhi from k_ret=inf -> INT32_MAX -> pos_min=0).
Optionally compares against a pristine dump (baseline venv run) if --pristine dir given.
"""
import hashlib
import json
import os
import sys

import torch
import flashinfer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fiwin_common import build_atfwin_module, make_case, run_atfwin, zhi_slice_vec8


def sha(t):
    return hashlib.sha256(t.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def main(outdir, pristine_dir=None):
    os.makedirs(outdir, exist_ok=True)
    report = {"meta": {"flashinfer": flashinfer.__version__,
                       "flashinfer_file": flashinfer.__file__,
                       "torch": torch.__version__,
                       "purpose": "G1 window-OFF bit identity (ATFLASH-B2)"},
              "cases": [], "all_pass": True}
    for model, kv in [("llama", 8192), ("llama", 32768), ("qwen", 8192), ("qwen", 32768)]:
        q, k, v, base = make_case(model, kv)
        d = q.shape[-1]
        o_stock = flashinfer.single_decode_with_kv_cache(q, k, v)
        mod = build_atfwin_module(torch.bfloat16, d)
        o_off = run_atfwin(mod, q, k, v, zhi_slice=None)
        zhi_inf = zhi_slice_vec8(d, base, float("inf"))
        o_inf = run_atfwin(mod, q, k, v, zhi_slice=zhi_inf)
        h_stock, h_off, h_inf = sha(o_stock), sha(o_off), sha(o_inf)
        entry = {"case": f"{model}_kv{kv}", "sha_stock": h_stock[:16],
                 "stock_eq_off": h_stock == h_off, "off_eq_inf": h_off == h_inf}
        torch.save(o_stock.cpu(), os.path.join(outdir, f"o_stock_{model}_{kv}.pt"))
        if pristine_dir:
            pf = os.path.join(pristine_dir, f"o_pristine_{model}_{kv}.pt")
            if os.path.exists(pf):
                blob = torch.load(pf)
                o_re = flashinfer.single_decode_with_kv_cache(
                    blob["q"].cuda(), blob["k"].cuda(), blob["v"].cuda())
                entry["pristine_eq_stock"] = sha(blob["o"].cuda()) == sha(o_re)
            else:
                entry["pristine_eq_stock"] = None  # pristine cannot run (e.g. qwen g7)
        ok = entry["stock_eq_off"] and entry["off_eq_inf"] and entry.get("pristine_eq_stock", True) in (True, None)
        report["all_pass"] = report["all_pass"] and ok
        report["cases"].append(entry)
        print(entry)
    with open(os.path.join(outdir, "g1_bitexact.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("G1_ALL_PASS" if report["all_pass"] else "G1_FAIL")
    return 0 if report["all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
