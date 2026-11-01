# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""CPU-only consistency checks of the window tables and the closed-form study.

Needs Python 3.9+ and PyTorch (CPU build is enough); no GPU, no weights, no
network. Run from anywhere:

    python tools/cpu_checks.py

Checks:
  1. twin       atflash/window_table.py (torch, used by the reference) and
                window_tables/atflash_window.py (torch-free, emits the kernel
                tables) give bit-identical fp32 widths and the same D_eff.
  2. zhi        every zhi level boundary of the torch-free table is exactly where
                the torch reference's D_eff drops below that level.
  3. selfcheck  window_tables/atflash_window.py runs its structural self-check.
  4. recorded   window_tables/c1_window_precheck.py, rerun now, prints the same
                wavelengths and FA4 16-dim / FlashInfer 8-dim tables as the
                recorded window_tables/c1_window_tables.txt.
  5. closed     evaluations/partial_rope_qk_closed_form/qk_reduction_partial_rope.py
                rewrites results.csv byte for byte, and its printed percentages
                equal the table in that study's README.
Exit status is 0 only if every check passes.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="",
           PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "kernels", "flashinfer_port"),
                                       os.path.join(ROOT, "window_tables"), ROOT]))
sys.path[:0] = [os.path.join(ROOT, "window_tables"), ROOT]

import atflash_window as AW  # noqa: E402  (torch-free twin)
from atflash import window_table as WT  # noqa: E402  (torch reference)

BASES = (1e4, 5e5, 1e6, 5e6, 1e7)
K_RETS = (1.0, 1.5, 2.0, 3.0, 4.0)
HEAD_DIMS = (64, 128)
results = []


def report(name: str, ok: bool, detail: str) -> None:
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, os.path.join(ROOT, script), *args],
                          cwd=ROOT, env=ENV, capture_output=True, text=True)


def check_twin() -> None:
    n_z = 0
    bad = []
    for d in HEAD_DIMS:
        for base in BASES:
            for k in K_RETS:
                wt = WT.wavelengths_sorted(d, base, k)
                aw = AW.wavelengths_sorted(d, base, k)
                if wt.tolist() != list(aw):
                    bad.append((d, base, k, "widths"))
                    continue
                for z in sorted({0, 1, 7, 8, 9, 15, 16, 17} | {int(w) + e for w in aw[-12:] for e in (-1, 0, 1)}):
                    n_z += 1
                    if WT.deff_of_zmin(z, wt, d) != AW.deff_of_zmin(z, aw, d):
                        bad.append((d, base, k, z))
    report("twin", not bad, f"{len(HEAD_DIMS) * len(BASES) * len(K_RETS)} (head_dim, base, k) "
           f"cases: fp32 widths compared bit for bit, D_eff compared at {n_z} distances"
           + (f"; mismatches {bad[:5]}" if bad else ""))


def check_zhi() -> None:
    n_levels = 0
    bad = []
    for d in HEAD_DIMS:
        for base in BASES:
            for k in K_RETS:
                for n_ctx in (8192, 131072, 1_100_000):
                    wt = WT.wavelengths_sorted(d, base, k)
                    zhi = AW.window_zhi_from_kret(d, base, k, n_ctx)
                    for i, z in enumerate(zhi):
                        e = 16 * (i + 1)
                        n_levels += 1
                        if WT.deff_of_zmin(z, wt, d) < e:
                            bad.append((d, base, k, n_ctx, i, "below"))
                        if z < n_ctx and WT.deff_of_zmin(z + 1, wt, d) >= e:
                            bad.append((d, base, k, n_ctx, i, "not last"))
    report("zhi", not bad, f"{n_levels} level boundaries checked against the torch reference D_eff"
           + (f"; mismatches {bad[:5]}" if bad else ""))


def check_selfcheck() -> None:
    p = run("window_tables/atflash_window.py")
    ok = p.returncode == 0 and p.stdout.strip().endswith("atflash_window self-check OK")
    report("selfcheck", ok, "window_tables/atflash_window.py: "
           + (p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr.strip()[-300:]))


def table_lines(text: str) -> list:
    keep = ("model=", "wavelengths_sorted", "k=", "      FI  8snap")
    return [l.rstrip() for l in text.splitlines() if l.startswith(keep)]


def check_recorded() -> None:
    with open(os.path.join(ROOT, "window_tables", "c1_window_tables.txt"), encoding="utf-8") as f:
        recorded = table_lines(f.read())
    p = run("window_tables/c1_window_precheck.py")
    now = table_lines(p.stdout)
    gate = "ARGSORT gate (real argsort == raw canonical): True" in p.stdout
    ok = p.returncode == 0 and gate and now == recorded and len(recorded) == 8
    report("recorded", ok, f"{len(now)} table lines rerun vs {len(recorded)} recorded "
           f"(k = 2, 3, 4; FA4 16-dim and FlashInfer 8-dim), identical={now == recorded}, "
           f"argsort gate={gate}" + ("" if p.returncode == 0 else f"; stderr: {p.stderr[-300:]}"))


def check_closed_form() -> None:
    study = os.path.join(ROOT, "evaluations", "partial_rope_qk_closed_form")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "results.csv")
        p = run("evaluations/partial_rope_qk_closed_form/qk_reduction_partial_rope.py", out)
        if p.returncode != 0:
            report("closed", False, p.stderr[-300:])
            return
        with open(out, "rb") as f:
            new = f.read()
    with open(os.path.join(study, "results.csv"), "rb") as f:
        same_csv = new == f.read()
    printed = [re.findall(r"\d+\.\d+%", l) for l in p.stdout.splitlines()[1:]]
    with open(os.path.join(study, "README.md"), encoding="utf-8") as f:
        readme = [re.findall(r"\d+\.\d+%", l) for l in f if re.match(r"\| [\d,]+ \| \d", l)]
    same_table = printed == readme and len(readme) == 8
    report("closed", same_csv and same_table,
           f"results.csv byte-identical={same_csv}; printed table equals README table "
           f"({len(readme)} rows x 4 columns)={same_table}")


if __name__ == "__main__":
    check_twin()
    check_zhi()
    check_selfcheck()
    check_recorded()
    check_closed_form()
    print(f"{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)
