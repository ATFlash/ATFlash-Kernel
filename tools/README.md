# tools/ — checks that run on any CPU

| file | role |
|---|---|
| `cpu_checks.py` | Consistency checks of the window tables and the closed-form study; needs PyTorch (CPU build), no GPU, no weights. |
| `make_overview_figure.py` | Draws `../figs/overview_top1.png` / `.pdf` (shown in `../docs/overview.md`) from the evaluations' `results.csv` files; needs matplotlib and numpy. The output is byte-identical across runs. |

```bash
python tools/cpu_checks.py      # prints PASS/FAIL per check, exit status 0 if all pass
```

What it checks:

1. **twin** — `atflash/window_table.py` (torch, used by the reference attention)
   and `window_tables/atflash_window.py` (torch-free, emits the kernel tables)
   give bit-identical fp32 widths and the same `D_eff` for head dims 64 and 128,
   five RoPE bases and five values of `k`.
2. **zhi** — every level boundary of the kernel's `zhi` table sits exactly where
   the reference's `D_eff` drops below that level.
3. **selfcheck** — `window_tables/atflash_window.py` passes its own structural
   self-check.
4. **recorded** — `window_tables/c1_window_precheck.py`, rerun now, prints the
   same wavelengths and the same FA4 16-dim / FlashInfer 8-dim tables as the
   recorded `window_tables/c1_window_tables.txt` (Qwen2.5-7B-Instruct-1M,
   `k = 2, 3, 4`).
5. **closed** — the closed-form script in
   `evaluations/partial_rope_qk_closed_form/` rewrites `results.csv` byte for
   byte, and its printed percentages equal the table in that README.

The file checksums of the release are in `../MD5SUMS.txt`
(`md5sum -c MD5SUMS.txt` from the repository root).
