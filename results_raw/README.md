# results_raw/ — per-table/figure raw summaries

The raw run summaries behind the confirmed speed/quality tables. Full per-item
npz arrays (≈40 GB) are excluded per the size policy and are not part of this
repository.

| path | contents |
|---|---|
| `summaries/*.e2e_summary.json` | Per-configuration summaries: B2 (Llama-3.2-3B, 8K–128K + k-dial) and C2 (Qwen2.5-7B-1M, 256K–1M) — top1 / ppl / KL / accuracy / median speed. Runs tagged `_tim_` are the timing arm of the same configurations (the harness separates the timing and quality passes), so a k-dial row's quality and its cost come from the matching pair. |

Each run name starts with its UTC start time and then combines the series, the
arm or group, the length and the window setting; `../docs/glossary.md` decodes
the parts (for example `phaseb` = Llama-3.2-3B series, `phasec` =
Qwen2.5-7B-Instruct-1M series, `p3d2` = prefill `k = 3`, decode `k = 2`). The
settings of each run are also in the `meta` block of its summary.

Measured 2026-07-17 on RTX PRO 6000, quality and timing runs.
