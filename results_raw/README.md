# results_raw/ — per-table/figure raw summaries

The raw run summaries behind the confirmed speed/quality tables, plus one
representative per-item record. Full per-item npz arrays (≈40 GB) are excluded per
the size policy; a single representative item is included so the format is
inspectable.

| path | contents |
|---|---|
| `summaries/*.e2e_summary.json` | Per-configuration summaries: B2 (Llama-3.2-3B, 8K–128K + k-dial) and C2 (Qwen2.5-7B-1M, 256K–1M) — top1 / ppl / KL / accuracy / median speed. Runs tagged `_tim_` are the timing arm of the same configurations (the harness separates the timing and quality passes), so a k-dial row's quality and its cost come from the matching pair. |
| `representative_item_lbv2_*.npz` | One representative item's raw arrays (18 keys): generated tokens per arm (`tokens_off` / `tokens_win_free` / `tokens_cudnn`) and their match flags, per-step latency (`step_ms_off` / `step_ms_win_free` / `step_ms_cudnn`, `prefill_ms`, `wall_s`), quality (`nll_off`, `nll_win_forced`, `kl_win_vs_off`), and memory (`peak_alloc`, `free_before`, `free_after`, `prompt_len`). Full logits are **not** stored. |

Source: 2026-07-17 RTX staging snapshot (`shared/results/`) plus the timing
runs of the same date. The quality-benchmark harnesses and their per-item
records (∞Bench / MRCR / RULER / GSM8K, sequence-length scaling) are not part
of this speed-focused repository; see the paper for those results.
