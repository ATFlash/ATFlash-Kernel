# e2e_pipeline/ — Table 3 / §7.2 end-to-end 1M-context

The integrated end-to-end harness (FA4 prefill + FlashInfer decode, window ON/OFF
in the same binary, chunked prefill, timing/quality separated) plus the pool
builders and launch drivers behind Table 3.

| file | role |
|---|---|
| `atflash_e2e_fafi.py` | The HF-integrated e2e harness: FA4 prefill + FI decode, window ON/OFF same binary, chunked prefill, CUDA-event per-forward timing, quality (top1/ppl/KL) decomposition. |
| `build_long_pool.py` | LongBench-v2 long-doc pool builder. |
| `truncate_pool.py`, `truncate_pool_1m.py` | Center-truncate the pool to 132K / 1M bands. |
| `combine_1m_n6.py` | Combine the n=3 run + extension → n=6 median request / prefill / decode speedup and OFF→ON seconds. |

The shell drivers that launched these runs are not included: they were specific
to one machine's paths and scheduler, and carried no information beyond the
arguments. The harness takes its configuration on the command line — read the
`argparse` block at the top of `atflash_e2e_fafi.py`, and the `meta_start.json`
of any run under `n6_1m/` records the exact settings that produced it.

Sources: `atflash_e2e_fafi.py` and the pool builders from the 2026-07-17 RTX
staging; `combine_1m_n6.py` from the 2026-07-21 extension. Raw run summaries are
in `../results_raw/summaries/`.

`n6_1m/` holds the three extension runs (documents 3, 4, 5) that take the 1M
point from n=3 to n=6: each run's `e2e_summary.json` and `meta_start.json`, plus
`N6_1M_DIGEST.txt` — the per-request OFF/ON seconds and the pooled n=6 medians
(request, prefill and decode speedup) that the 1M row reports. The n=3 runs it
is combined with are the `phasec_*` summaries in `../results_raw/summaries/`.
