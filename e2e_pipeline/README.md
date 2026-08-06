# e2e_pipeline/ — Table 3 / §7.2 end-to-end 1M-context

The integrated end-to-end harness (FA4 prefill + FlashInfer decode, window ON/OFF
in the same binary, chunked prefill, timing/quality separated) plus the pool
builders and launch drivers behind Table 3.

| file | role |
|---|---|
| `atflash_e2e_fafi.py` | The HF-integrated e2e harness: FA4 prefill + FI decode, window ON/OFF same binary, chunked prefill, CUDA-event per-forward timing, quality (top1/ppl/KL) decomposition. |
| `build_long_pool.py` | LongBench-v2 long-doc pool builder. |
| `truncate_pool.py`, `truncate_pool_1m.py` | Center-truncate the pool to 132K / 1M bands. |
| `c2_1m.sh`, `c2_256k.sh`, `c2_512k.sh`, `c2_rest.sh` | C2 (Qwen2.5-7B-1M) launch drivers for Table 3 (1M / 512K / 256K). |
| `b2_n8_kdial.sh`, `b2_timing.sh` | B2 (Llama-3.2-3B) e2e / k-dial drivers. |
| `gpu_lock.sh` | GPU-occupancy lock wrapper (`ATFLASH_AGENT` / `ATFLASH_MIN_FREE_MB` / `ATFLASH_CMD_TIMEOUT`). |
| `run_1m_e2e_ext.sh` | 1M n=3 → n=6 extension driver (docs 3,4,5; graceful yield to foreign GPU processes). |
| `combine_1m_n6.py` | Combine the n=3 run + extension → n=6 median request / prefill / decode speedup and OFF→ON seconds. |

Sources: `atflash_e2e_fafi.py` and the drivers/pool-builders from the 2026-07-17
RTX staging; `run_1m_e2e_ext.sh` / `combine_1m_n6.py` from the 2026-07-21 extension.
Raw run summaries are in `../results_raw/summaries/`.

`n6_1m/` holds the three extension runs (documents 3, 4, 5) that take the 1M
point from n=3 to n=6: each run's `e2e_summary.json` and `meta_start.json`, plus
`N6_1M_DIGEST.txt` — the per-request OFF/ON seconds and the pooled n=6 medians
(request, prefill and decode speedup) that the 1M row reports. The n=3 runs it
is combined with are the `phasec_*` summaries in `../results_raw/summaries/`.
