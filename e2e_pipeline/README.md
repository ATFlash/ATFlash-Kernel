# e2e_pipeline/ — end-to-end, up to 1M tokens

The integrated end-to-end harness (FA4 prefill + FlashInfer decode, window ON/OFF
in the same binary, chunked prefill, timing/quality separated) plus the pool
builders of the end-to-end runs. Run names such as `phasec_1m_tim_p2d2` are decoded in
`../docs/glossary.md`.

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

Sources: `atflash_e2e_fafi.py` and the pool builders as used in the 2026-07-17
runs on RTX PRO 6000; `combine_1m_n6.py` from the 2026-07-21 extension. Raw run summaries are
in `../results_raw/summaries/`.

`n6_1m/` holds the three extension runs (documents 3, 4, 5) that take the 1M
point from n=3 to n=6: each run's `e2e_summary.json` and `meta_start.json`, plus
`N6_1M_DIGEST.txt` — the per-request OFF/ON seconds and the pooled n=6 medians
(request, prefill and decode speedup) of the 1M point. The n=3 runs it
is combined with are the `phasec_*` summaries in `../results_raw/summaries/`.

## Wide-window end-to-end measurements

The same pipeline (FA4-port prefill feeding FlashInfer-port decode, window ON/OFF
in the same binary, RTX PRO 6000) was also measured with a wider window.

**A wide window is still faster.** With `k = 64`, the setting that retrieves
100/100 needles at 128K in `../evaluations/ruler_niah_128k/`, the whole request
on Qwen2.5-7B-Instruct-1M is 1.09× faster at 256K and 1.14× at 512K.

| context | `k` | whole-request speedup | range over the six rounds |
|---:|---:|---:|---:|
| 256K | 64 | 1.09× (1.0924) | 1.0920–1.0926 |
| 512K | 64 | 1.14× (1.1411) | 1.1402–1.1413 |

<details>
<summary>Setup</summary>

- Six LongBench-v2 documents per length, truncated to each length; six rounds.
- Speedup = window-OFF / window-ON whole-request time (prefill and generation)
  per document; each round takes the median over documents, and the table gives
  the median of the six round medians.

</details>
