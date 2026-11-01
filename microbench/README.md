# microbench/ — port-latency figure

The 7B prefill microbenchmark behind the port-latency figure plus the
diagnostic probes that fixed the measurement protocol.

| file | role |
|---|---|
| `phasec_7b_prefill_mb.py` + `7b_prefill_mb.json` | The 4-series sweep behind the figure (stock FA4 / window / cuDNN / FlashInfer, 8K–512K) and its output. |
| `remeasure_512k.py` | Reproduce the >512K quadratic degradation. |
| `b1_bench_protocol.py`, `b1_correctness.py` | Prefill sweep, and the correctness check of the windowed path against `../atflash/reference.py`. |

One measurement decision is worth recording, because it changes which baseline
is legitimate: PyTorch SDPA's `is_causal=True` applies a **top-left** causal mask
when the query length is shorter than the key length, which is the wrong mask for
chunked prefill and makes that path look faster than it is. The baselines here
therefore use a bottom-right causal reference (the flash-family convention), not
cuDNN chunked prefill.

`ports_fig/` holds the measurements the two-panel port-latency figure is drawn
from, on the RTX PRO 6000 (sm120), Llama-3.2-3B:

| file | contents |
|---|---|
| `fiwin_bench2_rtx_sm120.json` | Decode panel: per-call CUDA-event latency for `sdpa_cudnn` / `sdpa_flash` / `fi_stock` / `fiwin_on` at kv_len 8K / 32K / 64K / 160K, same values per copy across implementations. |
| `b1_bench_A.json`, `b1_bench_B.json`, `meta_start_A.json` | Prefill panel: the FA4 window sweep (two processes) with the environment pin and the window/zhi configuration in `meta`. |

The figure's **128K group** comes from a later dedicated speed run, included here
with the `_n128k` suffix (the files above cover 8K/32K/64K):

| file | contents |
|---|---|
| `N128K_DIGEST.txt` | The digest both panels' 128K bars are read from. Decode @131072: `fi_stock` 0.34032 vs `fiwin_on` 0.26445 ms (stock/window = 1.2869). Prefill @131072: `fa4_off_packT` 329.6252 vs `fa4_on_packT` 254.7770 ms (1.2938), and against the fastest available FA-class implementation `min(cuDNN, FI)` = 304.6196 ms (1.1956). |
| `fiwin_bench2_n128k.json` | Decode measurements at kv_len 131072 (same series and protocol as the 8K–160K file). |
| `b1_bench_A_n128k.json`, `b1_bench_B_n128k.json`, `meta_start_A_n128k.json`, `meta_start_B_n128k.json` | Prefill measurements and environment pins for the same run. |
| `results_stream_A_n128k.jsonl`, `results_stream_B_n128k.jsonl` | Per-call streams behind the aggregates. |
| `clocks_start_n128k.txt`, `clocks_end_n128k.txt` | GPU clock/power state at the start and end of the run, so the timings can be judged against a stable clock. |

The digest also carries series the figure does not draw (`fa4_off_packF`,
`fa4_on_packF`, `fi_prefill`, `sdpa_flash`), kept so the selection is auditable.

Measured 2026-07-17 on RTX PRO 6000. Raw run summaries in `../results_raw/`.
