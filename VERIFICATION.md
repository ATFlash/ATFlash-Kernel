# What was verified, and how

This file records what has actually been measured or proven about the window
ports, on what hardware, and where in this repository the evidence sits.

## Measurement environment

Taken from the run metadata in `microbench/7b_prefill_mb.json` and
`microbench/ports_fig/meta_start_A_n128k.json`:

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0 (sm_120), 96 GB GDDR7 |
| PyTorch / CUDA / cuDNN | 2.11.0+cu128 / 12.8 / 91900 |
| FlashInfer | 0.6.13 (released wheel) |
| FlashAttention-4 | built from source at PR #2656 head `4f1fde1` (upstream squash-merged it as `5835c73` with minor edits in `flash_fwd.py`; the prefill-window diff applies cleanly to both) |
| Arithmetic | bf16 operands, TF32 off, batch 1 |
| Window setting | `k_ret = 2.0` unless a row says otherwise |
| Dates | prefill microbenchmark 2026-07-17; port latency runs 2026-07-16 to 07-19; correctness re-run (row 3) and end-to-end runs of row 7a in October 2026 |

The comparison baseline is the per-case faster of cuDNN (through PyTorch SDPA)
and FlashInfer, since those are the only FA-class implementations available as
public wheels for sm_120.

## Verified

| # | Claim | How it was checked | Evidence in this repo |
|---|---|---|---|
| 1 | With the window off, the FA4 port is bit-identical to stock FA4, and the FlashInfer port to stock FlashInfer. | Both kernels run in one process on identical inputs; outputs compared bitwise. | `gates/b0_dualkernel_smoke.py`, `gates/g1_bitexact.py` |
| 2 | Chunked prefill equals single-shot prefill bitwise, window on and off. | Same input prefilled in one shot and in 64K chunks; outputs compared bitwise. | `gates/c0_chunk_neutrality.py` |
| 3 | With the window on, the ported path is numerically correct: FA4 and FlashInfer each pass 9/9 cases, with maximum absolute error 0.0087 (FA4) and 0.0053 (FlashInfer). | fp32 comparison against the torch reference implementation of the same window, on Llama- and Qwen-shaped heads with random inputs at N = 512–32,768 plus one real Qwen decode-form case (N = 32,628). A case passes when its error is at most twice the bf16 reference's own deviation from fp32 plus 0.002. Re-run in October 2026 with matmul TF32 off and float32 matmul precision `highest` recorded in the process; every error value is identical to the July runs. | `gates/g2_correctness.py` (FlashInfer), `microbench/b1_correctness.py` (FA4), `atflash/reference.py` |
| 4 | The window table the kernels consume is the one the rule prescribes. | Table regenerated from the model config and compared against the tables actually loaded. | `window_tables/c1_window_precheck.py`, `window_tables/c1_window_tables.txt`; `tools/cpu_checks.py` reruns the comparison on any CPU |
| 5 | Prefill: the windowed kernel is the fastest of the four systems at every length from 8K to 512K on Qwen2.5-7B-1M. | Single-forward sweep, CUDA-event timing, median of 200 iterations, against stock FA4 / cuDNN / FlashInfer. | `microbench/phasec_7b_prefill_mb.py`, `microbench/7b_prefill_mb.json` |
| 6 | Decode and prefill both beat their stock hosts on Llama-3.2-3B at 8K–160K, by margins that grow with context. | Per-call CUDA-event latency, same run for all series; the KV pool is sized past L2 so decode is not measuring cache residency. | `microbench/ports_fig/`, in particular `N128K_DIGEST.txt` |
| 7 | End to end, prefill and decode ports compose into one pipeline and the whole-request time improves at 256K / 512K / 1M. | Window switched on and off inside the same binary, same documents truncated to each length, n=6 per length. | `e2e_pipeline/atflash_e2e_fafi.py`, `e2e_pipeline/n6_1m/`, `results_raw/summaries/` |
| 7a | The end-to-end gain holds with a wide window: at `k = 64` the whole request is 1.09× faster at 256K and 1.14× at 512K (Qwen2.5-7B-Instruct-1M). | Same pipeline as row 7, 6 documents per length, six rounds; per round the median over documents of OFF/ON whole-request time, then the median of the six round medians. | `e2e_pipeline/README.md` (summary; the per-run outputs are not in this repository) |
| 8 | The gains are not an artifact of one kernel family. | The same insertion method was applied to two unrelated codebases (a CuTeDSL kernel and a CUDA-template kernel) and both recover a similar fraction of the predicted reduction. | `ATFlash-Kernel-FA4` (companion repo), `kernels/flashinfer_port/` |

## Output quality

These output-quality measurements use the same window rule.
Each was recomputed independently from its raw outputs. The
numbers, with their conditions, are in the study's README and data files.

| # | Claim | How it was checked | Evidence in this repo |
|---|---|---|---|
| 9 | It is the wavelength assignment, not the term count, that preserves the output. With the kept count fixed at every distance, the rule keeps top-1 agreement at 0.962–0.978. Reversed and random assignments of the same widths give 0.013–0.489, and an input-dependent magnitude ranking gives 0.919–0.960. | Qwen2.5-0.5B and Llama-3.2-3B at 8K–64K; teacher-forced, fp32 with TF32 off, GH200. | `evaluations/window_assignment/` |
| 10 | The removed terms are small. At every distance with removals, their mean magnitude is 2.2–7.1× below that of the kept terms. | Per-pair query–key terms on real rotated `q`, `k` at 32K, all layers and heads, Qwen2.5-0.5B and Llama-3.2-3B; H100, reproduced on RTX PRO 6000. | `evaluations/dropped_vs_kept_terms/` |
| 11 | The fidelity holds on larger models. At 8K–32K, Qwen2.5-7B/14B/32B, Qwen3-8B/14B (base) and Llama-3.1-8B/70B (8B also at 64K, 70B at 8K and 16K) keep top-1 agreement at 0.9640–0.9896 and change PPL by −0.138% to +0.208%, while 36.84–48.58% of the QK terms are pruned. | Teacher-forced exact vs window (`k = 2`) on the same text, fp32 with TF32 off, H100. | `evaluations/model_scale/` |
| 12 | A single needle is still retrieved at 128K: 99/100 (window, `k = 2`) and 100/100 (window, `k = 64`) vs 100/100 (exact) under RULER's `string_match`. | 100 RULER items at about 131K tokens on Qwen2.5-7B-Instruct-1M, fp32 with TF32 off; answers are teacher-forced. | `evaluations/ruler_niah_128k/` |
| 13 | On partial-RoPE models the window belongs on the rotary part. There, top-1 is 0.940–0.978 and \|ΔPPL\| ≤ 0.46%. The same thresholds on the non-rotary part prune fewer terms but raise PPL by 9.8–184%. | SmolLM3-3B (8K–64K) and Phi-4-mini (8K–128K), four conditions per length, fp32 with TF32 off, RTX PRO 6000. | `evaluations/partial_rope/` |
| 14 | The same contrast holds on MiniMax-M2.7. On prose, the window on the rotary half removes 16.9–21.4% of all QK terms, which is 33.9–42.8% of the terms in the rotary dimensions, with KL 1.4–1.6× the stack's own chunk-size floor, while removing as many terms from the other half raises PPL by 43–108%. In RULER generation at 16K, the rotary-half window keeps 200/200 needles with the same judgment as full attention on every item; the non-rotary cut keeps 0/200. | Teacher-forced prose at 8K–32K, and greedy generation at 16K (all 100 items per task); released FP8 weights in vLLM on B200, window in an eager PyTorch attention path. | `evaluations/minimax_m27_partial_rope/` |
| 15 | On MiMo-V2.6, LongBench-v2 accuracy with k = 2 differs from no window by −1.59 to +2.51 points over all 503 items in four bands (16K–128K), and by 0.0 points at 512K (99 items, inside the native 1M context; paired 95% interval −3.5 to +3.5). Every paired 95% interval includes 0. At 16K–128K, position-wise KL is 1.1–1.3× the implementation floor (KL and PPL have no intervals). | Released quantized checkpoint in vLLM on B200, window in an eager PyTorch attention path on the 10 full-attention layers. The 512K band takes every item whose full text has at least 262,144 tokens (prompts of 271,101–524,846 tokens). | `evaluations/mimo_v26_longbench_v2/` |
| 16 | The pruned share of partial-RoPE layouts follows a closed form, which agrees with the exact counts of rows 13 and 14 to within 3·10⁻⁵. | A script on `atflash.window_table`; no weights needed. | `evaluations/partial_rope_qk_closed_form/` |

Rows 9–15 summarize measurements whose code is not in this repository. Rows
9–13 used an eager PyTorch reference harness. Rows 14–15 used a vLLM
integration with an eager PyTorch window path, which will be released
separately. Row 16 runs here.

## A note on the patch files

The FA4 window patches live in the companion repository `ATFlash-Kernel-FA4`
(under `patches/`, applied commit by commit in its history) and are kept
exactly as measured: the line counts and the binaries they produce are the ones
behind the speed numbers here. Comments inside them refer to modules of the
development tree (`kernel_window/...`) that are not part of this release; those
references are provenance, not dependencies. The window-table code they refer
to is `atflash/window_table.py` here, and its torch-free twin used to emit the
tables is `window_tables/atflash_window.py`.
