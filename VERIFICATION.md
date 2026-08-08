# What was verified, and how

This file records what has actually been measured or proven about the window
ports, on what hardware, and where in this repository the evidence sits. The
second table is the complement: what is *not* covered. Both are meant to be read
before trusting a number, and the second one doubles as a list of things a pull
request could usefully add.

## Measurement environment

Taken from the run metadata in `microbench/7b_prefill_mb.json` and
`microbench/ports_fig/meta_start_A_n128k.json`, not from recollection:

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0 (sm_120), 96 GB GDDR7 |
| PyTorch / CUDA / cuDNN | 2.11.0+cu128 / 12.8 / 91900 |
| FlashInfer | 0.6.13 (released wheel) |
| FlashAttention-4 | built from source; upstream commit `5835c73` |
| Arithmetic | bf16 operands, TF32 off, batch 1 |
| Window setting | `k_ret = 2.0` unless a row says otherwise |
| Dates | prefill microbenchmark 2026-07-17; port latency runs 2026-07-16 to 07-19 |

The comparison baseline is the per-case faster of cuDNN (through PyTorch SDPA)
and FlashInfer, since those are the only FA-class implementations available as
public wheels for sm_120.

## Verified

| # | Claim | How it was checked | Evidence in this repo |
|---|---|---|---|
| 1 | With the window off, the FA4 port is bit-identical to stock FA4, and the FlashInfer port to stock FlashInfer. | Both kernels run in one process on identical inputs; outputs compared bitwise. | `gates/b0_dualkernel_smoke.py`, `gates/g1_bitexact.py` |
| 2 | Chunked prefill equals single-shot prefill bitwise, window on and off. | Same input prefilled in one shot and in 64K chunks; outputs compared bitwise. | `gates/c0_chunk_neutrality.py` |
| 3 | With the window on, the ported path is numerically correct. | fp32 comparison against the torch reference implementation of the same window. | `gates/g2_correctness.py`, `microbench/b1_correctness.py`, `atflash/reference.py` |
| 4 | The window table the kernels consume is the one the rule prescribes. | Table regenerated from the model config and compared against the tables actually loaded. | `window_tables/c1_window_precheck.py`, `window_tables/c1_window_tables.txt` |
| 5 | Prefill: the windowed kernel is the fastest of the four systems at every length from 8K to 512K on Qwen2.5-7B-1M. | Single-forward sweep, CUDA-event timing, median of 200 iterations, against stock FA4 / cuDNN / FlashInfer. | `microbench/phasec_7b_prefill_mb.py`, `microbench/7b_prefill_mb.json` |
| 6 | Decode and prefill both beat their stock hosts on Llama-3.2-3B at 8K–160K, by margins that grow with context. | Per-call CUDA-event latency, same run for all series; the KV pool is sized past L2 so decode is not measuring cache residency. | `microbench/ports_fig/`, in particular `N128K_DIGEST.txt` |
| 7 | End to end, prefill and decode ports compose into one pipeline and the whole-request time improves at 256K / 512K / 1M. | Window switched on and off inside the same binary, same documents truncated to each length, n=6 per length. | `e2e_pipeline/atflash_e2e_fafi.py`, `e2e_pipeline/n6_1m/`, `results_raw/summaries/` |
| 8 | The gains are not an artifact of one kernel family. | The same insertion method was applied to two unrelated codebases (a CuTeDSL kernel and a CUDA-template kernel) and both recover a similar fraction of the predicted reduction. | `kernels/fa4_port/`, `kernels/flashinfer_port/` |

Output quality — divergence from full attention, top-1 agreement, and absolute
benchmark scores — was measured separately, on GH200 and H100, and is reported
in the paper. Those harnesses are not part of this repository, which is scoped
to the kernels and their speed.

## Not covered

Nothing below has been measured. They are listed so that no one infers them
from the table above, and because each is a self-contained piece of work.

| Area | Status |
|---|---|
| Other GPU generations | Every speed number here is sm_120 with GDDR7. Behaviour on other generations is unmeasured, and should not be extrapolated: the window trades arithmetic and K-traffic for a shorter contraction, so its payoff depends on where a given machine's bottleneck sits. |
| Other vendors and backends | No non-CUDA backend has been tried. |
| Batch > 1 and concurrent requests | All measurements are batch 1, single request. |
| Distributed and tensor-parallel execution | Not attempted. The window is per-head and per-pair, so it should be orthogonal to a head-dimension split, but that is an argument, not a measurement. |
| Other attention kernels | Only FlashAttention-4 and FlashInfer have been ported. |
| Architectures other than standard RoPE attention | The rule is derived for attention layers that apply RoPE in the usual pairwise form. Layers that transform positions differently are out of scope, and layers without rotation are simply never pruned. |
| Training | Inference only. The window has not been used during training or fine-tuning. |
| Automatic choice of `k` | `k` is set by hand. There is no procedure here for picking it per model or per task beyond the closed-form cost and the measurements above. |

## A note on the patch files

`kernels/fa4_port/*.patch` are reproduced exactly as measured, because the line
counts and the resulting binaries are what the paper reports. Comments inside
them refer to modules of the development tree (`kernel_window/...`) that are not
part of this release; those references are provenance, not dependencies. The
window-table code they refer to is `atflash/window_table.py` here, and its
torch-free twin used to emit the tables is
`window_tables/atflash_window_zhi_gen.py`.
