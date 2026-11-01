# evaluations/ — what the window does to model outputs

Output-quality evaluations of the window rule (`atflash/window_table.py`, `w_r = k·λ_r`). Each subdirectory holds one study: a README that opens with the result and its figure, and the per-row data the numbers come from. Every value in the READMEs is copied from those data files, which are not rounded.

**Why the rule works**

| study | result |
|---|---|
| [`window_assignment/`](window_assignment/) | The wavelength assignment, not the number of kept terms, carries the fidelity: top-1 0.962–0.978 for the rule against 0.013–0.489 for reversed or random assignments of the same widths (Qwen2.5-0.5B, Llama-3.2-3B). |
| [`dropped_vs_kept_terms/`](dropped_vs_kept_terms/) | The removed query–key terms are 2.2–7.1× smaller than the kept ones at every distance (Qwen2.5-0.5B, Llama-3.2-3B). |

**Larger models and long contexts**

| study | result |
|---|---|
| [`model_scale/`](model_scale/) | `k = 2` prunes 36.84–48.58% of the QK terms at top-1 agreement 0.9640–0.9896 and \|ΔPPL\| ≤ 0.21% (Qwen2.5-7B/14B/32B, Qwen3-8B/14B, Llama-3.1-8B/70B). |
| [`ruler_niah_128k/`](ruler_niah_128k/) | Needles at about 131K tokens: 99/100 with `k = 2` and 100/100 with `k = 64` under `string_match`; exact attention 100/100 (Qwen2.5-7B-Instruct-1M). |

**Models that rotate only part of attention**

| study | result |
|---|---|
| [`partial_rope/`](partial_rope/) | On the rotary part the window keeps top-1 at 0.940–0.978; the same thresholds on the non-rotary part prune fewer terms and raise PPL by 9.8–184% (SmolLM3-3B, Phi-4-mini-instruct). |
| [`minimax_m27_partial_rope/`](minimax_m27_partial_rope/) | On the rotary half, top-1 0.873–0.902 at 16.9–21.4% of all QK terms pruned (33.9–42.8% of the rotary-dimension terms); in RULER generation at 16K, 100/100 per task with the same right/wrong verdict as Full, against 0/100 for the non-rotary cut. |
| [`mimo_v26_longbench_v2/`](mimo_v26_longbench_v2/) | Over all 503 items at 16K–128K the window stays within −1.59 to +2.51 points of the unwindowed model, and at 512K, inside the native 1M context, it changes accuracy by 0.0 points (99 items); every 95% interval includes 0. |
| [`partial_rope_qk_closed_form/`](partial_rope_qk_closed_form/) | The pruned share of partial-RoPE layouts is a closed form of the configuration, with a script; it matches the measured counts to within 3·10⁻⁵ (MiniMax-M2.7, MiMo-V2.6, SmolLM3-3B). |

**Multi-token prediction**

| study | result |
|---|---|
| [`mimo_v26_mtp_compatibility/`](mimo_v26_mtp_compatibility/) | With the window on (k = 1–4), MTP top-1 agreement is 0.966–0.977 (stock FA4 kernel 0.972–0.977), and the MTP–target agreement stays within 0.5 points of full attention. |

These studies measure output quality; the speed numbers come from the kernel ports and are documented in `../VERIFICATION.md`. Figures are in `../figs/`.

<details>
<summary>How the measurements were made</summary>

- `partial_rope_qk_closed_form/` is arithmetic on model configurations, and its script runs on the `atflash` package alone.
- `window_assignment/`, `dropped_vs_kept_terms/`, `model_scale/`, `partial_rope/` and `ruler_niah_128k/` ran in an eager-mode PyTorch reference harness on Hugging Face models in fp32. The harness computes attention block by block with the window applied per pair; each README states the procedure, the inputs with their hashes, and the metric definitions.
- `minimax_m27_partial_rope/`, `mimo_v26_longbench_v2/` and `mimo_v26_mtp_compatibility/` ran through a vLLM integration with an eager PyTorch window path, which will be released separately.

</details>

<details>
<summary>Layout of each study</summary>

Every study README opens with **Result.** (one or two sentences), then the figure and a short explanation; the setup, the tables and the file descriptions are folded under **Setup**, **Numbers** and **Files**. The data files follow one naming scheme:

| file | contents |
|---|---|
| `results.csv` | The main table of the study. |
| `results_<part>.csv` | Further tables of the same study (for example `results_paired.csv`). |
| `floors*.csv` | Comparisons without any window (precision, chunk size, kernel swap). |
| `items.csv` | Per-item data. |
| `text_manifest.json` | The input documents with their character spans and hashes. |

</details>

<details>
<summary>Shared definitions</summary>

- **exact / full / Full / F:** the same model and code path with the window off.
- **window:** the per-pair window with `k = 2` unless stated otherwise. Pairs whose window reaches past the sequence are never pruned. Dimensions without rotation are outside the scope of the window rule: they call for a different technique, and the pruned shares reported here count them as kept.
- **teacher-forced:** the model reads a fixed token sequence once, and every position predicts the next token of that sequence. No sampling is involved.
- **top-1 agreement:** the fraction of positions where the window's argmax equals the reference argmax, i.e. agreement with the unpruned model.
- **mean KL:** the mean over positions of `KL(P_reference ‖ P_window)` over the full vocabulary, in nats.
- **ΔPPL %:** `100 · (PPL_window / PPL_reference − 1)`, with PPL computed on the sequence's own next tokens.
- **QK pruned share:** the fraction of query–key inner-product terms (query, key, frequency pair) over the causal triangle that the window removes. It is a count, not a speed measurement.
- **precision:** where a study says fp32, weights and activations are fp32 with TF32 disabled (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`, `NVIDIA_TF32_OVERRIDE=0`, `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`, `torch.get_float32_matmul_precision() == "highest"`), recorded in each run. The vLLM studies run the released quantized checkpoints and state their precision individually.

</details>
