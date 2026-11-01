# window_tables/ — Window Rule

Generators and materialized JSON for the per-RoPE-wavelength window table
`w[r] = k · λ_r` (convex schedule k=2). The tables are consumed by the kernel
ports, by the HuggingFace / FA4 reference path, and by a vLLM integration that
is released separately; window OFF is bit-identical to full attention. Terms
such as zhi, the 16-dim and 8-dim snaps and `D_eff` are defined in
`../docs/glossary.md`.

## Window-width (zhi) generators
| file | role |
|---|---|
| `atflash_window.py` | Generate the zhi window table from the prime base / real `inv_freq` (FA4 16-multiple snap, `wavelengths_sorted`). Scripts import it as `atflash_window`. |
| `c1_window_precheck.py` | 7B-1M (base 1e7) zhi generation + sha (argsort-separated gate). CPU only: `PYTHONPATH=kernels/flashinfer_port:window_tables python window_tables/c1_window_precheck.py` from the repository root. |
| `c1_window_tables.txt` | The generated 7B-1M zhi table + checksum, as recorded at the time of the runs. `../tools/cpu_checks.py` reruns the pre-check and compares its tables with this file. |

## w_meta generators + materialized tables (HF / serving consumption)
| file | role |
|---|---|
| `w_meta_from_hf.py` | Load a HF model's rotary `inv_freq` (float64) → `λ_r=2π/inv_freq` → `w_r=k·λ_r`. No formula reimplementation; CPU only. |
| `w_meta_7b1m.py` | Window meta for Qwen2.5-7B-Instruct-1M (native, convex k=2). |
| `w_meta_7b1m_yarn_s2.py` | YaRN s=2 variant (262144 → 524288 effective). |
| `w_meta_qwen2505b_convexk2.json` | Materialized table, Qwen2.5-0.5B (the `model` field of the file). |
| `w_meta_qwen257b1m_convexk2.json` | Materialized table, Qwen2.5-7B-Instruct-1M. |
| `w_meta_qwen257b1m_yarn_s2_convexk2.json` | Materialized table, 7B-1M under YaRN s=2. |

Sources: zhi generators as used in the 2026-07-17 runs on RTX PRO 6000; the w_meta
generators and materialized tables from 2026-07-21 — both the copies used by the
serving integration and the ones used by the canonical runs, which are identical.
