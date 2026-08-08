# window_tables/ — §3.2 Window Rule

Generators and materialized JSON for the per-RoPE-wavelength window table
`w[r] = k · λ_r` (convex schedule k=2). The tables are consumed both by the vLLM
CUSTOM window backend and by the HuggingFace / FA4 reference path; window OFF is
bit-identical to full attention.

## Window-width (zhi) generators — the §3.2 rule
| file | role |
|---|---|
| `atflash_window_zhi_gen.py` | Generate the zhi window table from the prime base / real `inv_freq` (FA4 16-multiple snap, `wavelengths_sorted`). |
| `c1_window_precheck.py` | 7B-1M (base 1e7) zhi generation + sha (argsort-separated gate). |
| `c1_window_tables.txt` | The generated 7B-1M zhi table + checksum. |

## w_meta generators + materialized tables (backend / HF consumption)
| file | role |
|---|---|
| `w_meta_from_hf.py` | Load a HF model's rotary `inv_freq` (float64) → `λ_r=2π/inv_freq` → `w_r=k·λ_r`. No formula reimplementation; CPU only. |
| `w_meta_7b1m.py` | Window meta for Qwen2.5-7B-Instruct-1M (native, convex k=2). |
| `w_meta_7b1m_yarn_s2.py` | YaRN s=2 variant (262144 → 524288 effective). |
| `w_meta_qwen2505b_convexk2.json` | Materialized table, Qwen2.5-0.5B-Instruct. |
| `w_meta_qwen257b1m_convexk2.json` | Materialized table, Qwen2.5-7B-Instruct-1M. |
| `w_meta_qwen257b1m_yarn_s2_convexk2.json` | Materialized table, 7B-1M under YaRN s=2. |

Sources: zhi generators from the 2026-07-17 RTX staging snapshot; the w_meta
generators and materialized tables from 2026-07-21 — both the copies used by the
serving backend and the ones used by the canonical runs, which are identical.
