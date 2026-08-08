# kernels/ — §5 Window ports into released implementations

The per-RoPE-wavelength window injected into three released attention
implementations. Window OFF is bit-identical to stock (validated by `../gates/`).

## `fa4_port/` — FlashAttention-4 window (prefill + decode split-KV)
| file | role |
|---|---|
| `fa4_prefill_window_PAPER_296.patch` | The minimal prefill-only window diff (the "minimal diff into FA4 prefill" of §5). |
| `fa4_prefill_window_MEASURED_bc21b79.patch` | Prefill window diff at the measured commit. |
| `fa4_full_window_HEAD.patch` | Full window diff (prefill fwd + decode split-KV). |
| `fa4_diffstat_ALL.txt`, `fa4_window_commits.txt` | Diffstat and the commit range for the ports. |
| `fa4_cute_interface.py` | FA4 CuTe interface with the window (zhi) opt-in plumbing. |
| `FA4_PATCH_RECONCILIATION.md` | Reconciles the PAPER vs MEASURED vs HEAD patches. |

## `flashinfer_port/` — FlashInfer decode window
| file | role |
|---|---|
| `decode.cuh.diff`, `utils.cuh.diff` | Unified diffs vs pristine FlashInfer 0.6.13 (91-line window + group dispatch). |
| `fi_window_helpers.py` | FI window-table vec8 snap + custom JIT module builder. |

## Notes

The patch files are reproduced **exactly as measured** — the line counts and the
binaries they produce are what the paper reports, so they are not tidied up.
Comments inside them refer to modules of the development tree (`kernel_window/...`)
that are not part of this release; those are provenance references, not
dependencies. The window-rule code they point at lives here as
`../atflash/window_table.py`.

The reference implementation the ports were validated against is
`../atflash/reference.py`; the acceptance tests are in `../gates/`.

Source: 2026-07-17 RTX staging snapshot.
