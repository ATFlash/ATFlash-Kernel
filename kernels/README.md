# kernels/ — §5 Window ports into released implementations

The per-RoPE-wavelength window injected into released attention
implementations. Window OFF is bit-identical to stock (validated by `../gates/`).

## FlashAttention-4 window (prefill + decode split-KV) — separate repository

Everything derived from FlashAttention (BSD-3-Clause) lives in the companion
repository **`ATFlash-Kernel-FA4`**, which keeps this repository purely
Apache-2.0. That repository carries the upstream tree at the pinned base with
the window applied commit by commit, and, under `patches/`, the three window
diffs (paper / measured / full) with their reconciliation ledger.

## `flashinfer_port/` — FlashInfer decode window
| file | role |
|---|---|
| `decode.cuh.diff`, `utils.cuh.diff` | Unified diffs vs pristine FlashInfer 0.6.13 (91-line window + group dispatch). |
| `fi_window_helpers.py` | FI window-table vec8 snap + custom JIT module builder. |

## Notes

The FA4 patch files (in `ATFlash-Kernel-FA4/patches/`) are reproduced
**exactly as measured** — the line counts and the binaries they produce are
what the paper reports, so they are not tidied up. Comments inside them refer
to modules of the development tree (`kernel_window/...`) that are not part of
this release; those are provenance references, not dependencies. The
window-rule code they point at lives here as `../atflash/window_table.py`.

The reference implementation the ports were validated against is
`../atflash/reference.py`; the acceptance tests are in `../gates/`.

Source: 2026-07-17 RTX staging snapshot.
