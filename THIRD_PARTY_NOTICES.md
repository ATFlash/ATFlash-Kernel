# Third-party notices

Original code in this repository is licensed under Apache-2.0 (see `LICENSE`
and `NOTICE`). The portions listed below are derived from other open-source
projects and remain under their original licenses, whose full texts are
included under `third_party/`.

Neither upstream project is affiliated with, sponsors, or endorses the
modifications in this repository. Bugs in the modified paths are ours, not
theirs. The upstream authors' names are used only to satisfy the attribution
requirements of their licenses.

## FlashAttention — moved to the companion repository

This repository contains **no FlashAttention-derived code**. Everything derived
from FlashAttention (BSD-3-Clause) — the window patches, their reconciliation
ledger, and the upstream tree with the window applied — lives in the companion
repository `ATFlash-Kernel-FA4`, under upstream's own license, so that this
repository is purely Apache-2.0.

## FlashInfer — Apache-2.0

- Upstream: https://github.com/flashinfer-ai/flashinfer
- Base version for all diffs here: released wheel `0.6.13`
- License text: [`third_party/flashinfer/LICENSE`](third_party/flashinfer/LICENSE)
- Upstream attribution notices, carried forward per Apache-2.0 section 4(d):
  [`third_party/flashinfer/NOTICE`](third_party/flashinfer/NOTICE)

Affected files:

| file | relationship to upstream |
|---|---|
| `kernels/flashinfer_port/decode.cuh.diff` | Diff against upstream (48 lines of upstream context). |
| `kernels/flashinfer_port/utils.cuh.diff` | Diff against upstream (6 lines of upstream context). |

`kernels/flashinfer_port/fi_window_helpers.py` is original work; it calls the
FlashInfer API but contains no upstream code.

## Everything else

All other files — the window rule and tables, the torch reference, the gates,
the benchmarks and harnesses, the figures and the documentation — are original
work of the copyright holder named in `NOTICE`, under Apache-2.0.
