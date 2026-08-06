# Third-party notices

This repository contains derivative work of the following projects. The diffs
and modified files remain under the upstream licenses noted below; the upstream
license texts must be bundled verbatim before publication (TODO: copy
`LICENSE` files from the upstream trees — tracked as a pre-publication task).

## FlashAttention (FA4 port)

- Upstream: https://github.com/Dao-AILab/flash-attention (base commit `5835c73`)
- License: BSD-3-Clause
- Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
  Pradeep Ramani, Tri Dao.
- Affected files: `kernels/fa4_port/fa4_cute_interface.py` (modified copy of an
  upstream source file); `kernels/fa4_port/*.patch` (diffs against upstream).

## FlashInfer (decode port)

- Upstream: https://github.com/flashinfer-ai/flashinfer (released wheel 0.6.13)
- License: Apache-2.0
- Affected files: `kernels/flashinfer_port/decode.cuh.diff`,
  `kernels/flashinfer_port/utils.cuh.diff` (diffs against the released 0.6.13
  sources).

All other code in this repository is original to the ATFlash authors and will
be released under the repository license (see `README.md`, License — TBD).
