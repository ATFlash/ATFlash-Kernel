# Third-party notices

Original code in this repository is licensed under Apache-2.0 (see `LICENSE`
and `NOTICE`). The portions listed below are derived from other open-source
projects and remain under their original licenses, whose full texts are
included under `third_party/`.

Neither upstream project is affiliated with, sponsors, or endorses the
modifications in this repository. Bugs in the modified paths are ours, not
theirs. The upstream authors' names are used only to satisfy the attribution
requirements of their licenses.

## FlashAttention — BSD-3-Clause

- Upstream: https://github.com/Dao-AILab/flash-attention
- Base commit for all diffs here: `5835c73`
- License text: [`third_party/flash-attention/LICENSE`](third_party/flash-attention/LICENSE)
- Upstream authors: [`third_party/flash-attention/AUTHORS`](third_party/flash-attention/AUTHORS)

Affected files:

| file | relationship to upstream |
|---|---|
| `kernels/fa4_port/fa4_cute_interface.py` | Modified copy of an upstream source file. Retains the upstream per-file copyright header. |
| `kernels/fa4_port/fa4_prefill_window_PAPER_296.patch` | Diff against upstream (123 lines of upstream context). |
| `kernels/fa4_port/fa4_prefill_window_MEASURED_bc21b79.patch` | Diff against upstream (123 lines of upstream context). |
| `kernels/fa4_port/fa4_full_window_HEAD.patch` | Diff against upstream (208 lines of upstream context). |

BSD-3-Clause clause 3 forbids using the upstream authors' names to endorse or
promote products derived from their software; this repository makes no such
claim of endorsement.

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
work of the authors, under Apache-2.0.
