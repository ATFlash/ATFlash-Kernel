# FA4 window patches — reconciling the three sizes

The paper states the FA4 prefill port as "4 files, +296/−13 lines". Three patch
files sit in this directory with three different sizes. This note pins down
which is which, so the numbers can be checked rather than trusted.

**Base commit.** All diffs are against `4f1fde1`, the head of upstream PR #2656
("Pack-GQA on SM120"). Upstream merged that PR as the squash commit `5835c73`,
so the two trees have identical content: a diff against `4f1fde1` is a diff
against `5835c73`. The paper cites the public post-merge commit.

**Window commit chain** (on top of the base):
`4611f08` (ampere_helpers `k_chunks`) → `15c9564` (flash_fwd distance-banded
dim-prefix window) → `860e2e9` (`atflash_window.py`, the host-side table) →
`c7fd744` (interface opt-in) → `bc21b79` (trace/constexpr compile fix) → then
decode split-KV: `0e9a9a2` / `398961b` / `46eebce` = HEAD.

| range | files | +/− | what it is |
|---|---|---|---|
| `4f1fde1..c7fd744` | 4 | **+296 / −13** | The window logic itself — **the paper's number**, exact match. |
| `4f1fde1..bc21b79` | 4 | +308 / −13 | The binary the prefill latency figure was measured with. The extra +12 is the trace/constexpr **compile fix; the window logic is unchanged**. |
| `4f1fde1..HEAD (46eebce)` | 4 | +562 / −64 | The binary of the 7B prefill microbenchmark and the end-to-end runs. The prefill window is identical; decode split-KV accounts for the growth. |

**Conclusions.**
- "4 files, +296/−13" is exact for the window-logic diff at `c7fd744`. The file
  count (4) and the deletion count (−13) are invariant across all three
  snapshots.
- The strictest reading — "the window diff of the measured binary" — would be
  +308/−13 (`bc21b79`), where the +12 is a non-semantic compile fix.
- The prefill window is byte-for-byte the same in `c7fd744`, `bc21b79` and HEAD;
  decode split-KV is a separate mechanism layered on afterwards.
