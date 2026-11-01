# gates/ — window-OFF = stock bit-identity / neutrality gates

Correctness gates asserting that the window ports do not perturb attention when the
window is OFF, and that chunked prefill equals single-shot. Run them on your build
before trusting any speed number. All four need a GPU.

| gate | file | what must hold |
|---|---|---|
| B0 | `b0_dualkernel_smoke.py` | Dual-kernel (FA4 + FlashInfer) in one process; window OFF is bit-identical to stock. |
| C0 | `c0_chunk_neutrality.py` | Chunked == single-shot bit-match (OFF / ON / KV-K), through the end-to-end harness at 262K. |
| G1 | `g1_bitexact.py` | FlashInfer decode: window OFF == stock, bit-exact. |
| G2 | `g2_correctness.py` | fp32 correctness of the window path (the script also reports G3, the window's effect on real attention weights). |

The scripts as run on 2026-07-17 (RTX PRO 6000). They show that the window
is a pure reduction-axis slice (OFF is identity). The gate IDs are explained in
`../docs/glossary.md`.

**Environment and inputs.** `g1_bitexact.py` / `g2_correctness.py` import
`fiwin_common` (`../kernels/flashinfer_port/fiwin_common.py`) and `atflash_window`
(`../window_tables/atflash_window.py`); put both directories on `PYTHONPATH`. They
also need the patched FlashInfer shadow copy — apply the diffs under
`../kernels/flashinfer_port/` first (see `../SETUP.md`). `c0_chunk_neutrality.py`
imports the end-to-end harness `../e2e_pipeline/atflash_e2e_fafi.py`.
The G2 reference tensors `qkv_qwen_L12.npz` and `meta_qwen.json` are **not
included** (they are large captured activations); the gate regenerates or expects
them locally. Absolute paths in the scripts are placeholders — adjust for your
environment.
