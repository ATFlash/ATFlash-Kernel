# gates/ — §5 window-OFF = stock bit-identity / neutrality gates

Correctness gates asserting that the window ports do not perturb attention when the
window is OFF, and that chunked prefill equals single-shot.

| file | role |
|---|---|
| `b0_dualkernel_smoke.py` | Dual-kernel (FA4 + FlashInfer) in one process; window OFF is bit-identical to stock. |
| `c0_chunk_neutrality.py` | Chunked == single-shot bit-match (OFF / ON / KV-K). |
| `g1_bitexact.py` | FlashInfer decode: window OFF == stock, bit-exact. |
| `g2_correctness.py` | fp32 correctness of the window path. |

Source: 2026-07-17 RTX staging snapshot. These back the §5 claim that the window
is a pure reduction-axis slice (OFF is identity).

**Environment and inputs.** `g1_bitexact.py` / `g2_correctness.py` import
`fiwin_common` and `atflash_window`, which exist only inside the built port tree
— apply the patches under `../kernels/` first and run the gates from that build.
The G2 reference tensors `qkv_qwen_L12.npz` and `meta_qwen.json` are **not
included** (they are large captured activations); the gate regenerates or expects
them locally. Absolute paths in the scripts are placeholders — adjust for your
environment.
