# Setup: building the ports and reproducing the numbers

Everything below was run on one machine; the exact software pins live in
`VERIFICATION.md` and inside each run's `meta_start.json`. Short version:
RTX PRO 6000 Blackwell WS (sm_120, 96 GB GDDR7), Ubuntu 24.04, Python 3.11/3.12,
PyTorch 2.11.0+cu128, CUDA 12.8, FlashInfer 0.6.13, `nvidia-cutlass-dsl==4.2.0`.

## 0. Environment

```bash
python3 -m venv venv && . venv/bin/activate
pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/cu128
pip install nvidia-cutlass-dsl==4.2.0 flashinfer-python==0.6.13 \
            transformers numpy
```

The harnesses assume model weights are available through the usual Hugging Face
cache (`HF_HOME`). Absolute `/home/USER/...` paths inside the scripts are
placeholders — point them at your checkout.

## 1. FA4 prefill port

The public FA2–FA4 wheels do not cover sm_120, so FlashAttention-4 is built
from source. Use the companion repository, which is the upstream tree at the
pinned base with the window already applied:

```bash
git clone https://github.com/ATFlash/ATFlash-Kernel-FA4
cd ATFlash-Kernel-FA4
pip install -e .   # CuTeDSL path: python-only, no C++ compilation at install
```

The raw diffs live in that repository's `patches/` directory; its
`patches/FA4_PATCH_RECONCILIATION.md` maps each patch to its measured number
(the minimal patch "+296/−13", the measured binary "+308/−13", and the full port).
Its git history applies them one commit per step, so `git diff` between
commits shows exactly what the window adds.

The window is opt-in: without the `atflash_window_zhi` argument the code path
is stock.

## 2. FlashInfer decode port

The released 0.6.13 wheel is patched as a shadow copy — upstream is not
rebuilt:

```bash
pip download flashinfer-python==0.6.13 --no-deps -d /tmp/fi
mkdir -p shadow && cd shadow && unzip -q /tmp/fi/*.whl
patch -p1 -d flashinfer < /path/to/kernels/flashinfer_port/decode.cuh.diff
patch -p1 -d flashinfer < /path/to/kernels/flashinfer_port/utils.cuh.diff
export PYTHONPATH=$PWD:$PYTHONPATH
export FLASHINFER_WORKSPACE_BASE=$PWD/cache
```

`kernels/flashinfer_port/fiwin_common.py` shows the JIT-module builder and
the vec8 window-table snap this port consumes.

## 3. Window tables

```bash
export PYTHONPATH=$PWD/kernels/flashinfer_port:$PWD/window_tables:$PYTHONPATH
python window_tables/c1_window_precheck.py   # CPU only, no weights
python tools/cpu_checks.py                   # CPU only: tables vs reference and vs the recorded run
```

The pre-check prints the argsort separation gate and the zhi tables for several
`k`, with hashes; `tools/cpu_checks.py` compares them with the recorded
`window_tables/c1_window_tables.txt` and with the torch reference. `window_tables/w_meta_from_hf.py` derives tables for any HF model from
its config (`inv_freq`); the `w_meta_*.json` files are the exact tables the
published runs used.

## 4. Acceptance gates — run these before believing anything

```bash
python gates/b0_dualkernel_smoke.py    # window OFF == stock, bit-identical, both ports
python gates/c0_chunk_neutrality.py    # chunked prefill == single-shot, OFF and ON
python gates/g1_bitexact.py            # FlashInfer decode OFF == stock (needs the shadow copy)
python gates/g2_correctness.py         # fp32 correctness of the windowed path
```

`g2` compares against `atflash/reference.py`; its captured-activation inputs
are large and not bundled — it regenerates or expects them locally.

## 5. Reproducing the headline numbers

```bash
# prefill microbenchmark (Qwen2.5-7B-1M shapes, 8K..512K, 4 systems)
python microbench/phasec_7b_prefill_mb.py

# end-to-end whole-request, window ON/OFF in the same binary
python e2e_pipeline/atflash_e2e_fafi.py --model qwen7b1m --bench longbench \
    --lb-long-only --ctx-cap 262144 --prefill-chunk 65536 --no-cudnn --timing-only
```

`e2e_pipeline/README.md` explains the pool builders and the flags; every
recorded run's exact configuration is in its `meta_start.json`, and the raw
summaries behind the tables are in `results_raw/summaries/`.
