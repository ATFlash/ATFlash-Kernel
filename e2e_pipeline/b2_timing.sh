#!/usr/bin/env bash
# Phase B: TIMING-ONLY run (no per-step logits.cpu() sync) -> clean full-req wall
# + prefill + decode for the same n=8 configs. Quality (top1/ppl/KL) comes from
# the paired quality sweep (b2_n8_kdial). Same pool, same settings, separate pass.
set -u
cd /home/USER/atflash
POOL=/home/USER/atflash/agents/agentB/phasea/lb_long_pool_132k.jsonl
ENVV="HF_HOME=/home/USER/atflash/shared/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PATH=/home/USER/atflash/shared/venv/bin:/usr/local/cuda/bin:/usr/bin:/bin PYTHONPATH=/home/USER/atflash/agents/agentA/shim_b1:/home/USER/atflash/agents/agentB/fiwin/pkgs FLASHINFER_WORKSPACE_BASE=/home/USER/atflash/agents/agentB/phasea/fi_cache CUDA_HOME=/usr/local/cuda"
PY=agents/agentB/pbenv/bin/python
SC=agents/agentB/phasea/atflash_e2e_fafi.py
LOG=agents/agentB/phasea/b2_timing.log
: > "$LOG"

run() {  # ctxcap kP kD timeout minfree topic
  echo "=== $(date -u +%H:%M:%S) START ctx=$1 kP=$2 kD=$3 topic=$6" | tee -a "$LOG"
  ATFLASH_AGENT=agentB ATFLASH_MIN_FREE_MB=$5 ATFLASH_CMD_TIMEOUT=$4 \
    bin/gpu_lock.sh env $ENVV $PY $SC --model llama --bench longbench \
    --lb-file "$POOL" --lb-long-only --ctx-cap $1 --n 8 --lb-max-new 40 \
    --k-ret-prefill $2 --k-ret-decode $3 --timing-only --topic $6 >> "$LOG" 2>&1
  echo "=== $(date -u +%H:%M:%S) END rc=$? ctx=$1 kP=$2 kD=$3" | tee -a "$LOG"
}

run 8000    2 2 900  12000 phaseb_tim_8k_p2d2
run 32000   2 2 900  18000 phaseb_tim_32k_p2d2
run 64000   2 2 1200 26000 phaseb_tim_64k_p2d2
run 128000  2 2 1800 40000 phaseb_tim_128k_p2d2
run 64000   3 2 1200 26000 phaseb_tim_64k_p3d2
run 64000   4 2 1200 26000 phaseb_tim_64k_p4d2
run 64000   3 3 1200 26000 phaseb_tim_64k_p3d3
run 128000  3 2 1800 40000 phaseb_tim_128k_p3d2
run 128000  4 2 1800 40000 phaseb_tim_128k_p4d2
run 128000  3 3 1800 40000 phaseb_tim_128k_p3d3
echo "B2_TIMING_DONE $(date -u +%H:%M:%S)" | tee -a "$LOG"
