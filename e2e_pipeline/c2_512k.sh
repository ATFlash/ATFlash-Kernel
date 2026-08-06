#!/usr/bin/env bash
# Phase C / C2 @512K: timing + quality, ON/OFF k=(2,2), 1M-pool 6件→512K, chunked 64k, cuDNN あり.
set -u
cd /home/USER/atflash
POOL=/home/USER/atflash/agents/agentB/phasea/lb_long_pool_1m.jsonl
ENVV="HF_HOME=/home/USER/atflash/shared/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PATH=/home/USER/atflash/shared/venv/bin:/usr/local/cuda/bin:/usr/bin:/bin PYTHONPATH=/home/USER/atflash/agents/agentA/shim_b1:/home/USER/atflash/agents/agentB/fiwin/pkgs FLASHINFER_WORKSPACE_BASE=/home/USER/atflash/agents/agentB/phasea/fi_cache CUDA_HOME=/usr/local/cuda"
PY=agents/agentB/pbenv/bin/python
SC=agents/agentB/phasea/atflash_e2e_fafi.py
LOG=agents/agentB/phasea/c2_512k.log
: > "$LOG"
BASE="--model qwen7b1m --bench longbench --lb-file $POOL --lb-long-only --ctx-cap 512000 --n 6 --lb-max-new 40 --prefill-chunk 65536 --k-ret-prefill 2 --k-ret-decode 2"

echo "=== $(date -u +%H:%M:%S) 512K timing" | tee -a "$LOG"
ATFLASH_AGENT=agentB ATFLASH_MIN_FREE_MB=42000 ATFLASH_CMD_TIMEOUT=5400 \
  bin/gpu_lock.sh env $ENVV $PY $SC $BASE --timing-only --topic phasec_512k_tim_p2d2 >> "$LOG" 2>&1
echo "=== $(date -u +%H:%M:%S) 512K timing rc=$?" | tee -a "$LOG"

echo "=== $(date -u +%H:%M:%S) 512K quality" | tee -a "$LOG"
ATFLASH_AGENT=agentB ATFLASH_MIN_FREE_MB=42000 ATFLASH_CMD_TIMEOUT=6000 \
  bin/gpu_lock.sh env $ENVV $PY $SC $BASE --topic phasec_512k_qual_p2d2 >> "$LOG" 2>&1
echo "=== $(date -u +%H:%M:%S) 512K quality rc=$?" | tee -a "$LOG"
echo "C2_512K_DONE $(date -u +%H:%M:%S)" | tee -a "$LOG"
