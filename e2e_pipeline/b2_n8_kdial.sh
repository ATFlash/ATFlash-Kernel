#!/usr/bin/env bash
# Phase B: n=8 main re-measure (k=2,2, 4 sizes) + k-dial ({3,2}/{4,2}/{3,3} x
# 64k/128k). Same 10-item 132k long pool. width=素基数, order=real permute, only k moves.
set -u
cd /home/USER/atflash
POOL=/home/USER/atflash/agents/agentB/phasea/lb_long_pool_132k.jsonl
ENVV="HF_HOME=/home/USER/atflash/shared/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PATH=/home/USER/atflash/shared/venv/bin:/usr/local/cuda/bin:/usr/bin:/bin PYTHONPATH=/home/USER/atflash/agents/agentA/shim_b1:/home/USER/atflash/agents/agentB/fiwin/pkgs FLASHINFER_WORKSPACE_BASE=/home/USER/atflash/agents/agentB/phasea/fi_cache CUDA_HOME=/usr/local/cuda"
PY=agents/agentB/pbenv/bin/python
SC=agents/agentB/phasea/atflash_e2e_fafi.py
LOG=agents/agentB/phasea/b2_n8_kdial.log
: > "$LOG"

run() {  # ctxcap kP kD timeout minfree topic
  echo "=== $(date -u +%H:%M:%S) START ctx=$1 kP=$2 kD=$3 topic=$6" | tee -a "$LOG"
  ATFLASH_AGENT=agentB ATFLASH_MIN_FREE_MB=$5 ATFLASH_CMD_TIMEOUT=$4 \
    bin/gpu_lock.sh env $ENVV $PY $SC --model llama --bench longbench \
    --lb-file "$POOL" --lb-long-only --ctx-cap $1 --n 8 --lb-max-new 40 \
    --k-ret-prefill $2 --k-ret-decode $3 --topic $6 >> "$LOG" 2>&1
  echo "=== $(date -u +%H:%M:%S) END rc=$? ctx=$1 kP=$2 kD=$3" | tee -a "$LOG"
}

# main n=8 (k=2,2)
run 8000    2 2 1200 12000 phaseb_b2n8_8k_p2d2
run 32000   2 2 1500 18000 phaseb_b2n8_32k_p2d2
run 64000   2 2 1500 26000 phaseb_b2n8_64k_p2d2
run 128000  2 2 2400 40000 phaseb_b2n8_128k_p2d2
# k-dial extra (64k)
run 64000   3 2 1500 26000 phaseb_kdial_64k_p3d2
run 64000   4 2 1500 26000 phaseb_kdial_64k_p4d2
run 64000   3 3 1500 26000 phaseb_kdial_64k_p3d3
# k-dial extra (128k)
run 128000  3 2 2400 40000 phaseb_kdial_128k_p3d2
run 128000  4 2 2400 40000 phaseb_kdial_128k_p4d2
run 128000  3 3 2400 40000 phaseb_kdial_128k_p3d3
echo "B2N8_KDIAL_DONE $(date -u +%H:%M:%S)" | tee -a "$LOG"
