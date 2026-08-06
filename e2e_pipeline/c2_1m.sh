#!/usr/bin/env bash
# Phase C / C2 @1M: quality-run 単独(wall流用)・--no-cudnn・n=3,
# k=(2,2)主軸 + ON腕 k=(3,3). 1M-pool→1M, chunked 64k. prefill/decode=CUDA-event クリーン.
set -u
cd /home/USER/atflash
POOL=/home/USER/atflash/agents/agentB/phasea/lb_long_pool_1m.jsonl
ENVV="HF_HOME=/home/USER/atflash/shared/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PATH=/home/USER/atflash/shared/venv/bin:/usr/local/cuda/bin:/usr/bin:/bin PYTHONPATH=/home/USER/atflash/agents/agentA/shim_b1:/home/USER/atflash/agents/agentB/fiwin/pkgs FLASHINFER_WORKSPACE_BASE=/home/USER/atflash/agents/agentB/phasea/fi_cache CUDA_HOME=/usr/local/cuda"
PY=agents/agentB/pbenv/bin/python
SC=agents/agentB/phasea/atflash_e2e_fafi.py
LOG=agents/agentB/phasea/c2_1m.log
: > "$LOG"
COMMON="--model qwen7b1m --bench longbench --lb-file $POOL --lb-long-only --ctx-cap 1000000 --n 3 --lb-max-new 40 --prefill-chunk 65536 --no-cudnn"

echo "=== $(date -u +%H:%M:%S) 1M quality k=(2,2)" | tee -a "$LOG"
ATFLASH_AGENT=agentB ATFLASH_MIN_FREE_MB=20000 ATFLASH_CMD_TIMEOUT=10800 \
  bin/gpu_lock.sh env $ENVV $PY $SC $COMMON --k-ret-prefill 2 --k-ret-decode 2 --topic phasec_1m_qual_p2d2 >> "$LOG" 2>&1
echo "=== $(date -u +%H:%M:%S) 1M k22 rc=$?" | tee -a "$LOG"

echo "=== $(date -u +%H:%M:%S) 1M quality k=(3,3) ON腕" | tee -a "$LOG"
ATFLASH_AGENT=agentB ATFLASH_MIN_FREE_MB=20000 ATFLASH_CMD_TIMEOUT=10800 \
  bin/gpu_lock.sh env $ENVV $PY $SC $COMMON --k-ret-prefill 3 --k-ret-decode 3 --topic phasec_1m_qual_p3d3 >> "$LOG" 2>&1
echo "=== $(date -u +%H:%M:%S) 1M k33 rc=$?" | tee -a "$LOG"
echo "C2_1M_DONE $(date -u +%H:%M:%S)" | tee -a "$LOG"
