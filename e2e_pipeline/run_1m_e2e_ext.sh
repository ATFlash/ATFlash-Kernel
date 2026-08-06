#!/usr/bin/env bash
# Table 3 1M e2e n=3 -> n=6 extension. VERBATIM c2_1m.sh p2d2 config, only adds
# --lb-skip {3,4,5} (the other 3 docs of the 6-doc 1M pool). Per-doc --n 1 so a
# FOREIGN GPU process (another user's job / llama-server) appearing between docs -> clean yield
# (no mid-1M-kernel kill). Distinct topics = no overwrite. Same binary/env as Table 3.
set -u
cd /home/USER/atflash
A=/home/USER/atflash
POOL=$A/agents/agentB/phasea/lb_long_pool_1m.jsonl
ENVV="HF_HOME=$A/shared/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PATH=$A/shared/venv/bin:/usr/local/cuda/bin:/usr/bin:/bin PYTHONPATH=$A/agents/agentA/shim_b1:$A/agents/agentB/fiwin/pkgs FLASHINFER_WORKSPACE_BASE=$A/agents/agentB/phasea/fi_cache CUDA_HOME=/usr/local/cuda"
PY=$A/agents/agentB/pbenv/bin/python
SC=$A/agents/agentB/phasea/atflash_e2e_fafi.py
LOG=$A/agents/agentB/phasea/c2_1m_ext.log
: > "$LOG"
COMMON="--model qwen7b1m --bench longbench --lb-file $POOL --lb-long-only --ctx-cap 1000000 --n 1 --lb-max-new 40 --prefill-chunk 65536 --no-cudnn --k-ret-prefill 2 --k-ret-decode 2"

foreign_gpu() {   # true if any GPU compute-app process exists (before we launch ours)
  local p; p=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
  [ -n "$p" ] && { echo "$p"; return 0; } || return 1
}

echo "$(date -u +%H:%M:%S) 1M e2e EXT START (docs 3,4,5 = p2d2 config)" | tee -a "$LOG"
for skip in 3 4 5; do
  f=$(foreign_gpu) && { echo "$(date -u +%H:%M:%S) YIELD: foreign GPU proc before doc$skip -> stop ($f)" | tee -a "$LOG"; break; }
  echo "=== $(date -u +%H:%M:%S) doc$skip START (lb-skip $skip)" | tee -a "$LOG"
  ATFLASH_AGENT=agentB ATFLASH_MIN_FREE_MB=20000 ATFLASH_CMD_TIMEOUT=10800 \
    bin/gpu_lock.sh env $ENVV $PY $SC $COMMON --lb-skip $skip --topic phasec_1m_tim_ext_d$skip >> "$LOG" 2>&1
  echo "=== $(date -u +%H:%M:%S) doc$skip rc=$?" | tee -a "$LOG"
done
echo "C2_1M_EXT_DONE $(date -u +%H:%M:%S)" | tee -a "$LOG"
