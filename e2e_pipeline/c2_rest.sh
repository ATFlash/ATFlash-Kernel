#!/usr/bin/env bash
# Phase C / C2 self-sequencing: wait for 256K done, then 512K, then 1M.
set -u
D=/home/USER/atflash/agents/agentB/phasea
echo "waiting for C2_256K_DONE..."
while ! grep -q C2_256K_DONE "$D/c2_256k.log" 2>/dev/null; do sleep 30; done
echo "256K done -> 512K"
bash "$D/c2_512k.sh"
echo "512K done -> 1M"
bash "$D/c2_1m.sh"
echo "C2_REST_DONE $(date -u +%H:%M:%S)"
