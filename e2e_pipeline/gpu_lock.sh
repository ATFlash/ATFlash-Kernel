#!/usr/bin/env bash
# ATFlash GPU 占有ロック wrapper v2 (RTX PRO 6000 = gpu0 のみ / 他ユーザ llama-server 同居中)
# 使い方:  gpu_lock.sh <command...>
# 環境変数:
#   ATFLASH_LOCK_WAIT    lock 取得待ち秒 (default 7200)
#   ATFLASH_CMD_TIMEOUT  コマンド実行上限秒 (default 3600; ハングが lock を握り続けない)
#   ATFLASH_MIN_FREE_MB  実行前に要求する VRAM 空き MB (default 1500; 不足なら実行せず exit 42)
#   ATFLASH_AGENT        呼び出し元 agent 名 (ログ用; 未設定なら unknown)
# 規則: GPU を使う process は必ずこれを経由。他ユーザ process には一切干渉しない。
set -u
ROOT=/home/USER/atflash
mkdir -p "$ROOT/locks"
LOG="$ROOT/locks/gpu0.log"

run_locked() {
  ROOT=/home/USER/atflash
  LOG="$ROOT/locks/gpu0.log"
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  MIN=${ATFLASH_MIN_FREE_MB:-1500}
  TS=$(date +%Y-%m-%dT%H:%M:%S)
  if [ "$FREE" -lt "$MIN" ]; then
    echo "$TS AGENT=${ATFLASH_AGENT:-unknown} ABORT free=${FREE}MB<min=${MIN}MB CMD=$*" >> "$LOG"
    echo "gpu_lock: VRAM free ${FREE}MB < required ${MIN}MB — not starting (exit 42)" >&2
    exit 42
  fi
  echo "$TS AGENT=${ATFLASH_AGENT:-unknown} START free=${FREE}MB timeout=${ATFLASH_CMD_TIMEOUT:-3600}s CMD=$*" >> "$LOG"
  timeout --signal=TERM --kill-after=30 "${ATFLASH_CMD_TIMEOUT:-3600}" "$@"
  RC=$?
  echo "$(date +%Y-%m-%dT%H:%M:%S) AGENT=${ATFLASH_AGENT:-unknown} END rc=$RC CMD=$*" >> "$LOG"
  exit $RC
}
export -f run_locked >/dev/null 2>&1 || true

exec flock -w "${ATFLASH_LOCK_WAIT:-7200}" "$ROOT/locks/gpu0.lock" bash -c 'run_locked "$@"' _ "$@"
