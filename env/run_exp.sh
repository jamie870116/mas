#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-decen}"     # summary | log | decen
TASKSET="${2:-ALL}"      # ALL | TASKS_1 | TASKS_2 | TASKS_3 | TASKS_4

CHUNK=3

case "$METHOD" in
  summary) START=3 ;;
  log)     START=13 ;;
  decen)   START=23 ;;
  *)       echo "Unknown METHOD: $METHOD"; exit 1 ;;
esac
case "$METHOD" in
  summary) END=3 ;;
  log)     END=13 ;;
  decen)   END=23 ;;
  *)       echo "Unknown METHOD: $METHOD"; exit 1 ;;
esac

TIMEOUT=600
SLEEP_AFTER=5
MAX_RETRIES=1
IDLE_LIMIT=60     #  90 sec沒有任何輸出就重啟（自行調整）
POLL_INTERVAL=15  # 每 15 秒檢查一次 log 是否有更新
KILL_GRACE=10     # 先 TERM，等 10 秒不退就 KILL
kill_ai2thor() {
  echo "=== Cleanup: killing AI2-THOR ===" | tee -a "$LOG"
  pkill -f "/MacOS/AI2-THOR" 2>/dev/null || true
  sleep 2
  pkill -9 -f "/MacOS/AI2-THOR" 2>/dev/null || true
}

mkdir -p exp_logs

while true; do
  STAMP="$(date +%F_%H%M%S)"
  LOG="exp_logs/${METHOD}_${TASKSET}_${STAMP}.log"

  echo "=== RUN chunk: method=${METHOD}, taskset=${TASKSET} ===" | tee -a "$LOG"

  python -u env/run_exp.py \
  --method "$METHOD" \
  --taskset "$TASKSET" \
  --chunk "$CHUNK" \
  --start "$START" \
  --end "$END" \
  --timeout "$TIMEOUT" \
  --sleep_after "$SLEEP_AFTER" \
  --delete_frames \
  --max_retries "$MAX_RETRIES" \
  >> "$LOG" 2>&1 &
  PID=$!

  # 監控：如果 log 在 IDLE_LIMIT 秒內沒有更新 -> 重啟
  while kill -0 "$PID" 2>/dev/null; do
    sleep "$POLL_INTERVAL"


    # macOS: stat -f %m 取最後修改時間（epoch seconds）
    LAST_MOD=$(stat -f %m "$LOG" 2>/dev/null || echo 0)
    NOW=$(date +%s)
    DIFF=$((NOW - LAST_MOD))

    if [ "$DIFF" -ge "$IDLE_LIMIT" ]; then
      echo "=== IDLE WATCHDOG: no output for ${DIFF}s (>= ${IDLE_LIMIT}s). Restarting chunk. ===" | tee -a "$LOG"
      
      kill -INT "$PID" 2>/dev/null || true
      sleep 5
      if kill -0 "$PID" 2>/dev/null; then
        kill -KILL "$PID" 2>/dev/null || true
      fi
      kill_ai2thor
      break
    fi
  done

  # 等待 python 結束（避免殘留背景行程）
  wait "$PID" 2>/dev/null || true

  # 若 chunk 完成時印出 [DONE]，代表全部跑完，停止外層 while
  if grep -q "^\[DONE\]" "$LOG"; then
    echo "=== ALL DONE ===" | tee -a "$LOG"
    exit 0
  fi

  echo "=== Restarting process after chunk ===" | tee -a "$LOG"
  sleep 5
done