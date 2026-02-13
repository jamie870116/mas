#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-decen}"     # summary | log | decen
TASKSET="${2:-ALL}"      # ALL | TASKS_1 | TASKS_2 | TASKS_3 | TASKS_4

CHUNK=5
START=1
END=1
TIMEOUT=300
SLEEP_AFTER=5

IDLE_LIMIT=90     #  90 sec沒有任何輸出就重啟（自行調整）
POLL_INTERVAL=15  # 每 15 秒檢查一次 log 是否有更新
KILL_GRACE=10     # 先 TERM，等 10 秒不退就 KILL

mkdir -p exp_logs

while true; do
  STAMP="$(date +%F_%H%M%S)"
  LOG="exp_logs/${METHOD}_${TASKSET}_${STAMP}.log"

  echo "=== RUN chunk: method=${METHOD}, taskset=${TASKSET} ===" | tee -a "$LOG"

  # 啟動 python（放背景），stdout/stderr 同時寫 log + 螢幕
  python -u run_exp.py \
    --method "$METHOD" \
    --taskset "$TASKSET" \
    --chunk "$CHUNK" \
    --start "$START" \
    --end "$END" \
    --timeout "$TIMEOUT" \
    --sleep_after "$SLEEP_AFTER" \
    2>&1 | tee -a "$LOG" &
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

      # 先嘗試優雅結束
      kill -TERM "$PID" 2>/dev/null || true
      sleep "$KILL_GRACE"

      # 還活著就強制 kill
      if kill -0 "$PID" 2>/dev/null; then
        echo "=== Force kill PID $PID ===" | tee -a "$LOG"
        kill -KILL "$PID" 2>/dev/null || true
      fi

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