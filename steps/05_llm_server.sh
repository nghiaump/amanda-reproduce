#!/usr/bin/env bash
# Bước 05 — server LLM cho các agent (vLLM, OpenAI-compatible).
#   bash steps/05_llm_server.sh          dựng server (nếu chưa có) rồi chờ tới khi sẵn sàng
#   bash steps/05_llm_server.sh status   xem server có trả lời không
#   bash steps/05_llm_server.sh stop     tắt server do chính bước này dựng (theo state/vllm.pid)
# Nếu AMANDA_LLM_URL đã trỏ tới một server đang chạy đúng model thì bước này không làm gì cả.
# Bước này KHÔNG ghi state/*.done: server là tiến trình sống, mỗi lần khởi động máy phải dựng lại.
source "$(dirname "$0")/../lib/common.sh"

case "${1:-start}" in
  status)
    if llm_ready; then log "✓ server $AMANDA_LLM_MODEL trả lời ở $AMANDA_LLM_URL"; else log "✗ không có server ở $AMANDA_LLM_URL"; exit 1; fi
    exit 0 ;;
  stop)
    if [ -f "$STATE_DIR/vllm.pid" ] && kill -0 "$(cat "$STATE_DIR/vllm.pid")" 2>/dev/null; then
      kill "$(cat "$STATE_DIR/vllm.pid")"; rm -f "$STATE_DIR/vllm.pid"; log "đã tắt vLLM"
    else log "không có vLLM nào do bước này dựng"; fi
    exit 0 ;;
esac

if llm_ready; then log "✓ server $AMANDA_LLM_MODEL đã sẵn sàng ở $AMANDA_LLM_URL — không dựng thêm"; exit 0; fi
[ -x .venv-vllm/bin/python ] || die "chưa có .venv-vllm — chạy steps/02b_env_vllm.sh, hoặc trỏ AMANDA_LLM_URL tới server có sẵn"
if [ -f "$STATE_DIR/vllm.pid" ] && kill -0 "$(cat "$STATE_DIR/vllm.pid")" 2>/dev/null; then
  log "vLLM (pid $(cat "$STATE_DIR/vllm.pid")) đang khởi động — chỉ chờ"
else
  free=$(gpu_free_mb "$GPU_LLM")
  log "GPU_LLM=$GPU_LLM còn ${free:-?} MiB trống (cần ~21000 với LLM_GPU_UTIL=$LLM_GPU_UTIL)"
  VLOG="$LOG_DIR/$(date +%Y%m%d-%H%M%S)_vllm.log"
  log "dựng vLLM $LLM_HF_ID → cổng $LLM_PORT, tên $AMANDA_LLM_MODEL · log ${VLOG#"$ROOT"/}"
  CUDA_VISIBLE_DEVICES=$GPU_LLM nohup .venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$LLM_HF_ID" --served-model-name "$AMANDA_LLM_MODEL" \
    --gpu-memory-utilization "$LLM_GPU_UTIL" --max-model-len "$LLM_MAX_LEN" --max-num-seqs "$LLM_MAX_SEQS" \
    --limit-mm-per-prompt '{"image":0,"video":0}' --host 127.0.0.1 --port "$LLM_PORT" > "$VLOG" 2>&1 &
  echo $! > "$STATE_DIR/vllm.pid"
  ledger 05_llm_server START "pid $! log ${VLOG#"$ROOT"/}"
fi
for i in $(seq 1 90); do                     # tối đa 15 phút (lần đầu phải tải ~17 GB trọng số)
  llm_ready && { log "✓ vLLM sẵn sàng sau ~$((i * 10)) giây"; ledger 05_llm_server READY ""; exit 0; }
  kill -0 "$(cat "$STATE_DIR/vllm.pid")" 2>/dev/null || die "vLLM đã thoát — xem log vllm mới nhất trong logs/"
  sleep 10
done
die "vLLM chưa sẵn sàng sau 15 phút — xem log; chạy lại bước này để chờ tiếp"
