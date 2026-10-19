# Hàm dùng chung cho mọi bước. Được `source` từ steps/*.sh, không chạy trực tiếp.
#
# Ba cơ chế:
#   1. LOG THEO THỜI GIAN — mỗi lần chạy một bước tạo logs/<YYYYmmdd-HHMMSS>_<bước>.log; mỗi lượt chạy
#      một arm tạo logs/runs/<YYYYmmdd-HHMMSS>_<arm>-<tag>.log. logs/ledger.tsv ghi một dòng cho mỗi sự kiện.
#   2. RESUME — bước nào xong thì ghi state/<bước>.done; chạy lại sẽ bỏ qua (FORCE=1 để chạy lại).
#      Bên trong mỗi lượt chạy, predictions.jsonl được ghi sau MỖI câu, nên đứt ở đâu chạy tiếp từ đó.
#   3. TỰ CHẠY TIẾP — một lượt bị đứt (server LLM rớt, OOM...) được thử lại tối đa MAX_TRIES lần.

set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set -a
# shellcheck disable=SC1091
[ -f "$ROOT/config.local.env" ] && source "$ROOT/config.local.env"   # nạp TRƯỚC: config.env chỉ điền biến còn trống
source "$ROOT/config.env"
set +a

PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/logs"
STATE_DIR="$ROOT/state"
mkdir -p "$LOG_DIR/runs" "$STATE_DIR" "$ROOT/runs"
export PYTHONPATH="$ROOT/src" PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export WANDB_PROJECT WANDB_ENTITY WANDB_MODE

# File caption dùng chung cho mọi arm (bước 06 sinh, các bước 07–12 đọc qua --captions).
CAPTIONS="runs/amanda-A/captions_patched.json"
export AMANDA_CAPTIONS="$(basename "$CAPTIONS")"      # tên file trong runs/amanda-A, cho các script phân tích

# Cờ W&B cho run_pipeline: WANDB_MODE=disabled ⇒ không log; online/offline ⇒ --wandb.
WB=()
if [ "${WANDB_MODE:-offline}" != "disabled" ]; then WB=(--wandb); fi

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }
die() { log "LỖI: $*"; ledger "${STEP:-?}" "FAIL" "$*"; exit 1; }
ledger() { printf '%s\t%s\t%s\t%s\n' "$(ts)" "$1" "$2" "${3:-}" >> "$LOG_DIR/ledger.tsv"; }

# start_step <tên>: mở file log có dấu thời gian, bỏ qua nếu bước đã xong (trừ khi FORCE=1).
start_step() {
  STEP="$1"
  if [ -f "$STATE_DIR/$STEP.done" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$(ts)] bước $STEP đã xong lúc $(cat "$STATE_DIR/$STEP.done") — bỏ qua (FORCE=1 để chạy lại)"
    exit 0
  fi
  LOG_FILE="$LOG_DIR/$(date +%Y%m%d-%H%M%S)_${STEP}.log"
  exec > >(tee -a "$LOG_FILE") 2>&1
  STEP_T0=$(date +%s)
  log "════ BẮT ĐẦU $STEP ════  log: ${LOG_FILE#"$ROOT"/}"
  log "máy $(hostname) · GPU_LLAVA=$GPU_LLAVA · LLM=$AMANDA_LLM_MODEL @ $AMANDA_LLM_URL · WORKERS=$WORKERS · W&B=$WANDB_MODE"
  ledger "$STEP" "START" "${LOG_FILE#"$ROOT"/}"
}

finish_step() {
  local dur=$(( $(date +%s) - STEP_T0 ))
  ts > "$STATE_DIR/$STEP.done"
  log "════ XONG $STEP sau $((dur / 3600))h$(( (dur % 3600) / 60 ))m$((dur % 60))s ════"
  ledger "$STEP" "DONE" "${dur}s"
}

need_venv() { [ -x "$PY" ] || die "chưa có .venv — chạy steps/02_env.sh trước"; }
need_file() { [ -e "$1" ] || die "thiếu $1 — ${2:-chạy các bước trước}"; }

llm_ready() {
  curl -s -m 5 "${AMANDA_LLM_URL%/}/models" 2>/dev/null | grep -q "\"$AMANDA_LLM_MODEL\""
}

# wait_llm: chờ server LLM tối đa LLM_WAIT_MIN phút (không tự dựng — dựng ở bước 05).
wait_llm() {
  llm_ready && return 0
  log "server LLM chưa sẵn sàng ở $AMANDA_LLM_URL (model $AMANDA_LLM_MODEL) — chờ tối đa $LLM_WAIT_MIN phút"
  local i
  for i in $(seq 1 $((LLM_WAIT_MIN * 2))); do
    sleep 30
    llm_ready && { log "server LLM đã sẵn sàng"; return 0; }
  done
  return 1
}

gpu_free_mb() {
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null \
    | awk -F', ' '{print $1-$2}'
}

check_gpu_llava() {
  local free; free=$(gpu_free_mb "$GPU_LLAVA")
  [ -n "$free" ] || die "không đọc được GPU $GPU_LLAVA (nvidia-smi)"
  if [ "$free" -lt 15000 ]; then
    log "CẢNH BÁO: GPU $GPU_LLAVA chỉ còn ${free} MiB trống, LLaVA-Med cần ~15000 MiB — có thể OOM"
  else
    log "GPU $GPU_LLAVA còn ${free} MiB trống"
  fi
}

# show_metrics <thư mục lượt chạy>: in Open / Closed từ metrics.json
show_metrics() {
  "$PY" - "$1/metrics.json" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))["metrics"]
print(f"Open {m['Open Recall']:.2f} · Closed {m['Closed Accuracy']:.2f}")
PYEOF
}

# copy_missing <thư mục nguồn> <thư mục đích>: chép các file đích còn thiếu, không ghi đè file đã có
copy_missing() {
  local f; mkdir -p "$2"
  for f in "$1"/*; do [ -e "$2/$(basename "$f")" ] || cp "$f" "$2/"; done
}

# n_lines <file>: số dòng không rỗng (0 nếu chưa có file)
n_lines() { [ -f "$1" ] && grep -c . "$1" || echo 0; }

# run_arm <arm> <tag> <số câu cần có> [cờ thêm cho run_pipeline...]
#   Bỏ qua nếu thư mục đã đủ câu và có metrics.json. Đứt thì thử lại (mỗi lần resume từ câu cuối đã ghi).
run_arm() {
  local arm=$1 tag=$2 need=$3; shift 3
  local sub="full"; [ "$need" != "1061" ] && sub="limit$need"
  local dir="runs/amanda-$arm/$sub${tag:+-$tag}"
  if [ -f "$dir/metrics.json" ] && [ "$(n_lines "$dir/predictions.jsonl")" -ge "$need" ]; then
    log "  $arm/$tag đã xong ($(n_lines "$dir/predictions.jsonl") câu) — bỏ qua"; return 0
  fi
  local limit=(); [ "$need" != "1061" ] && limit=(--limit "$need")
  local try
  for try in $(seq 1 "$MAX_TRIES"); do
    wait_llm || { log "  server LLM không lên — dừng $arm/$tag"; ledger "$STEP" "FAIL" "$arm/$tag: LLM"; return 1; }
    local rlog="$LOG_DIR/runs/$(date +%Y%m%d-%H%M%S)_${arm}-${tag:-notag}.log"
    log "  ▶ $arm/$tag lần $try · đã có $(n_lines "$dir/predictions.jsonl")/$need câu · log ${rlog#"$ROOT"/}"
    ledger "$STEP" "RUN" "$arm/$tag try=$try"
    CUDA_VISIBLE_DEVICES=$GPU_LLAVA "$PY" -u -m amanda_lab.run_pipeline --arm "$arm" \
      ${tag:+--tag "$tag"} "${limit[@]}" --workers "$WORKERS" \
      --llm-base-url "$AMANDA_LLM_URL" --llm-model "$AMANDA_LLM_MODEL" "$@" > "$rlog" 2>&1
    if [ -f "$dir/metrics.json" ] && [ "$(n_lines "$dir/predictions.jsonl")" -ge "$need" ]; then
      log "  ✓ $arm/$tag: $(show_metrics "$dir")"
      ledger "$STEP" "OK" "$arm/$tag"
      return 0
    fi
    log "  ⚠ $arm/$tag đứt ở $(n_lines "$dir/predictions.jsonl")/$need — dòng cuối log:"
    tail -3 "$rlog" | sed 's/^/      /'
    sleep 20
  done
  ledger "$STEP" "FAIL" "$arm/$tag sau $MAX_TRIES lần"
  log "  ✗ $arm/$tag thất bại sau $MAX_TRIES lần — chạy lại bước này để tiếp tục từ câu cuối"
  return 1
}
