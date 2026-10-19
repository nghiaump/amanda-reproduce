#!/usr/bin/env bash
# Bước 00 — kiểm máy: công cụ, GPU, dung lượng đĩa, mạng. Không cài gì, không sửa gì.
source "$(dirname "$0")/../lib/common.sh"
start_step 00_check

bad=0
for t in git curl unzip uv nvidia-smi; do
  if command -v "$t" >/dev/null; then log "✓ có $t"; else log "✗ thiếu $t"; bad=1; fi
done
command -v uv >/dev/null || log "  cài uv: curl -LsSf https://astral.sh/uv/install.sh | sh"

log "GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/    /'
free=$(gpu_free_mb "$GPU_LLAVA"); log "GPU_LLAVA=$GPU_LLAVA còn ${free:-?} MiB trống (cần ~15000)"

avail=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
log "đĩa trống ở $ROOT: ${avail} GB (cần ~30 GB: venv 7 · LLaVA-Med 15 · SLAKE 0,5 · kết quả 1–2; LLM tự dựng thêm ~17 ở HF cache)"
[ "${avail:-0}" -ge 30 ] || { log "✗ đĩa có thể không đủ"; bad=1; }

for u in "$SLAKE_URL/test.json" "https://github.com" "$SPOKE_URL/api/v1/types"; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -L -m 20 "$u")
  log "mạng $u → HTTP $code"
done
if llm_ready; then log "✓ đã có server LLM $AMANDA_LLM_MODEL ở $AMANDA_LLM_URL — có thể bỏ qua bước 05"
else log "· chưa có server LLM ở $AMANDA_LLM_URL — bước 05 sẽ dựng"; fi

[ $bad = 0 ] || die "máy chưa đủ điều kiện, xem các dòng ✗ ở trên"
finish_step
