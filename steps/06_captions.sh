#!/usr/bin/env bash
# Bước 06 — caption cho 96 ảnh SLAKE bằng Perceiver (LLaVA-Med) của AMANDA, qua adapter trong
# src/amanda_lab/patched_adapter.py (cờ --patched-decode) → runs/amanda-A/captions_patched.json.
# Caption phụ thuộc ảnh chứ không phụ thuộc câu hỏi, nên sinh một lần rồi mọi arm dùng chung (--captions).
# Resume: file được lưu sau mỗi 5 ảnh; chạy lại chỉ sinh ảnh còn thiếu.
# USE_SNAPSHOT=1: chép snapshot/amanda-A/captions_patched.json thay vì sinh bằng GPU.
source "$(dirname "$0")/../lib/common.sh"
start_step 06_captions
need_venv; need_file data/slake/image_root.txt "chạy steps/03_data.sh"
mkdir -p runs/amanda-A
if [ "$USE_SNAPSHOT" = "1" ]; then
  [ -f "$CAPTIONS" ] || cp snapshot/amanda-A/captions_patched.json "$CAPTIONS"
  log "USE_SNAPSHOT=1 → chép caption từ snapshot/amanda-A/captions_patched.json"
else
  need_file models/llava-med-v1.5-mistral-7b/config.json "chạy steps/04_weights.sh"
  check_gpu_llava
  CUDA_VISIBLE_DEVICES=$GPU_LLAVA "$PY" -u scripts/amanda_captions.py --patched-decode --out "$CAPTIONS" \
    || die "sinh caption đứt — chạy lại để tiếp tục"
fi
n=$("$PY" -c "import json; print(len(json.load(open('$CAPTIONS'))))")
[ "$n" -ge 96 ] || die "mới có $n/96 caption — chạy lại bước này"
log "✓ $n caption → $CAPTIONS"
finish_step
