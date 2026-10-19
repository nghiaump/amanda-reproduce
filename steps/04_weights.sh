#!/usr/bin/env bash
# Bước 04 — trọng số LLaVA-Med v1.5 (Mistral 7B, ~15 GB) vào models/. Tải lại được: file đã đủ thì bỏ qua.
# Trọng số LLM agent (gemma AWQ, ~17 GB) chỉ cần khi tự dựng server ở bước 05; vLLM tự tải vào HF cache,
# đặt PREFETCH_LLM=1 để tải trước ở bước này.
source "$(dirname "$0")/../lib/common.sh"
start_step 04_weights
need_venv
mkdir -p models
"$PY" - <<PYEOF || die "tải trọng số thất bại (chạy lại để tải tiếp)"
from huggingface_hub import snapshot_download
p = snapshot_download("$LLAVA_HF_ID", local_dir="models/llava-med-v1.5-mistral-7b")
print("    LLaVA-Med →", p)
if "${PREFETCH_LLM:-0}" == "1":
    print("    LLM →", snapshot_download("$LLM_HF_ID"))
PYEOF
need_file models/llava-med-v1.5-mistral-7b/config.json
log "dung lượng: $(du -sh models/llava-med-v1.5-mistral-7b | cut -f1)"
finish_step
