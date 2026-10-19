#!/usr/bin/env bash
# Bước 15 — đưa các lượt chạy đã xong (và file phân tích của bước 13) lên Weights & Biases
# bằng scripts/wandb_upload.py. Lượt đã được log trực tiếp (--wandb, có WANDB_RUN.txt) thì bỏ qua.
#   WANDB_MODE=online  bash steps/15_wandb.sh      # cần `wandb login` trước
#   WANDB_MODE=offline bash steps/15_wandb.sh      # ghi vào ./wandb; đưa lên sau bằng: wandb sync wandb/offline-run-*
# WANDB_MODE=disabled ⇒ bỏ qua. Luôn chạy lại (không đánh dấu xong); upload lại cùng tên ghi vào đúng run cũ.
source "$(dirname "$0")/../lib/common.sh"
STEP=15_wandb
LOG_FILE="$LOG_DIR/$(date +%Y%m%d-%H%M%S)_${STEP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
need_venv; need_file data/slake/test_en_amanda.json "chạy steps/03_data.sh"
log "════ W&B ════  project=$WANDB_PROJECT entity=${WANDB_ENTITY:-<mặc định>} mode=$WANDB_MODE · log: ${LOG_FILE#"$ROOT"/}"
if [ "$WANDB_MODE" = "disabled" ]; then log "WANDB_MODE=disabled — bỏ qua"; exit 0; fi
"$PY" -c "import wandb" 2>/dev/null || die "chưa cài wandb trong .venv (chạy lại steps/02_env.sh với FORCE=1)"
"$PY" -u scripts/wandb_upload.py "$@"; rc=$?
ledger "$STEP" "$([ $rc = 0 ] && echo OK || echo FAIL)" "wandb_upload rc=$rc"
exit $rc
