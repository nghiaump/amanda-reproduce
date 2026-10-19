#!/usr/bin/env bash
# Chạy toàn bộ quy trình theo thứ tự. Bước nào đã xong thì tự bỏ qua; lượt nào dở thì chạy tiếp từ câu cuối.
# Dừng giữa chừng (Ctrl-C, mất điện, OOM...) thì chỉ cần chạy lại đúng lệnh này.
#
#   bash run_all.sh                       # các bước bắt buộc
#   INCLUDE_OPTIONAL=1 bash run_all.sh    # thêm bước 12 (đối chứng bộ giải mã, n = 200)
#   EXTRA_ARMS="C2 B2 C5" bash run_all.sh # thêm các arm C2, B2, C5 vào bước 10
#   nohup bash run_all.sh > /dev/null 2>&1 &     # chạy nền; theo dõi bằng: bash status.sh
source "$(dirname "$0")/lib/common.sh"
RLOG="$LOG_DIR/$(date +%Y%m%d-%H%M%S)_run_all.log"
exec > >(tee -a "$RLOG") 2>&1
log "run_all bắt đầu · log ${RLOG#"$ROOT"/}"

step () { log "→ $1"; bash "steps/$1.sh" || { log "✗ dừng ở $1 — sửa lỗi rồi chạy lại run_all.sh"; exit 1; }; }

step 00_check
step 01_clone
step 02_env
step 03_data
step 04_weights
if ! llm_ready; then
  step 02b_env_vllm
  step 05_llm_server
fi
step 06_captions
step 07_retrieval
step 08_baseline
step 09_gate
step 10_main
step 11_forced
[ "${INCLUDE_OPTIONAL:-0}" = "1" ] && step 12_decode_control
step 13_analyze
step 15_wandb
log "run_all XONG — đầu ra: runs/ và results/"
