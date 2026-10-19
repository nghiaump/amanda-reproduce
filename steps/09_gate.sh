#!/usr/bin/env bash
# Bước 09 — cổng kiểm harness: A4 = bỏ Perceiver (các agent không nhận thông tin gì từ ảnh), một lượt.
#   → runs/amanda-A4/full-g1
source "$(dirname "$0")/../lib/common.sh"
start_step 09_gate
need_venv; need_file runs/amanda-C/SLAKE-native-loose.json "chạy steps/07_retrieval.sh"
need_file "$CAPTIONS" "chạy steps/06_captions.sh"
check_gpu_llava
run_arm A4 g1 1061 --spoke live --captions "$CAPTIONS" "${WB[@]}" || die "A4 chưa xong — chạy lại bước này để tiếp tục"
finish_step
