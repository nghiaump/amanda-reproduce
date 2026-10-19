#!/usr/bin/env bash
# Bước 12 (TUỲ CHỌN) — đối chứng 2×2 bộ giải mã, n = 200: {A1, A2} × {vá, gốc}.
#   → runs/amanda-{A1,A2}/limit200-ctl{fix,raw}; --raw-decode dùng _decode_output gốc của adapter AMANDA.
source "$(dirname "$0")/../lib/common.sh"
start_step 12_decode_control
need_venv; need_file runs/amanda-C/SLAKE-native-loose.json "chạy steps/07_retrieval.sh"
need_file "$CAPTIONS" "chạy steps/06_captions.sh"
COMMON=(--spoke live --captions "$CAPTIONS" "${WB[@]}")
check_gpu_llava
ok=0
run_arm A1 ctlfix 200 "${COMMON[@]}"              || ok=1
run_arm A1 ctlraw 200 "${COMMON[@]}" --raw-decode || ok=1
run_arm A2 ctlfix 200 "${COMMON[@]}"              || ok=1
run_arm A2 ctlraw 200 "${COMMON[@]}" --raw-decode || ok=1
[ $ok = 0 ] || die "còn lượt chưa xong — chạy lại bước này để tiếp tục"
finish_step
