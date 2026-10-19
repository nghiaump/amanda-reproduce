#!/usr/bin/env bash
# Bước 10 — khối chính, mỗi arm REPS lượt (mặc định 3), cùng một server LLM:
#   A1 AMANDA đầy đủ (Retriever = SPOKE, --spoke live) · A2 bỏ Retriever · A3 bỏ Explorer
#   → runs/amanda-{A1,A2,A3}/full-r<k>
# Arm thêm, chỉ chạy khi liệt kê trong EXTRA_ARMS (vd. EXTRA_ARMS="C2 B2 C5"):
#   C2 Retriever = SLAKE-KG · C5 SLAKE-KG nối vào caption từ vòng 0
#   B2 SHAM: văn bản y khoa vô quan, cùng độ dài, bắn đúng những câu C2 (cùng lượt) đã bắn ⇒ cần C2 trong EXTRA_ARMS
# Thứ tự xen kẽ theo lượt (r1 của mọi arm, rồi r2...) để trôi dạt của server chia đều cho các arm.
source "$(dirname "$0")/../lib/common.sh"
start_step 10_main
need_venv; need_file runs/amanda-C/SLAKE-native-loose.json "chạy steps/07_retrieval.sh"
need_file "$CAPTIONS" "chạy steps/06_captions.sh"
check_gpu_llava
COMMON=(--spoke live --captions "$CAPTIONS" "${WB[@]}")
want () { [[ " $EXTRA_ARMS " == *" $1 "* ]]; }
[ -n "$EXTRA_ARMS" ] && log "arm thêm: $EXTRA_ARMS"
ok=0
for r in $(seq 1 "$REPS"); do
  run_arm A1 "r$r" 1061 "${COMMON[@]}" || ok=1
  run_arm A2 "r$r" 1061 "${COMMON[@]}" || ok=1
  run_arm A3 "r$r" 1061 "${COMMON[@]}" || ok=1
  want C2 && { run_arm C2 "r$r" 1061 "${COMMON[@]}" || ok=1; }
  want C5 && { run_arm C5 "r$r" 1061 "${COMMON[@]}" || ok=1; }
  if want B2; then
    if [ -f "runs/amanda-C2/full-r$r/metrics.json" ]; then
      run_arm B2 "r$r" 1061 --sham-reference "runs/amanda-C2/full-r$r/predictions.jsonl" "${COMMON[@]}" || ok=1
    else
      log "  B2/r$r cần C2/r$r xong trước (thêm C2 vào EXTRA_ARMS)"; ok=1
    fi
  fi
done
[ $ok = 0 ] || die "còn lượt chưa xong — chạy lại bước này để tiếp tục"
finish_step
