#!/usr/bin/env bash
# Bước 11 — chế độ ÉP: mọi câu qua ≥1 vòng Explorer, Retriever tắt, để đo riêng Explorer.
#   F0 Explorer rỗng · F1 Explorer gốc của AMANDA · FQ question bank (data/qbank/qbank_v1.yaml, đã đóng băng)
#   → runs/amanda-{F0,F1,FQ}/full-r<k>
source "$(dirname "$0")/../lib/common.sh"
start_step 11_forced
need_venv; need_file runs/amanda-C/SLAKE-native-loose.json "chạy steps/07_retrieval.sh"
need_file "$CAPTIONS" "chạy steps/06_captions.sh"
want=$(cut -d' ' -f1 data/qbank/FROZEN); got=$(sha256sum data/qbank/qbank_v1.yaml | cut -d' ' -f1)
[ "$want" = "$got" ] || die "qbank_v1.yaml đã bị sửa (sha256 $got ≠ FROZEN $want)"
log "✓ question bank khớp sha256 đóng băng"
"$PY" scripts/qbank_dryrun.py > "$LOG_DIR/runs/$(date +%Y%m%d-%H%M%S)_qbank-dryrun.log" 2>&1 \
  || die "chạy khô question bank đỏ — xem logs/runs/*qbank-dryrun.log"
log "✓ chạy khô question bank (LLM giả) xanh"
check_gpu_llava
ok=0
for r in $(seq 1 "$REPS"); do
  run_arm F0 "r$r" 1061 --captions "$CAPTIONS" "${WB[@]}" || ok=1
  run_arm F1 "r$r" 1061 --captions "$CAPTIONS" "${WB[@]}" || ok=1
  run_arm FQ "r$r" 1061 --captions "$CAPTIONS" "${WB[@]}" || ok=1
done
[ $ok = 0 ] || die "còn lượt chưa xong — chạy lại bước này để tiếp tục"
finish_step
