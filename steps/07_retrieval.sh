#!/usr/bin/env bash
# Bước 07 — Retriever của AMANDA trên SPOKE (và trên SLAKE-KG để đối chiếu). Cần server LLM.
#   a) scripts/amanda_retriever_probe.py --arm A1 : LLM trích entity từ câu hỏi, hỏi SPOKE, lưu mọi phản hồi
#      vào runs/amanda-A/spoke_cache/ → runs/amanda-A/A1.json (resume qua runs/amanda-A/A1.partial.jsonl).
#   b) scripts/c0_slake_fire.py   : tỉ lệ Retriever trả context khi dùng SPOKE và SLAKE-KG (CPU) → runs/amanda-C/
#   c) scripts/c1_context_quality.py : context lấy được có chứa đáp án không (CPU)
#
# SPOKE không trả lời → nạp sẵn snapshot/amanda-A/spoke_cache rồi chạy tiếp; entity nào không có trong cache được
# đếm là LỖI và coi như rỗng (in ra cuối bước a).
# USE_SNAPSHOT=1: luôn nạp sẵn snapshot/amanda-A/spoke_cache trước khi chạy (a).
source "$(dirname "$0")/../lib/common.sh"
start_step 07_retrieval
need_venv; need_file "$CAPTIONS" "chạy steps/06_captions.sh"
need_file data/slake/test_en_amanda.json "chạy steps/03_data.sh"
mkdir -p runs/amanda-A/spoke_cache runs/amanda-C

if [ "$USE_SNAPSHOT" = "1" ]; then
  copy_missing snapshot/amanda-A/spoke_cache runs/amanda-A/spoke_cache
  log "USE_SNAPSHOT=1 → đã nạp sẵn cache SPOKE từ snapshot/amanda-A/spoke_cache"
else
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 30 "$SPOKE_URL/api/v1/neighborhood/Disease/name/pneumonia?depth=1")
  if [ "$code" != "200" ]; then
    log "CẢNH BÁO: SPOKE trả HTTP $code — nạp cache SPOKE từ snapshot/amanda-A/spoke_cache ($(ls snapshot/amanda-A/spoke_cache | wc -l) file)"
    copy_missing snapshot/amanda-A/spoke_cache runs/amanda-A/spoke_cache
    ledger 07_retrieval NOTE "SPOKE HTTP $code, dùng cache snapshot"
  else
    log "✓ SPOKE trả lời (HTTP 200) — truy vấn trực tiếp, lưu cache"
  fi
fi
wait_llm || die "không có server LLM (bước 05)"
"$PY" -u scripts/amanda_retriever_probe.py --arm A1 --workers 4 || die "probe đứt — chạy lại để tiếp tục"
need_file runs/amanda-A/A1.json

"$PY" -u scripts/c0_slake_fire.py || die "c0_slake_fire thất bại"
"$PY" -u scripts/c1_context_quality.py || log "c1_context_quality lỗi (không chặn các bước sau)"
need_file runs/amanda-C/SLAKE-native-loose.json
finish_step
