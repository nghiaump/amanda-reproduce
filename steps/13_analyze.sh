#!/usr/bin/env bash
# Bước 13 — phân tích (CPU, không gọi mô hình; riêng mục 8 cần server LLM). Chạy được bất cứ lúc nào, kể cả khi
# mới xong một phần; luôn chạy lại (không đánh dấu xong). Mọi đầu ra nằm trong results/.
source "$(dirname "$0")/../lib/common.sh"
STEP=13_analyze
LOG_FILE="$LOG_DIR/$(date +%Y%m%d-%H%M%S)_${STEP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
need_venv
mkdir -p results
log "════ PHÂN TÍCH ════  log: ${LOG_FILE#"$ROOT"/}"

section () { echo; log "── $1"; }
have () { ls $1 >/dev/null 2>&1; }       # có ít nhất một file khớp mẫu

section "1. bảng tổng hợp mọi arm (scripts/summarize.py)"
"$PY" scripts/summarize.py
section "2. câu đóng: nền hằng số, bộ chấm chặt, lật đáp án (scripts/closed_bias_audit.py)"
"$PY" scripts/closed_bias_audit.py | tee results/closed_bias_audit.txt
section "3. độ dài câu trả lời và Open Recall (scripts/verbosity_probe.py)"
"$PY" scripts/verbosity_probe.py | tee results/verbosity.txt
section "4. câu hỏi con của Explorer (scripts/amanda_explorer_audit.py)"
"$PY" scripts/amanda_explorer_audit.py | tee results/explorer_audit.txt
if have "runs/amanda-F0/full-r*/metrics.json" && have "runs/amanda-F1/full-r*/metrics.json" \
   && have "runs/amanda-FQ/full-r*/metrics.json"; then
  section "5. question bank: F0 / F1 / FQ (scripts/qbank_analyze.py)"
  "$PY" scripts/qbank_analyze.py --tex | tee results/qbank_analysis.txt
fi
if [ -f runs/amanda-C/C0_summary.json ]; then
  section "6. Retriever: tỉ lệ trả context, SPOKE và SLAKE-KG (runs/amanda-C/C0_summary.json)"
  "$PY" - <<'PYEOF' | tee results/retriever_fire.txt
import json
for r in json.load(open("runs/amanda-C/C0_summary.json")):
    print(f"  {r['arm']:<26} {r['fired_cap']:>4}/1061 ({r['rate_cap']:5.2f}%) có caption · "
          f"{r['fired_nocap']:>4} không caption")
PYEOF
fi
if have "runs/amanda-A1/full-r*/metrics.json" && [ -f runs/amanda-A0/full/metrics.json ] \
   && [ -f runs/amanda-A4/full-g1/metrics.json ]; then
  section "7. các lượt A1: phễu pipeline, tri thức trong prompt (scripts/summarize_runs.py)"
  "$PY" scripts/summarize_runs.py | tee results/runs_summary.txt
  section "7b. arm cắt bỏ so với A1 (scripts/ablation_recompute.py)"
  "$PY" scripts/ablation_recompute.py | tee results/ablation_summary.txt
  # 8. cần server LLM và mạng tới SPOKE; chạy một lần (FORCE=1 để chạy lại)
  if [ ! -f results/retriever_investigation.json ] || [ "${FORCE:-0}" = "1" ]; then
    if llm_ready; then
      section "8. phát lại các lời gọi Retriever (scripts/retriever_investigation.py, cần server LLM)"
      "$PY" -u scripts/retriever_investigation.py | tee results/retriever_investigation.txt
    else
      section "8. bỏ qua scripts/retriever_investigation.py: không có server LLM ở $AMANDA_LLM_URL"
    fi
  fi
fi
echo; log "xong — đầu ra trong results/"
