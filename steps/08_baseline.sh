#!/usr/bin/env bash
# Bước 08 — mốc A0: LLaVA-Med trả lời MỘT bước, không agent nào (không cần server LLM).
#   A0            : bộ giải mã đã vá  → runs/amanda-A0/full
#   A0 BUG-rawdecode : bộ giải mã GỐC của AMANDA (cắt mất đầu câu trả lời) → runs/amanda-A0/full-BUG-rawdecode
source "$(dirname "$0")/../lib/common.sh"
start_step 08_baseline
need_venv; need_file data/slake/test_en_amanda.json "chạy steps/03_data.sh"
check_gpu_llava

a0 () {   # <tag> [cờ thêm]
  local tag=$1; shift
  local dir="runs/amanda-A0/full${tag:+-$tag}" try
  for try in $(seq 1 "$MAX_TRIES"); do
    if [ -f "$dir/metrics.json" ] && [ "$(n_lines "$dir/predictions.jsonl")" -ge 1061 ]; then
      log "  ✓ A0${tag:+/$tag}: $(show_metrics "$dir")"; return 0
    fi
    local rlog="$LOG_DIR/runs/$(date +%Y%m%d-%H%M%S)_A0-${tag:-notag}.log"
    log "  ▶ A0${tag:+/$tag} lần $try · đã có $(n_lines "$dir/predictions.jsonl")/1061 · log ${rlog#"$ROOT"/}"
    CUDA_VISIBLE_DEVICES=$GPU_LLAVA "$PY" -u scripts/amanda_run.py --arm A0 ${tag:+--tag "$tag"} "$@" > "$rlog" 2>&1
  done
  [ -f "$dir/metrics.json" ] || die "A0${tag:+/$tag} thất bại sau $MAX_TRIES lần"
}
a0 ""
a0 BUG-rawdecode --raw-decode
finish_step
