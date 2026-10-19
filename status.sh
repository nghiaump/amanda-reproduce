#!/usr/bin/env bash
# Xem tiến độ: bước nào xong, mỗi lượt chạy được bao nhiêu câu, server LLM, GPU, sự kiện gần nhất.
cd "$(dirname "$0")"
set -a; [ -f config.local.env ] && source config.local.env; source config.env; set +a

echo "── các bước"
for s in 00_check 01_clone 02_env 02b_env_vllm 03_data 04_weights 06_captions 07_retrieval \
         08_baseline 09_gate 10_main 11_forced 12_decode_control; do
  if [ -f "state/$s.done" ]; then printf '  ✓ %-18s %s\n' "$s" "$(cat "state/$s.done")"
  else printf '  · %-18s\n' "$s"; fi
done

echo "── các lượt chạy (câu đã có / cần)"
for d in runs/amanda-A0/full* runs/amanda-{A1,A2,A3,A4,B2,C2,C5,F0,F1,FQ}/*/; do
  [ -d "$d" ] || continue
  d=${d%/}; f="$d/predictions.jsonl"; need=1061; [[ $d == *limit200* ]] && need=200
  wb=""; [ -f "$d/WANDB_RUN.txt" ] && wb="  W&B"
  n=$( [ -f "$f" ] && grep -c . "$f" || echo 0 )
  mark="…"; [ -f "$d/metrics.json" ] && [ "$n" -ge "$need" ] && mark="✓"
  printf '  %s %-34s %5s/%s%s\n' "$mark" "${d#runs/}" "$n" "$need" "$wb"
done

echo "── server LLM"
if curl -s -m 5 "${AMANDA_LLM_URL%/}/models" | grep -q "\"$AMANDA_LLM_MODEL\""; then
  echo "  ✓ $AMANDA_LLM_MODEL @ $AMANDA_LLM_URL"; else echo "  ✗ không trả lời ở $AMANDA_LLM_URL"; fi
echo "── W&B: project=$WANDB_PROJECT entity=${WANDB_ENTITY:-<mặc định>} mode=$WANDB_MODE"
echo "── GPU"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "── 8 sự kiện gần nhất (logs/ledger.tsv)"
[ -f logs/ledger.tsv ] && tail -8 logs/ledger.tsv | sed 's/^/  /'
