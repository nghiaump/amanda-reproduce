#!/usr/bin/env bash
# Bước 02 — môi trường Python cho LLaVA-Med + harness + phân tích: .venv (Python 3.10).
# torch 2.1.2+cu121 cài TRƯỚC để các gói khác không kéo torch bản mới về. Các phiên bản còn lại
# ghim theo requirements-lock.txt (danh sách đã chạy được); requirements-override.txt ghi đè pydantic (xem file đó).
source "$(dirname "$0")/../lib/common.sh"
start_step 02_env
need_file third_party/LLaVA-Med/pyproject.toml "chạy steps/01_clone.sh trước"

[ -x .venv/bin/python ] || uv venv --python 3.10 .venv || die "uv venv thất bại"
log "cài torch 2.1.2 + torchvision 0.16.2 (CUDA 12.1)"
uv pip install --python .venv/bin/python torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121 || die "cài torch thất bại"
log "cài LLaVA-Med (editable) + các gói ghim"
uv pip install --python .venv/bin/python -e third_party/LLaVA-Med \
  -c requirements-lock.txt --override requirements-override.txt || die "cài LLaVA-Med thất bại"
uv pip install --python .venv/bin/python -c requirements-lock.txt --override requirements-override.txt \
  "numpy<2" setuptools pyyaml requests pillow "openai==1.12.0" huggingface-hub wandb || die "cài gói phụ thất bại"

log "kiểm import:"
"$PY" - <<'PYEOF' || die "kiểm import thất bại"
import sys
sys.path[:0] = ["third_party/AMANDA/src", "src"]
import torch, transformers, llava, yaml, openai, wandb
from amanda.pipeline import AMANDAPipeline
from amanda.evaluation.evaluate_medvqa import evaluate_records
from amanda_lab.qbank_explorer import QBank
print(f"    torch {torch.__version__} · CUDA {torch.cuda.is_available()} ({torch.cuda.device_count()} GPU)"
      f" · transformers {transformers.__version__} · openai {openai.__version__} · wandb {wandb.__version__}")
assert torch.cuda.is_available(), "torch không thấy GPU"
PYEOF
finish_step
