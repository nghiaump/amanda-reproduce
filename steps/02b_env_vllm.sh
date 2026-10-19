#!/usr/bin/env bash
# Bước 02b (TUỲ CHỌN) — môi trường vLLM riêng (.venv-vllm, Python 3.12) để tự dựng server LLM ở bước 05.
# Bỏ qua nếu đã có một server OpenAI-compatible phục vụ gemma-4-26b (trỏ AMANDA_LLM_URL tới đó).
source "$(dirname "$0")/../lib/common.sh"
start_step 02b_env_vllm
[ -x .venv-vllm/bin/python ] || uv venv --python 3.12 .venv-vllm || die "uv venv thất bại"
uv pip install --python .venv-vllm/bin/python "vllm==0.19.1" || die "cài vLLM thất bại"
.venv-vllm/bin/python -c "import vllm; print('    vllm', vllm.__version__)" || die "import vllm thất bại"
finish_step
