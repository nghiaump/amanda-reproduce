# amanda-reproduce — chạy lại thực nghiệm AMANDA trên SLAKE

## 1. Mục đích

Gói này chạy lại các thực nghiệm tái lập **AMANDA** (*Agentic Medical Knowledge Augmentation for Data-Efficient
Medical VQA*, EMNLP 2025) trên tập test tiếng Anh của **SLAKE**, từng bước, có log theo thời gian và chạy tiếp được
khi bị ngắt. Mã nguồn của tác giả được clone ở commit ghim và dùng nguyên trạng; mọi phần thêm vào nằm trong
`src/amanda_lab/` và `scripts/`.

Cấu hình chi tiết của từng arm: [`docs/experiments.md`](docs/experiments.md).

---

## 2. Phần cứng

| | yêu cầu |
|---|---|
| GPU | NVIDIA, 2 card × 24 GB: một card cho LLaVA-Med (cần khoảng 15 GB trống, `GPU_LLAVA`), một card cho server vLLM (khoảng 21 GB, `GPU_LLM`). Nếu đã có server LLM ở nơi khác (mục 7, bước 05) thì chỉ cần 1 card cho LLaVA-Med. |
| Đĩa | khoảng 30 GB trong thư mục gói (môi trường Python, trọng số LLaVA-Med, SLAKE, kết quả), cộng khoảng 17 GB trong cache Hugging Face nếu tự dựng vLLM |
| Mạng | GitHub, Hugging Face, `pypi.org`, `download.pytorch.org`; SPOKE (`spoke.rbvi.ucsf.edu`) cho bước 07–12; `wandb.ai` nếu dùng W&B ở chế độ online |
| Thời gian | tổng cộng khoảng một ngày GPU cho các bước bắt buộc; thời gian từng bước ở mục 7 |

Thời gian ở mục 7 là ước lượng thô trên 2 × RTX 3090 với `WORKERS=6`, chỉ để lên lịch.

---

## 3. Phần mềm và môi trường

| | phiên bản |
|---|---|
| Hệ điều hành | Linux x86_64 |
| Driver NVIDIA | hỗ trợ CUDA 12.1 (wheel torch `cu121`) |
| Công cụ | `git`, `curl`, `unzip`, `nvidia-smi`, [`uv`](https://docs.astral.sh/uv/) (cài: `curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| `.venv/` | Python 3.10 — LLaVA-Med, harness, phân tích, `wandb`; torch 2.1.2+cu121, torchvision 0.16.2, transformers 4.36.2, openai 1.12.0, wandb 0.30.0 |
| `.venv-vllm/` | Python 3.12 — vLLM 0.19.1; chỉ cần khi tự dựng server LLM (bước 02b, 05) |

Bước 02 tạo `.venv` bằng `uv`. Các gói được ghim theo:

- `requirements-lock.txt` — phiên bản của các gói phụ thuộc (dùng làm constraint, `-c`);
- `requirements-override.txt` — ghi đè `pydantic` (LLaVA-Med khai báo `pydantic<2`, `wandb` cần `pydantic` 2).

Không cần tự kích hoạt môi trường: mọi script trong `steps/` gọi thẳng `.venv/bin/python` với `PYTHONPATH=src`.
Khi chạy tay một script Python, dùng cùng dạng đó từ thư mục gốc của gói, ví dụ
`PYTHONPATH=src .venv/bin/python scripts/summarize.py`.

---

## 4. Cấu trúc thư mục

```
amanda-reproduce/
├── README.md · LICENSE
├── config.env                 cấu hình chung (mục 5)
├── run_all.sh                 chạy mọi bước theo thứ tự, bỏ qua bước đã xong
├── status.sh                  xem tiến độ
├── steps/                     00_check … 13_analyze, 15_wandb — mỗi bước một script bash
├── lib/common.sh              hàm dùng chung: log theo thời gian, đánh dấu bước xong, chạy tiếp, run_arm
├── requirements-lock.txt      phiên bản gói đã ghim
├── requirements-override.txt  ghi đè pydantic khi cài
├── src/amanda_lab/
│   ├── run_pipeline.py        chạy AMANDAPipeline của tác giả với mọi arm (A1–A4, B2, C2, C5, F0, F1, FQ)
│   ├── patched_adapter.py     PatchedLLaVAMedAdapter (kế thừa LLaVAMedAdapter của AMANDA)
│   ├── slake_kg.py            SlakeKGClient: đọc knowledge graph của SLAKE qua giao diện của SPOKEClient
│   ├── qbank_explorer.py      QuestionBankExplorer cho arm FQ
│   └── wandb_log.py           ghi một lượt chạy lên Weights & Biases
├── scripts/
│   ├── prepare_slake.py       chuyển định dạng SLAKE, tải KG của SLAKE, self-test           (bước 03)
│   ├── amanda_captions.py     sinh caption 96 ảnh                                          (bước 06)
│   ├── amanda_retriever_probe.py  trích entity, truy vấn SPOKE, lưu cache                  (bước 07)
│   ├── c0_slake_fire.py       Retriever trên SPOKE và SLAKE-KG; lớp CachedSpokeReplay      (bước 07)
│   ├── c1_context_quality.py  context truy hồi có chứa đáp án không                         (bước 07)
│   ├── amanda_run.py          arm A0 (LLaVA-Med một bước)                                  (bước 08)
│   ├── qbank_dryrun.py        chạy khô arm FQ với LLM giả                                  (bước 11)
│   ├── summarize.py · closed_bias_audit.py · verbosity_probe.py · amanda_explorer_audit.py
│   ├── qbank_analyze.py · summarize_runs.py · ablation_recompute.py · retriever_investigation.py   (bước 13)
│   ├── wandb_upload.py        đưa lượt chạy đã xong lên W&B                                (bước 15)
│   └── qbank_graph.py         hàm dùng chung cho qbank_analyze.py; công cụ kiểm/đóng băng bank
├── data/qbank/                qbank_v1.yaml (question bank) · FROZEN (sha256) · NOTICE.md (nguồn trích dẫn)
├── snapshot/amanda-A/         captions_patched.json · spoke_cache/ (phản hồi SPOKE API đã lưu) — bước 06–07
├── docs/experiments.md        cấu hình từng arm
│
│   (các thư mục dưới đây do các bước tạo ra, không nằm trong repo)
├── third_party/               AMANDA, LLaVA-Med ở commit ghim                ← bước 01
├── .venv/  .venv-vllm/        môi trường Python                              ← bước 02, 02b
├── data/slake/                SLAKE: test.json, ảnh, KG, test_en_amanda.json ← bước 03
├── models/                    trọng số LLaVA-Med v1.5 Mistral 7B             ← bước 04
├── runs/                      đầu ra từng lượt chạy                          ← bước 06–12
├── results/                   đầu ra phân tích                               ← bước 13
├── logs/                      log theo thời gian (mục 10)
├── state/                     <bước>.done — bước đã xong
└── wandb/                     run W&B ở chế độ offline                       ← bước 09–12, 15
```

---

## 5. Cấu hình

Mọi giá trị nằm trong `config.env`. Có ba cách đổi, ưu tiên từ trên xuống:

1. biến môi trường ngay trên dòng lệnh: `REPS=1 bash steps/10_main.sh`;
2. file `config.local.env` (tạo mới, không đưa vào git), mỗi dòng một `BIẾN=giá_trị`;
3. sửa giá trị mặc định trong `config.env`.

| biến | mặc định | ý nghĩa |
|---|---|---|
| `GPU_LLAVA` | `1` | chỉ số GPU chạy LLaVA-Med |
| `GPU_LLM` | `0` | chỉ số GPU chạy vLLM (chỉ khi tự dựng ở bước 05) |
| `AMANDA_LLM_URL` | `http://127.0.0.1:4001/v1` | địa chỉ server OpenAI-compatible cho Reasoner, Evaluator, Explorer, trích entity |
| `AMANDA_LLM_MODEL` | `gemma-4-26b` | tên model mà server công bố (`served-model-name`) |
| `LLM_HF_ID` | `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` | trọng số LLM khi tự dựng vLLM |
| `LLM_PORT` · `LLM_MAX_LEN` · `LLM_GPU_UTIL` · `LLM_MAX_SEQS` | `4001` · `8192` · `0.88` · `16` | tham số vLLM ở bước 05 |
| `WORKERS` | `6` | số câu xử lý song song trong một lượt |
| `REPS` | `3` | số lượt lặp cho mỗi arm ở bước 10 và 11 |
| `EXTRA_ARMS` | trống | arm thêm cho bước 10, cách nhau bởi dấu cách: `C2 B2 C5` |
| `MAX_TRIES` | `3` | số lần tự chạy tiếp một lượt bị đứt |
| `LLM_WAIT_MIN` | `30` | số phút tối đa chờ server LLM trước mỗi lượt |
| `USE_SNAPSHOT` | `0` | `1` = bước 06 chép caption từ `snapshot/`, bước 07 nạp sẵn cache SPOKE từ `snapshot/` |
| `WANDB_PROJECT` | `amanda-slake-reproduction` | project W&B |
| `WANDB_ENTITY` | trống | tài khoản hoặc team W&B; trống = tài khoản mặc định của khoá API |
| `WANDB_MODE` | `offline` | `online` · `offline` · `disabled` (mục 9) |
| `AMANDA_GIT` · `AMANDA_COMMIT` | `REAL-Lab-NU/AMANDA` · `12fd7e41…` | mã nguồn AMANDA |
| `LLAVA_GIT` · `LLAVA_COMMIT` | `microsoft/LLaVA-Med` · `30697ca5…` | mã nguồn LLaVA-Med |
| `LLAVA_HF_ID` | `microsoft/llava-med-v1.5-mistral-7b` | trọng số LLaVA-Med |
| `SLAKE_URL` · `SLAKE_TEST_SHA256` | Hugging Face `BoKelvin/SLAKE` · sha256 của `test.json` | nguồn dữ liệu và mã kiểm |
| `SPOKE_URL` | `https://spoke.rbvi.ucsf.edu` | API SPOKE |

Ví dụ `config.local.env` khi đã có server LLM ở cổng khác:

```bash
AMANDA_LLM_URL=http://127.0.0.1:8000/v1
AMANDA_LLM_MODEL=gemma-4-26b
GPU_LLAVA=0
```

---

## 6. Chạy nhanh

```bash
git clone https://github.com/nghiaump/amanda-reproduce.git amanda-reproduce
cd amanda-reproduce
bash steps/00_check.sh                  # kiểm máy, không cài gì
bash run_all.sh                         # chạy mọi bước bắt buộc theo thứ tự
bash status.sh                          # xem tiến độ, gọi bao nhiêu lần cũng được
```

Các biến thể của `run_all.sh`:

```bash
INCLUDE_OPTIONAL=1 bash run_all.sh            # thêm bước 12
EXTRA_ARMS="C2 B2 C5" bash run_all.sh         # thêm arm C2, B2, C5 vào bước 10
nohup bash run_all.sh > /dev/null 2>&1 &      # chạy nền (nên dùng nohup hoặc tmux)
```

`run_all.sh` chạy lần lượt 00 → 01 → 02 → 03 → 04 → (02b → 05 nếu chưa có server LLM) → 06 → 07 → 08 → 09 → 10 →
11 → (12) → 13 → 15. Bị ngắt ở đâu thì chạy lại đúng lệnh `bash run_all.sh`: bước đã xong được bỏ qua, lượt đang dở
chạy tiếp từ câu cuối đã ghi.

### Lệnh cho từng nhóm thực nghiệm

Mỗi nhóm thực nghiệm trong báo cáo tương ứng với một nhóm run cùng tên trên W&B (mục 9).

| nhóm W&B | arm | lệnh |
|---|---|---|
| `baseline` | A0, A0-bug | `bash steps/08_baseline.sh` |
| `main` | A1 × 3 | `bash steps/10_main.sh` |
| `ablation` | A2 × 3, A3 × 3, A4 | `bash steps/09_gate.sh` (A4) · `bash steps/10_main.sh` (A2, A3) |
| `graph` | C2 × 3, B2 × 3 | `EXTRA_ARMS="C2 B2" bash steps/10_main.sh` |
| `qbank` | F0, F1, FQ × 3 | `bash steps/11_forced.sh` |
| `control` | A1 × 3 chỉ đọc cache SPOKE · A1 × 3 với caption sinh bằng adapter gốc | xem các lệnh ngay dưới bảng |
| `analysis` | không chạy mô hình | `bash steps/13_analyze.sh` → `results/`, rồi `bash steps/15_wandb.sh` |

Hai bộ của nhóm `control` chạy tay bằng `run_pipeline.py` (dạng lệnh đầy đủ ở mục 7), mỗi bộ ba lượt `k = 1, 2, 3`:

```bash
# A1 chỉ đọc cache SPOKE của bước 07, không gọi API cho entity thiếu
… --arm A1 --tag spokecache-r<k> --spoke cache --captions runs/amanda-A/captions_patched.json --wandb
# A1 với caption sinh bằng adapter gốc (chưa vá giải mã)
CUDA_VISIBLE_DEVICES=$GPU_LLAVA PYTHONPATH=src .venv/bin/python scripts/amanda_captions.py --out runs/amanda-A/captions_unpatched.json
… --arm A1 --tag oldcaption-r<k> --spoke cache --captions runs/amanda-A/captions_unpatched.json --wandb
```

Đối chứng bộ giải mã 2×2 trên 200 câu (bước 12, tuỳ chọn): `bash steps/12_decode_control.sh`.

---

## 7. Từng bước

Mỗi bước chạy bằng `bash steps/<tên>.sh` từ thư mục gốc của gói. Bước đã xong ghi `state/<tên>.done` và tự bỏ qua
lần sau; `FORCE=1 bash steps/<tên>.sh` để chạy lại. Các bước 07 và 09–12 cần server LLM (bước 05); mỗi lượt tự chờ
server tối đa `LLM_WAIT_MIN` phút.

Các bước chạy arm (09–12) gọi `src/amanda_lab/run_pipeline.py` qua hàm `run_arm` trong `lib/common.sh`, theo dạng:

```bash
CUDA_VISIBLE_DEVICES=$GPU_LLAVA PYTHONPATH=src .venv/bin/python -u -m amanda_lab.run_pipeline \
  --arm <ARM> --tag <TAG> [--limit N] --workers $WORKERS \
  --llm-base-url $AMANDA_LLM_URL --llm-model $AMANDA_LLM_MODEL \
  [--spoke live] --captions runs/amanda-A/captions_patched.json [--wandb] [cờ riêng của arm]
```

`--wandb` được thêm khi `WANDB_MODE` khác `disabled`. Mỗi lượt ghi vào `runs/amanda-<ARM>/full-<TAG>/` (hoặc
`limit<N>-<TAG>/`): `predictions.jsonl`, `trace.jsonl` (ghi sau từng câu), `metrics.json` (khi đủ câu) và
`WANDB_RUN.txt` (khi đã log lên W&B).

### Bước 00 — kiểm máy (vài giây)

```bash
bash steps/00_check.sh
```

Chỉ kiểm, không cài gì: có `git curl unzip uv nvidia-smi`, VRAM trống trên `GPU_LLAVA`, dung lượng đĩa, đường mạng
tới Hugging Face, GitHub, SPOKE, và đã có server LLM ở `AMANDA_LLM_URL` chưa. Ghi log vào `logs/`.

### Bước 01 — mã nguồn bên thứ ba (khoảng 1 phút)

```bash
bash steps/01_clone.sh
```

- Đầu vào: `AMANDA_GIT`/`AMANDA_COMMIT`, `LLAVA_GIT`/`LLAVA_COMMIT`.
- Đầu ra: `third_party/AMANDA/`, `third_party/LLaVA-Med/` ở đúng commit ghim. Dừng nếu một repo có thay đổi cục bộ.

### Bước 02 — môi trường Python (5–15 phút lần đầu)

```bash
bash steps/02_env.sh
```

- Tạo `.venv` (Python 3.10) bằng `uv`; cài torch 2.1.2 + torchvision 0.16.2 từ index `cu121`; cài
  `third_party/LLaVA-Med` dạng editable và các gói phụ (`wandb`, `openai`, `pyyaml`, `setuptools`…) với
  `-c requirements-lock.txt --override requirements-override.txt`.
- Kết thúc bằng một lần kiểm import (`amanda`, `llava`, `amanda_lab`, `wandb`) và kiểm torch thấy GPU.

### Bước 02b — môi trường vLLM (tuỳ chọn, 5–10 phút)

```bash
bash steps/02b_env_vllm.sh
```

Chỉ cần khi tự dựng server LLM. Tạo `.venv-vllm` (Python 3.12) và cài `vllm==0.19.1`.

### Bước 03 — dữ liệu SLAKE (khoảng 1 phút)

```bash
bash steps/03_data.sh
```

- Tải `test.json` và `imgs.zip` từ `SLAKE_URL` (có resume), kiểm sha256 của `test.json` với `SLAKE_TEST_SHA256`,
  giải nén ảnh.
- Gọi `scripts/prepare_slake.py --self-test`: tải `KG.zip` của SLAKE, sinh định dạng AMANDA, kiểm số câu, số ảnh,
  ảnh thiếu.
- Đầu ra: `data/slake/test.json`, `data/slake/imgs/`, `data/slake/KG/*.csv`, `data/slake/test_en_amanda.json`,
  `data/slake/image_root.txt`.

### Bước 04 — trọng số LLaVA-Med (khoảng 15 GB)

```bash
bash steps/04_weights.sh                 # PREFETCH_LLM=1 để tải luôn trọng số LLM_HF_ID vào cache Hugging Face
```

Đầu ra: `models/llava-med-v1.5-mistral-7b/` (tải bằng `huggingface_hub.snapshot_download`). Bị đứt thì chạy lại.

### Bước 05 — server LLM cho các agent

Chọn một trong hai cách.

**(a) Dùng server có sẵn.** Bất kỳ server nào nói giao thức OpenAI `/v1/chat/completions` và phục vụ model có tên
`AMANDA_LLM_MODEL` đều dùng được. Đặt `AMANDA_LLM_URL`, `AMANDA_LLM_MODEL` trong `config.local.env` rồi kiểm:

```bash
bash steps/05_llm_server.sh status       # in ✓ nếu server trả lời
```

**(b) Tự dựng vLLM trên `GPU_LLM`:**

```bash
bash steps/02b_env_vllm.sh
bash steps/05_llm_server.sh              # dựng rồi chờ sẵn sàng (lần đầu tải trọng số, chờ tối đa 15 phút)
bash steps/05_llm_server.sh stop         # tắt khi xong việc
```

Lệnh dựng: `.venv-vllm/bin/python -m vllm.entrypoints.openai.api_server --model $LLM_HF_ID
--served-model-name $AMANDA_LLM_MODEL --gpu-memory-utilization $LLM_GPU_UTIL --max-model-len $LLM_MAX_LEN
--max-num-seqs $LLM_MAX_SEQS --limit-mm-per-prompt '{"image":0,"video":0}' --host 127.0.0.1 --port $LLM_PORT`.
PID lưu ở `state/vllm.pid`, log ở `logs/<ngày-giờ>_vllm.log`. Server là tiến trình sống nên bước 05 không ghi
`.done`; khởi động lại máy thì phải dựng lại.

### Bước 06 — caption cho 96 ảnh (GPU, 5–10 phút)

```bash
bash steps/06_captions.sh
```

- Gọi `scripts/amanda_captions.py --patched-decode --out runs/amanda-A/captions_patched.json`: Perceiver của
  AMANDA mô tả từng ảnh qua adapter trong `src/amanda_lab/patched_adapter.py`, với `random.seed(0)`.
- Đầu vào: `data/slake/test.json`, ảnh, `models/llava-med-v1.5-mistral-7b/`.
- Đầu ra: `runs/amanda-A/captions_patched.json` (lưu sau mỗi 5 ảnh; chạy lại chỉ sinh ảnh còn thiếu). Mọi bước sau
  đọc file này qua `--captions`.
- `USE_SNAPSHOT=1 bash steps/06_captions.sh`: chép `snapshot/amanda-A/captions_patched.json` thay vì sinh bằng GPU.

### Bước 07 — Retriever trên SPOKE và SLAKE-KG (khoảng 10 phút, cần server LLM)

```bash
bash steps/07_retrieval.sh
```

1. `scripts/amanda_retriever_probe.py --arm A1 --workers 4`: LLM trích entity từ câu hỏi, truy vấn SPOKE
   `/api/v1/neighborhood/...`, lưu mọi phản hồi vào `runs/amanda-A/spoke_cache/`. Đầu ra `runs/amanda-A/A1.json`;
   chạy tiếp theo từng câu qua `runs/amanda-A/A1.partial.jsonl`.
2. `scripts/c0_slake_fire.py`: Retriever của tác giả trên cùng bộ entity, với SPOKE (từ cache) và với KG của SLAKE
   → `runs/amanda-C/*.json`, gồm `C0_summary.json` và `SLAKE-native-loose.json`.
3. `scripts/c1_context_quality.py` → `runs/amanda-C/C1_summary.json`.

Nếu SPOKE không trả HTTP 200, hoặc khi `USE_SNAPSHOT=1`, bước này chép trước `snapshot/amanda-A/spoke_cache/`
vào `runs/amanda-A/spoke_cache/`. Entity không có trong cache và không hỏi được SPOKE được đếm là lỗi (in ở cuối
mục 1) và không được ghi vào cache.

### Bước 08 — mốc A0 (GPU, 2 lượt × 15–30 phút, không cần server LLM)

```bash
bash steps/08_baseline.sh
```

- `scripts/amanda_run.py --arm A0` → `runs/amanda-A0/full/`
- `scripts/amanda_run.py --arm A0 --tag BUG-rawdecode --raw-decode` → `runs/amanda-A0/full-BUG-rawdecode/`

### Bước 09 — cổng: A4 (GPU + LLM, 30–45 phút)

```bash
bash steps/09_gate.sh
```

- `run_pipeline --arm A4 --tag g1 --spoke live --captions runs/amanda-A/captions_patched.json` →
  `runs/amanda-A4/full-g1/`.
- Đầu vào: caption của bước 06; `runs/amanda-A/A1.json` và `runs/amanda-C/` của bước 07.

### Bước 10 — khối chính (GPU + LLM, khoảng 30–45 phút mỗi lượt)

```bash
bash steps/10_main.sh
EXTRA_ARMS="C2 B2 C5" bash steps/10_main.sh     # thêm arm tuỳ chọn
```

- Với k = 1 … `REPS`, chạy xen kẽ A1, A2, A3 (rồi C2, C5, B2 nếu có trong `EXTRA_ARMS`), mỗi lượt với
  `--tag r<k> --spoke live --captions runs/amanda-A/captions_patched.json`.
- B2 thêm `--sham-reference runs/amanda-C2/full-r<k>/predictions.jsonl`, nên cần C2 cùng lượt chạy xong trước.
- Đầu ra: `runs/amanda-{A1,A2,A3}/full-r<k>/` (và `runs/amanda-{C2,C5,B2}/full-r<k>/`).
- `--spoke live` đọc cache của bước 07 trước, rồi `runs/amanda-A/spoke_cache_live/`; entity còn thiếu thì gọi SPOKE
  API và lưu vào `spoke_cache_live/`.

### Bước 11 — chế độ ép Explorer (GPU + LLM, khoảng 1–1.5 giờ mỗi lượt)

```bash
bash steps/11_forced.sh
```

- Kiểm sha256 của `data/qbank/qbank_v1.yaml` với `data/qbank/FROZEN`, rồi chạy `scripts/qbank_dryrun.py` (LLM giả,
  log ở `logs/runs/*_qbank-dryrun.log`).
- Với k = 1 … `REPS`, chạy F0, F1, FQ với `--tag r<k> --captions runs/amanda-A/captions_patched.json`.
- Đầu ra: `runs/amanda-{F0,F1,FQ}/full-r<k>/`.

### Bước 12 — đối chứng bộ giải mã (tuỳ chọn, GPU + LLM, khoảng 1 giờ)

```bash
bash steps/12_decode_control.sh          # hoặc INCLUDE_OPTIONAL=1 bash run_all.sh
```

- `--limit 200` với `--spoke live --captions …`: A1 và A2, mỗi arm một lượt với adapter vá (`--tag ctlfix`) và một
  lượt với adapter gốc (`--tag ctlraw --raw-decode`).
- Đầu ra: `runs/amanda-{A1,A2}/limit200-ctl{fix,raw}/`.

### Bước 13 — phân tích (CPU, vài phút; chạy lúc nào cũng được)

```bash
bash steps/13_analyze.sh
```

Đọc `runs/`, bỏ qua lượt chưa đủ câu, ghi vào `results/`. Bước này luôn chạy lại, không ghi `.done`.

| script | đầu vào | đầu ra |
|---|---|---|
| `scripts/summarize.py` | `runs/amanda-*/*/metrics.json` | `results/summary.md`, `results/summary.json` |
| `scripts/closed_bias_audit.py` | `predictions.jsonl` của A0, A1–A4, B2, C2, C5 | `results/closed_bias_audit.txt` |
| `scripts/verbosity_probe.py` | A0 `full`, các arm ở `full-r1` | `results/verbosity.txt`, `results/verbosity.json` |
| `scripts/amanda_explorer_audit.py` | `trace.jsonl` của A1, F1, FQ ở `full-r1` | `results/explorer_audit.txt` |
| `scripts/qbank_analyze.py --tex` (khi có đủ F0, F1, FQ) | `runs/amanda-{F0,F1,FQ}/full-r*` | `results/qbank_analysis.txt`, `results/qbank_numbers*.tex` |
| đoạn Python trong `steps/13_analyze.sh` | `runs/amanda-C/C0_summary.json` | `results/retriever_fire.txt` |
| `scripts/summarize_runs.py` (khi có A1 `full-r*`, A0, A4) | A1 `full-r*`, A0 `full`, A4 `full-g1` | `results/runs_summary.json`, `results/runs_summary.txt` |
| `scripts/ablation_recompute.py` (như trên) | `runs/amanda-{A1,A2,A3}/full-r*`, A4 `full-g1` | `results/ablation_summary.json`, `results/ablation_summary.txt` |
| `scripts/retriever_investigation.py` (cần server LLM; chạy một lần, `FORCE=1` để chạy lại) | `trace.jsonl` của A1 `full-r*`, caption, cache SPOKE | `results/retriever_investigation.json`, `results/retriever_investigation.txt` |

### Bước 15 — đưa lên Weights & Biases (vài phút)

```bash
bash steps/15_wandb.sh                   # theo WANDB_MODE trong cấu hình
bash steps/15_wandb.sh --list            # chỉ liệt kê lượt đã xong và trạng thái W&B
bash steps/15_wandb.sh --only A1-r1 analysis
```

Gọi `scripts/wandb_upload.py`: mỗi lượt đã xong trong `runs/` (chưa có `WANDB_RUN.txt`) thành một run W&B tên
`<ARM>-<TAG>`; thêm các run phân tích từ `results/runs_summary.json` và `results/retriever_investigation.json` nếu
đã có. `WANDB_MODE=disabled` thì bỏ qua. Chi tiết ở mục 9.

---

## 8. Danh sách arm

Cấu hình đầy đủ: [`docs/experiments.md`](docs/experiments.md).

| arm | bước | cờ chính | thành phần bị bỏ hoặc thay |
|---|---|---|---|
| A0 | 08 | `scripts/amanda_run.py --arm A0` | chỉ LLaVA-Med trả lời một bước; không có agent nào khác |
| A0-bug | 08 | `… --arm A0 --tag BUG-rawdecode --raw-decode` | như A0, adapter `LLaVAMedAdapter` gốc |
| A4 | 09 | `--arm A4 --tag g1` | Perceiver → `NoPerceiver` (trả chuỗi rỗng) |
| A1 | 10 | `--arm A1 --tag r<k>` | không thay gì (AMANDA đầy đủ, Retriever = SPOKE) |
| A2 | 10 | `--arm A2 --tag r<k>` | Retriever → `NullRetriever` |
| A3 | 10 | `--arm A3 --tag r<k>` | Explorer → `NoExplorer` |
| C2 | 10 (`EXTRA_ARMS`) | `--arm C2 --tag r<k>` | nguồn tri thức SPOKE → KG của SLAKE (`SlakeKGClient`) |
| C5 | 10 (`EXTRA_ARMS`) | `--arm C5 --tag r<k>` | Retriever → `NullRetriever`; context SLAKE-KG nối vào caption từ vòng 0 (`EagerKGAdapter`) |
| B2 | 10 (`EXTRA_ARMS`) | `--arm B2 --tag r<k> --sham-reference runs/amanda-C2/full-r<k>/predictions.jsonl` | Retriever → `ShamRetriever` (văn bản y khoa không liên quan, theo vị trí và độ dài context của C2) |
| F0 | 11 | `--arm F0 --tag r<k>` | chế độ ép (`ForcedExplorePipeline`), Retriever tắt, Explorer → `NoExplorer` |
| F1 | 11 | `--arm F1 --tag r<k>` | chế độ ép, Retriever tắt, Explorer gốc |
| FQ | 11 | `--arm FQ --tag r<k>` | chế độ ép, Retriever tắt, Explorer → `QuestionBankExplorer` (`data/qbank/qbank_v1.yaml`) |
| A1, A2 ctlfix / ctlraw | 12 | `--limit 200 --tag ctlfix` / `--tag ctlraw --raw-decode` | như A1, A2 trên 200 câu đầu, adapter vá hoặc gốc |

Mọi arm của bước 09–12 nhận thêm `--captions runs/amanda-A/captions_patched.json`; A1–A4, C2, C5, B2 và các lượt
của bước 12 nhận thêm `--spoke live`.

---

## 9. Weights & Biases

Ba chế độ, đặt bằng `WANDB_MODE` (trong `config.env`, `config.local.env` hoặc trên dòng lệnh):

| `WANDB_MODE` | hành vi |
|---|---|
| `offline` (mặc định) | `--wandb` ghi run vào `./wandb/` trên máy, không cần tài khoản. Đưa lên sau bằng `.venv/bin/wandb sync wandb/offline-run-*` |
| `online` | log trực tiếp lên `wandb.ai` |
| `disabled` | không thêm `--wandb`; bước 15 bỏ qua |

Cài đặt cho chế độ online, làm một lần:

```bash
.venv/bin/wandb login                     # dán API key lấy ở https://wandb.ai/authorize
cat >> config.local.env <<'EOF'
WANDB_MODE=online
WANDB_ENTITY=<tài khoản hoặc team>
WANDB_PROJECT=amanda-slake-reproduction
EOF
```

Không ghi API key vào `config.env` hay bất kỳ file nào trong repo; `wandb login` lưu khoá ở `~/.netrc`.

**Log trực tiếp.** Khi `WANDB_MODE` khác `disabled`, các bước 09–12 thêm `--wandb` cho `run_pipeline.py`. Mỗi lượt
là một run tên `<ARM>-<TAG>` (id cố định theo tên, nên chạy tiếp một lượt sẽ ghi vào đúng run cũ), nhóm `main`
(A1), `ablation` (A2, A3, A4), `graph` (C2, C5, B2) hoặc `qbank` (F0, F1, FQ). Trong lúc chạy, cứ 20 câu
log một lần điểm lũy kế (chấm bằng hàm của tác giả) và tốc độ. Khi xong, run nhận bảng dự đoán từng câu và một
artifact gồm `predictions.jsonl`, `trace.jsonl`, `metrics.json`; thư mục lượt nhận file `WANDB_RUN.txt`. Chạy tay:

```bash
WANDB_MODE=online CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python -m amanda_lab.run_pipeline \
  --arm A1 --limit 20 --tag thu --spoke live --captions runs/amanda-A/captions_patched.json --wandb
```

**Đưa lượt đã xong lên sau (bước 15).** `bash steps/15_wandb.sh` gọi `scripts/wandb_upload.py`: dựng lại đường cong
lũy kế từ `predictions.jsonl`, gắn log của lượt (`logs/runs/*_<ARM>-<TAG>.log`) và lệnh đã chạy (dựng lại từ
`metrics.json`), thẻ `post-hoc-upload`. Lượt đã có `WANDB_RUN.txt` được bỏ qua; thêm `--force` để upload lại.
Nhóm `baseline` dành cho A0, nhóm `analysis` cho các run phân tích.

**Công khai project.** Trên `wandb.ai`, mở project, vào phần chỉnh sửa project (tab *Overview* → *Edit*, hoặc biểu
tượng khoá cạnh tên project) và đặt *Project visibility* là **Public**. Link chia sẻ có dạng
`https://wandb.ai/<entity>/<project>`.

---

## 10. Log, chạy tiếp, theo dõi

```
logs/
├── <YYYYmmdd-HHMMSS>_<bước>.log       một file cho mỗi lần chạy một bước
├── <YYYYmmdd-HHMMSS>_run_all.log      log của run_all.sh
├── <YYYYmmdd-HHMMSS>_vllm.log         log server vLLM (nếu tự dựng)
├── runs/<YYYYmmdd-HHMMSS>_<ARM>-<TAG>.log   một file cho mỗi lần chạy một lượt
└── ledger.tsv                         mỗi sự kiện một dòng: thời điểm · bước · START/RUN/OK/FAIL/DONE/NOTE · chi tiết
```

Chạy tiếp ở ba mức:

1. **Bước.** Bước xong ghi `state/<bước>.done` và được bỏ qua lần sau; `FORCE=1 bash steps/<bước>.sh` để chạy lại.
2. **Lượt.** Lượt có `metrics.json` và đủ số câu thì được bỏ qua. Lượt bị đứt được thử lại tự động tới
   `MAX_TRIES` lần.
3. **Câu.** `predictions.jsonl` và `trace.jsonl` được ghi sau mỗi câu; chạy lại chỉ xử lý câu còn thiếu. Caption
   lưu sau mỗi 5 ảnh; probe Retriever lưu sau mỗi câu.

Theo dõi:

```bash
bash status.sh                                   # bước đã xong · số câu mỗi lượt · W&B · server LLM · GPU · sự kiện gần nhất
tail -f logs/runs/$(ls -t logs/runs | head -1)   # lượt đang chạy (in tốc độ và thời gian còn lại mỗi 20 câu)
column -t -s $'\t' logs/ledger.tsv | tail -20
```

Chạy lại một lượt từ đầu: chuyển thư mục của lượt ra chỗ khác (ví dụ `mv runs/amanda-A1/full-r2 /tmp/`) rồi chạy
lại bước tương ứng. Bước chưa có `.done` chừng nào còn lượt dở, nên không cần `FORCE`.

Chạy thử một lượt nhỏ bằng tay (kết quả ở `runs/amanda-A1/limit20-thu/`):

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python -m amanda_lab.run_pipeline \
  --arm A1 --limit 20 --tag thu --spoke live --captions runs/amanda-A/captions_patched.json
```

---

## 11. Xử lý sự cố

| hiện tượng | cách xử lý |
|---|---|
| `CUDA out of memory` khi nạp LLaVA-Med | Card `GPU_LLAVA` không đủ VRAM trống. Xem `nvidia-smi`, đổi `GPU_LLAVA`, chạy lại bước; lượt tiếp từ câu cuối. |
| log lượt in `LLM lỗi (…), thử lại sau …` | Server LLM không trả lời. Harness tự thử lại (chờ tăng dần tới 5 phút, 8 lần). Dựng lại server (bước 05) là lượt chạy tiếp. |
| `server LLM không lên — dừng …` | `bash steps/05_llm_server.sh status`; kiểm `AMANDA_LLM_MODEL` trùng tên model trong `curl $AMANDA_LLM_URL/models`. |
| vLLM tự dựng thoát ngay | Xem `logs/*_vllm.log`. Thường gặp: thiếu VRAM (giảm `LLM_GPU_UTIL` hoặc `LLM_MAX_LEN`) hoặc cổng bị chiếm (đổi `LLM_PORT` và `AMANDA_LLM_URL`). |
| bước 02 báo xung đột phụ thuộc | Chạy từ thư mục gốc (cần `requirements-lock.txt` và `requirements-override.txt`). Chuyển `.venv` ra chỗ khác rồi `FORCE=1 bash steps/02_env.sh`. |
| `No module named 'setuptools'` khi nạp transformers | Môi trường thiếu `setuptools`: `FORCE=1 bash steps/02_env.sh`. |
| bước 03 báo `sha256 … khác` | `test.json` trên Hugging Face đã đổi so với `SLAKE_TEST_SHA256`. |
| bước 07 báo nhiều lỗi SPOKE | SPOKE không trả lời. Chạy với `USE_SNAPSHOT=1` để nạp sẵn cache trong `snapshot/`, hoặc chờ SPOKE trở lại rồi chuyển `runs/amanda-A/A1.partial.jsonl` ra chỗ khác và `FORCE=1 bash steps/07_retrieval.sh`. |
| `spoke_stats.errors` khác 0 trong `metrics.json` | Một số lời gọi SPOKE API thất bại trong lượt `--spoke live`; kiểm mạng tới `SPOKE_URL`. Lỗi không được cache, lượt sau sẽ hỏi lại. |
| bước 11 báo `qbank_v1.yaml đã bị sửa` | `data/qbank/qbank_v1.yaml` phải khớp sha256 trong `data/qbank/FROZEN`; khôi phục file từ repo (`git checkout data/qbank/qbank_v1.yaml`). |
| `ModuleNotFoundError: amanda_lab` khi chạy tay | Chạy từ thư mục gốc với `PYTHONPATH=src` (các script tự thêm `third_party/AMANDA/src`). |
| `FileNotFoundError … runs/amanda-A/A1.json` khi chạy `run_pipeline` | Chạy bước 07 trước (`run_pipeline` nạp `scripts/c0_slake_fire.py`, vốn đọc file này). |
| W&B báo `api_key not configured` | `WANDB_MODE=online` nhưng chưa `wandb login`. Đăng nhập (mục 9) hoặc dùng `WANDB_MODE=offline`. |
| muốn chạy nhanh để thử | `REPS=1 bash steps/10_main.sh`, hoặc một lượt nhỏ bằng tay (mục 10). |

---

## 12. Giấy phép và nguồn

- **Mã nguồn của gói này** (`src/`, `scripts/`, `steps/`, `lib/`, các file `.sh`, tài liệu): giấy phép MIT, xem
  [`LICENSE`](LICENSE).
- **Mã bên thứ ba** không nằm trong repo; bước 01 clone vào `third_party/` ở commit ghim, và mỗi repo giữ giấy phép
  riêng:
  - AMANDA — <https://github.com/REAL-Lab-NU/AMANDA> @ `12fd7e41`. Ở commit này repo không kèm file `LICENSE`;
    điều kiện sử dụng theo repo gốc của tác giả.
  - LLaVA-Med — <https://github.com/microsoft/LLaVA-Med> @ `30697ca5`, theo *Microsoft Research License Terms*
    (file `LICENSE` trong repo); README của LLaVA-Med ghi dữ liệu theo CC BY-NC 4.0 và chỉ dùng cho nghiên cứu.
- **Dữ liệu SLAKE** (Liu và cộng sự, ISBI 2021) được bước 03 tải từ Hugging Face `BoKelvin/SLAKE`, theo giấy phép
  của SLAKE; không nằm trong repo.
- **Trọng số mô hình** được tải từ Hugging Face khi chạy, theo giấy phép của từng mô hình:
  `microsoft/llava-med-v1.5-mistral-7b` (bước 04) và `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` (bước 05, nếu tự dựng
  vLLM).
- **SPOKE** (<https://spoke.rbvi.ucsf.edu>): `snapshot/amanda-A/spoke_cache/` chứa phản hồi đã lưu từ API công khai
  của SPOKE, kèm theo để chạy được khi API không trả lời.
- **Question bank** (`data/qbank/qbank_v1.yaml`) chứa trích dẫn từ tài liệu đọc phim tiếng Việt; nguồn và điều kiện
  sử dụng ở [`data/qbank/NOTICE.md`](data/qbank/NOTICE.md).
- Bài báo: *AMANDA: Agentic Medical Knowledge Augmentation for Data-Efficient Medical Visual Question Answering*,
  EMNLP 2025, arXiv:2510.02328.
