# Cấu hình các thí nghiệm

Tài liệu này chỉ mô tả **cấu hình**: mỗi arm thay đổi gì, chạy bằng cờ nào, chấm điểm bằng hàm nào, lặp bao nhiêu
lần. Kết quả và cách diễn giải nằm trong báo cáo, không nằm ở đây.

## Cấu hình chung cho mọi arm chạy qua pipeline

Các arm (trừ A0) chạy bằng `src/amanda_lab/run_pipeline.py`, dựng `AMANDAPipeline` gốc của tác giả
(`third_party/AMANDA/src/amanda/pipeline`) với:

| thành phần | giá trị |
|---|---|
| `PipelineSettings` | `max_iterations=3`, `confidence_threshold=3`, `max_sub_questions=3`, `fewshot_enabled=False` |
| MLLM (Perceiver) | LLaVA-Med v1.5 Mistral 7B (`models/llava-med-v1.5-mistral-7b`), `conv_mode=mistral_instruct`, temperature 0 |
| adapter MLLM | `PatchedLLaVAMedAdapter` (`src/amanda_lab/patched_adapter.py`); cờ `--raw-decode` dùng `LLaVAMedAdapter` gốc |
| LLM (Reasoner, Evaluator, Explorer, trích entity) | server OpenAI-compatible ở `AMANDA_LLM_URL`, model `AMANDA_LLM_MODEL`, temperature 0, `max_tokens=1024` |
| caption | đọc từ `--captions` (bước 06 sinh `runs/amanda-A/captions_patched.json` cho 96 ảnh, `random.seed(0)`) |
| dữ liệu | `data/slake/test_en_amanda.json` (tập test tiếng Anh của SLAKE), ảnh theo `data/slake/image_root.txt` |
| song song | `--workers` câu cùng lúc (`WORKERS` trong `config.env`); mọi lời gọi LLaVA-Med đi qua một khoá GPU |

Arm bỏ hoặc thay một agent bằng cách **ghi đè thuộc tính** của pipeline sau khi dựng (`pipe.explorer`,
`pipe.perceiver`), hoặc bằng retriever/adapter khác truyền vào lúc dựng. Mã trong `third_party/AMANDA/` không bị
sửa.

## Chấm điểm

Mọi lượt được chấm bằng `amanda.evaluation.evaluate_medvqa.evaluate_records` của tác giả, trên toàn bộ
`predictions.jsonl` của lượt:

- **Open Recall** cho câu `answer_type = open`;
- **Closed Accuracy** cho câu `answer_type = close`.

Kết quả ghi vào `metrics.json` của từng lượt. Các script phân tích ở bước 13 hoặc đọc `metrics.json`
(`scripts/summarize.py`), hoặc chấm lại từ `predictions.jsonl` bằng cùng hàm này (`scripts/summarize_runs.py`,
`scripts/ablation_recompute.py`). `scripts/summarize.py` tính, cho mỗi arm, trung bình và độ lệch chuẩn qua các
lượt; với mỗi cặp arm trong cùng một khối, hiệu Δ và t = Δ / (SD gộp · √(1/k₁ + 1/k₂)), SD gộp trong-arm với
df = Σ(k − 1).

## Các arm

### Mốc — bước 08 (`scripts/amanda_run.py`)

| arm | thay đổi | lệnh | thư mục |
|---|---|---|---|
| A0 | chỉ LLaVA-Med trả lời một bước qua `Perceiver.initial_answer`; không Reasoner, Evaluator, Explorer, Retriever | `amanda_run.py --arm A0` | `runs/amanda-A0/full` |
| A0-bug | như A0, với adapter gốc `LLaVAMedAdapter` | `amanda_run.py --arm A0 --tag BUG-rawdecode --raw-decode` | `runs/amanda-A0/full-BUG-rawdecode` |

### Cổng — bước 09

| arm | thay đổi | cờ | thư mục |
|---|---|---|---|
| A4 | Perceiver thay bằng `NoPerceiver`: `caption`, `initial_answer`, `followup_answer` đều trả chuỗi rỗng | `--arm A4 --tag g1 --spoke live --captions …` | `runs/amanda-A4/full-g1` |

### Khối chính — bước 10

Mặc định chạy A1, A2, A3. C2, C5, B2 chỉ chạy khi có tên trong `EXTRA_ARMS`. Mọi arm của khối nhận
`--spoke live --captions runs/amanda-A/captions_patched.json`.

| arm | Retriever | Explorer | thay đổi khác | thư mục |
|---|---|---|---|---|
| A1 | `SpokeRetriever` của tác giả, `SpokeRetrieverConfig()` mặc định, `EntityExtractor(max_entities=3)`; nguồn SPOKE qua `LiveSpokeCache` (`--spoke live`) | gốc | — | `runs/amanda-A1/full-r<k>` |
| A2 | `NullRetriever` (luôn trả `None`) | gốc | — | `runs/amanda-A2/full-r<k>` |
| A3 | như A1 | `NoExplorer` (không sinh câu hỏi con) | — | `runs/amanda-A3/full-r<k>` |
| C2 | `SpokeRetriever` với `SlakeKGClient(loose_match=True)` (`src/amanda_lab/slake_kg.py`), `allowed_predicates` = các vị từ của KG SLAKE | gốc | — | `runs/amanda-C2/full-r<k>` |
| C5 | `NullRetriever` | gốc | `EagerKGAdapter`: nối context SLAKE-KG của câu (từ `runs/amanda-C/SLAKE-native-loose.json`) vào sau caption, nên có mặt từ vòng 0 | `runs/amanda-C5/full-r<k>` |
| B2 | `ShamRetriever`: ghép các câu y khoa cố định, không liên quan tới ảnh, ở đúng những câu mà lượt C2 cùng số đã có context, với số từ tối thiểu bằng context của C2 ở câu đó (RNG gieo theo `question_id`) | gốc | `--sham-reference runs/amanda-C2/full-r<k>/predictions.jsonl` | `runs/amanda-B2/full-r<k>` |

Nguồn SPOKE:

| cờ | lớp | hành vi |
|---|---|---|
| `--spoke cache` | `CachedSpokeReplay` (`scripts/c0_slake_fire.py`) | chỉ đọc `runs/amanda-A/spoke_cache/`; entity không có trong cache ⇒ danh sách rỗng |
| `--spoke live` | `LiveSpokeCache` (`src/amanda_lab/run_pipeline.py`) | đọc `runs/amanda-A/spoke_cache/`, rồi `runs/amanda-A/spoke_cache_live/`; thiếu thì gọi `SPOKEClient` của tác giả (`SPOKE_URL`, depth 1, timeout 60 s, thử 3 lần) và lưu vào `spoke_cache_live/` |

`metrics.json` của mỗi lượt ghi `spoke` (chế độ) và `spoke_stats` (số lần đọc cache, số lần gọi API, số lỗi).

### Chế độ ép Explorer — bước 11

Ba arm dùng `ForcedExplorePipeline`: ghi đè `_threshold_for_iteration` để ngưỡng ở vòng 0 là 6 (trên thang 1–5,
nên mọi câu đi tiếp ít nhất một vòng); từ vòng 1 giữ ngưỡng gốc. Retriever là `NullRetriever`. Cờ:
`--captions runs/amanda-A/captions_patched.json`.

| arm | Explorer | thư mục |
|---|---|---|
| F0 | `NoExplorer` | `runs/amanda-F0/full-r<k>` |
| F1 | Explorer gốc của tác giả | `runs/amanda-F1/full-r<k>` |
| FQ | `QuestionBankExplorer(llm, max_sub_questions=3)` (`src/amanda_lab/qbank_explorer.py`), đọc `data/qbank/qbank_v1.yaml` | `runs/amanda-FQ/full-r<k>` |

Trước khi chạy FQ, `run_pipeline.py` và bước 11 kiểm sha256 của `data/qbank/qbank_v1.yaml` với
`data/qbank/FROZEN`; khác nhau thì dừng. Bước 11 cũng chạy `scripts/qbank_dryrun.py` (LLM và MLLM giả, 20 câu).

### Đối chứng bộ giải mã — bước 12 (tuỳ chọn)

`--limit 200` (200 câu đầu của `test_en_amanda.json`), một lượt mỗi ô, cùng cờ `--spoke live --captions …`:

| | adapter vá | adapter gốc (`--raw-decode`) |
|---|---|---|
| A1 | `runs/amanda-A1/limit200-ctlfix` | `runs/amanda-A1/limit200-ctlraw` |
| A2 | `runs/amanda-A2/limit200-ctlfix` | `runs/amanda-A2/limit200-ctlraw` |

## Số lượt

| arm | số lượt | biến |
|---|---|---|
| A0, A0-bug | 1 mỗi arm | — |
| A4 | 1 (`g1`) | — |
| A1, A2, A3 (và C2, C5, B2 nếu bật) | `REPS` mỗi arm, chạy xen kẽ theo lượt (r1 của mọi arm, rồi r2, …) | `REPS`, mặc định 3 |
| F0, F1, FQ | `REPS` mỗi arm, xen kẽ theo lượt | `REPS`, mặc định 3 |
| đối chứng bộ giải mã | 1 mỗi ô | — |

Mỗi lượt ghi `predictions.jsonl` và `trace.jsonl` sau từng câu, rồi `metrics.json` khi đủ câu.
