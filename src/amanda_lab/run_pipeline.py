#!/usr/bin/env python3
"""Chạy AMANDA end-to-end trên SLAKE với các arm khác nhau. Dùng `AMANDAPipeline` GỐC của tác giả.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python -m amanda_lab.run_pipeline --arm A1 --limit 20

Arm — mỗi arm đổi **đúng một thứ** so với A1:

| arm | Retriever | ý nghĩa |
|---|---|---|
| A1 | SPOKE (`--spoke cache` / `--spoke live`) | ⭐ tái hiện AMANDA nguyên trạng — đích 🟩 54,39 / 70,43 |
| A2 | **không có** | tái hiện Table 4 (bỏ Retriever: 🟩 −1,49 / −0,96) |
| C2 | SLAKE-KG | ⭐ cải tiến đề xuất |
| B2 | **SHAM** — văn bản y khoa vô quan, cùng phân bố độ dài với A1 | cổng chống tự lừa mình |

⭐ **Chế độ ÉP (arm F\*)** — mọi câu đi qua ≥1 vòng Explorer (vòng 0 không bao giờ đạt ngưỡng), Retriever
**tắt** để cô lập Explorer. Ở cấu hình gốc Explorer chỉ chạy ở những câu Evaluator chấm dưới ngưỡng.

| arm | Explorer | ý nghĩa |
|---|---|---|
| F0 | **không có** | mốc dưới: vòng ép chạy nhưng không hỏi gì |
| F1 | gốc của tác giả | Explorer AMANDA khi được chạy trên MỌI câu |
| FQ | **question bank** (`qbank_explorer.py`) | 🟨 đề xuất: đồ thị câu hỏi đọc phim tiếng Việt → Anh |

⛔ `third_party/AMANDA/` giữ nguyên. Chỉ thay: adapter (đã vá bug giải mã) và nguồn tri thức.
⛔ Caption cache theo **96 ảnh** chứ không 1.061 câu (caption phụ thuộc ảnh, không phụ thuộc câu hỏi).
⛔ Ghi JSONL sau MỖI câu ⇒ resume được.

⚠️ **Khai báo thay thế**: LLM cho 4 vai agent là model cục bộ, không phải GPT-4o như bài báo.
   ⇒ đây là *reproduction có thay thế*, không phải *replication*.
⚠️ **fewshot tắt** ở các arm này (bài báo bật, top-k=4). Cần caption cho toàn bộ ảnh train mới bật
   được. Tắt đồng đều ở MỌI arm ⇒ so sánh giữa các arm vẫn hợp lệ; so với số bài báo thì không.
"""
from __future__ import annotations

import argparse, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "third_party" / "AMANDA" / "src"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from amanda.pipeline import AMANDAPipeline, PipelineSettings, QuestionSample   # noqa: E402
from amanda.llm.openai_chat import OpenAIChatLLM                               # noqa: E402
from amanda.retrieval.spoke.retriever import (                                 # noqa: E402
    SpokeRetriever, SpokeRetrieverConfig)
from amanda.retrieval.spoke.entity_extractor import EntityExtractor            # noqa: E402
from amanda.evaluation.evaluate_medvqa import (                                # noqa: E402
    evaluate_records, format_results_table)
from amanda.mllm.adapters.llava_med_adapter import LLaVAMedAdapter                 # noqa: E402
from amanda_lab.patched_adapter import PatchedLLaVAMedAdapter                  # noqa: E402
from amanda_lab.slake_kg import SlakeKGClient                                  # noqa: E402
from amanda_lab.qbank_explorer import QBANK, QuestionBankExplorer                # noqa: E402
from c0_slake_fire import CachedSpokeReplay                                    # noqa: E402

SLAKE = REPO / "data" / "slake"
MODEL = REPO / "models" / "llava-med-v1.5-mistral-7b"
LLAVA_REPO = REPO / "third_party" / "LLaVA-Med"
CAPTIONS = REPO / "runs" / "amanda-A" / "captions_patched.json"     # bước 06
PAPER = {"A1": {"open": 54.39, "closed": 70.43}}

# Văn bản y khoa THẬT nhưng hoàn toàn vô quan với ảnh X-quang/CT của SLAKE — dùng cho arm SHAM.
# ⭐ Mục đích: nếu A1 ≈ B2 thì phần cải thiện đến từ "prompt dài thêm", không từ "tri thức đúng".
SHAM_SENTENCES = [
    "Compound Metformin treats Disease Type 2 Diabetes Mellitus.",
    "Gene CFTR participates in Pathway Chloride Transport.",
    "Compound Warfarin interacts with Compound Amiodarone.",
    "Protein Insulin Receptor binds Compound Insulin.",
    "Disease Celiac Disease resembles Disease Tropical Sprue.",
    "Compound Levothyroxine treats Disease Hypothyroidism.",
    "Gene HFE associates Disease Hereditary Hemochromatosis.",
    "Pathway Glycolysis involves Protein Hexokinase.",
    "Compound Allopurinol treats Disease Gout.",
    "Disease Phenylketonuria isa Disease Amino Acid Metabolism Disorder.",
]


class ResilientLLM:
    """Bọc `OpenAIChatLLM`: thử lại khi server LLM rớt, thay vì làm gãy cả lần chạy.

    ⛔ Bài học 19/08: vLLM chết vì OOM VRAM ở câu 972/1061 và kéo gãy A1 cùng ba arm xếp sau.
    Có 972 câu đã ghi nên không mất việc, nhưng lần chạy thì đứt. Một lần chạy 2 giờ **không được
    phép** gãy vì một lỗi mạng nhất thời.

    Chờ tăng dần tới 5 phút để kịp cho server được khởi động lại bên ngoài.
    """
    def __init__(self, inner, max_attempts: int = 8) -> None:
        self._inner = inner
        self.max_attempts = max_attempts
        self.retries = 0
        self._lock = threading.Lock()

    def chat(self, messages, *, answer_type):
        delay = 5.0
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._inner.chat(messages, answer_type=answer_type)
            except Exception as exc:
                if attempt == self.max_attempts:
                    raise
                with self._lock:
                    self.retries += 1
                print(f"  ⚠️ LLM lỗi ({type(exc).__name__}), thử lại sau {delay:.0f}s "
                      f"[{attempt}/{self.max_attempts}]", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 300)


class NoExplorer:
    """Arm A3 — bỏ Explorer. Vòng lặp vẫn chạy nhưng không sinh câu hỏi con nào.

    ⛔ Không sửa `third_party/AMANDA/`: `AMANDAPipeline` tạo `self.explorer` trong `__init__`, ta **ghi đè
    thuộc tính đó sau khi dựng**. Bài báo (Table 4) báo bỏ Explorer mất −4,11 Open / −5,77 Closed
    trên SLAKE.
    """
    def generate(self, **_kw):
        return "", []


class NoPerceiver:
    """Arm A4 — bỏ Perceiver, tức **cắt hoàn toàn đường vào thị giác**.

    Dùng làm **đối chứng dương** của harness: bài báo (Table 4) báo hiệu ứng lớn nhất ở ablation này.
    """
    def caption(self, image, prompt=None):
        return ""

    def initial_answer(self, image, question, answer_type):
        return ""

    def followup_answer(self, image, question, answer_type):
        return ""


class NullRetriever:
    """Arm A2 — bỏ hẳn Retriever."""
    def retrieve(self, **_kw) -> str | None:
        return None


class ShamRetriever:
    """Arm B2 — trả về văn bản y khoa thật nhưng vô quan, **cùng phân bố độ dài** với A1.

    ⛔ Bắn đúng ở những câu mà arm tham chiếu đã bắn, và độ dài context khớp từng câu ⇒ khác biệt
    duy nhất so với arm tham chiếu là **nội dung có liên quan hay không**.
    """
    def __init__(self, reference: dict[str, int], seed: int = 0) -> None:
        self.reference = reference          # qid -> số từ context của arm tham chiếu (0 = không bắn)
        self.seed = seed
        self._tl = threading.local()        # ⛔ qid phải theo LUỒNG, không dùng chung

    def set_qid(self, qid: str) -> None:
        self._tl.qid = qid
        # RNG gieo theo qid ⇒ chạy bao nhiêu luồng cũng ra CÙNG một văn bản SHAM cho cùng một câu
        self._tl.rng = random.Random(f"{self.seed}:{qid}")

    def retrieve(self, **_kw) -> str | None:
        want = self.reference.get(str(getattr(self._tl, "qid", "")), 0)
        if want <= 0:
            return None
        out: list[str] = []
        n = 0
        while n < want:
            s = self._tl.rng.choice(SHAM_SENTENCES)
            out.append(s)
            n += len(s.split())
        return " ".join(out)


class EagerKGAdapter:
    """⭐ C5-EAGER — đưa tri thức đồ thị vào NGAY VÒNG 0, cho MỌI câu.

    ## Vì sao cần arm này

    AMANDA chỉ gọi Retriever ở **nhánh bổ sung tri thức**, sau khi Evaluator chấm dưới ngưỡng. Ba arm
    C2/B2/A2 đều đổi *nội dung* tri thức mà không đổi *thời điểm* nạp nó; C5 đổi thời điểm.

    ## Cách làm mà KHÔNG sửa một dòng nào của tác giả

    `AMANDAPipeline.run()` truyền `caption` vào **mọi** agent, kể cả `reasoner.initial_answer` ở
    vòng 0. Nối context của đồ thị vào caption ⇒ tri thức có mặt từ lượt suy luận đầu tiên, mà
    `third_party/AMANDA/` vẫn nguyên vẹn từng byte.

    ⚠️ Đây là **biến thể của gói này**, không phải AMANDA. Phải gọi đúng tên ở mọi bảng.
    ⛔ Context lấy từ `runs/amanda-C/SLAKE-native-loose.json` — **cùng bộ entity** mà LLM đã trích ở
       A1, tra qua **cùng code lọc** của tác giả. Không có LLM call mới, không rò rỉ nhãn.
    """
    def __init__(self, inner, cache: dict[str, str], kg_context: dict[str, str]) -> None:
        self._inner = inner
        self._cache = cache
        self._kg = kg_context
        self._gpu = threading.Lock()
        self._tl = threading.local()
        self.caption_misses = 0
        self.augmented = 0

    def set_image_key(self, key: str) -> None:
        self._tl.key = key

    def set_qid(self, qid: str) -> None:
        self._tl.qid = qid

    def caption(self, image, prompt=None):
        key = getattr(self._tl, "key", None)
        if key and key in self._cache:
            base = self._cache[key]
        else:
            with self._gpu:
                self.caption_misses += 1
                base = self._inner.caption(image, prompt) if prompt else self._inner.caption(image)
        ctx = self._kg.get(str(getattr(self._tl, "qid", "")))
        if not ctx:
            return base
        self.augmented += 1
        return f"{base}\n\nGrounded Medical Knowledge: {ctx}"

    def answer(self, image, question, answer_type):
        with self._gpu:
            return self._inner.answer(image, question, answer_type)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class CaptionCachedAdapter:
    """Bọc adapter LLaVA-Med: cache caption theo TÊN ẢNH + an toàn khi chạy nhiều luồng.

    Hai điều bắt buộc khi song song hoá:
    * **`image_key` phải theo LUỒNG.** Dùng chung một thuộc tính là luồng này ghi đè luồng kia
      ⇒ lấy nhầm caption của ảnh khác. Lỗi này sẽ KHÔNG làm chương trình chết, chỉ làm sai số liệu.
    * **Mọi lượt gọi GPU phải nằm trong khoá.** Chỉ có một instance LLaVA-Med trên GPU1 và
      `generate()` của HuggingFace không an toàn đa luồng.

    ⇒ Phần LLaVA-Med **tuần tự** (nó là nút cổ chai GPU), phần gọi vLLM **chạy song song**.
    """
    def __init__(self, inner, cache: dict[str, str]) -> None:
        self._inner = inner
        self._cache = cache
        self._gpu = threading.Lock()
        self._tl = threading.local()
        self.caption_misses = 0

    def set_image_key(self, key: str) -> None:
        self._tl.key = key

    def caption(self, image, prompt=None):
        key = getattr(self._tl, "key", None)
        if key and key in self._cache:
            return self._cache[key]           # cache hit ⇒ không đụng GPU, không cần khoá
        with self._gpu:
            self.caption_misses += 1
            return self._inner.caption(image, prompt) if prompt else self._inner.caption(image)

    def answer(self, image, question, answer_type):
        with self._gpu:
            return self._inner.answer(image, question, answer_type)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class ForcedExplorePipeline(AMANDAPipeline):
    """Chế độ ÉP (arm F*): ngưỡng vòng 0 = 6 (thang 1–5 ⇒ không bao giờ đạt) ⇒ MỌI câu qua ≥1 vòng Explorer.
    Từ vòng 1 giữ ngưỡng gốc. ⛔ Ghi đè đúng một phương thức (`medvqa_pipeline.py:194`), third_party/AMANDA/ nguyên vẹn."""
    def _threshold_for_iteration(self, iteration: int) -> int:
        return 6 if iteration == 0 else super()._threshold_for_iteration(iteration)


FORCED = ("F0", "F1", "FQ")


class LiveSpokeCache:
    """`--spoke live`: SPOKE API thật + cache đĩa. Đọc cache của bước 07 (runs/amanda-A/spoke_cache) trước, rồi cache
    sống (runs/amanda-A/spoke_cache_live); thiếu thì gọi `SPOKEClient` của tác giả. Lỗi mạng KHÔNG được cache và được
    đếm trong `stats["errors"]` (ghi vào metrics.json → spoke_stats).

    `--spoke cache` (`CachedSpokeReplay`) thì chỉ đọc cache của bước 07 và coi entity không có trong cache là rỗng.
    """
    OLD = REPO / "runs/amanda-A/spoke_cache"
    LIVE = REPO / "runs/amanda-A/spoke_cache_live"

    def __init__(self) -> None:
        from amanda.retrieval.spoke.spoke_client import SPOKEClient, SpokeAPIConfig
        self.LIVE.mkdir(parents=True, exist_ok=True)
        self._client = SPOKEClient(SpokeAPIConfig(base_uri=os.environ.get("SPOKE_URL", "https://spoke.rbvi.ucsf.edu"), timeout_seconds=60, depth=1))
        self._lock = threading.Lock()
        self.stats = {"hit_old": 0, "hit_live": 0, "fetched": 0, "errors": 0}

    def get_neighborhood(self, node_value: str):
        key = "".join(c if c.isalnum() else "_" for c in node_value)[:120] or "_empty"
        for tag, d in (("hit_old", self.OLD), ("hit_live", self.LIVE)):
            f = d / f"{key}.json"
            if f.is_file():
                with self._lock: self.stats[tag] += 1
                return json.loads(f.read_text())
        for attempt in range(3):
            try:
                out = self._client.get_neighborhood(node_value)
                (self.LIVE / f"{key}.json").write_text(json.dumps(out))
                with self._lock: self.stats["fetched"] += 1
                return out
            except Exception:
                time.sleep(2 * (attempt + 1))
        with self._lock: self.stats["errors"] += 1
        return []


def build_retriever(arm: str, llm, reference: dict[str, int] | None, spoke: str = "cache"):
    if arm in FORCED:              # cô lập Explorer; A2 đã cho thấy Retriever không có tác dụng đo được
        return NullRetriever(), None
    if arm in ("A2", "C5"):        # C5 nạp tri thức qua caption ⇒ Retriever của vòng lặp để trống
                                   # ⇒ đúng MỘT thay đổi so với A2: thời điểm nạp tri thức.
        return NullRetriever(), None
    if arm == "B2":
        return ShamRetriever(reference or {}), None
    extractor = EntityExtractor(llm, max_entities=3)
    if arm == "C2":
        client = SlakeKGClient(loose_match=True)
        cfg = SpokeRetrieverConfig(allowed_predicates=tuple(client.native_predicates))
    else:                                    # A1 — SPOKE nguyên trạng
        client, cfg = (LiveSpokeCache() if spoke == "live" else CachedSpokeReplay()), SpokeRetrieverConfig()
    return SpokeRetriever(entity_extractor=extractor, client=client, config=cfg), client


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["A1", "A2", "B2", "C2", "C5", "A3", "A4", *FORCED])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=1,
                    help="số câu chạy song song. LLaVA-Med vẫn tuần tự (khoá GPU); "
                         "song song chỉ giúp ở phần gọi vLLM. ⛔ Đổi số này KHÔNG được đổi kết quả.")
    ap.add_argument("--conv-mode", default="mistral_instruct")
    ap.add_argument("--llm-base-url", default=os.environ.get("AMANDA_LLM_URL", "http://127.0.0.1:4001/v1"))
    ap.add_argument("--llm-model", default=os.environ.get("AMANDA_LLM_MODEL", "gemma-4-26b"))
    ap.add_argument("--sham-reference", default="", help="B2: file predictions.jsonl của arm tham chiếu")
    ap.add_argument("--raw-decode", action="store_true",
                    help="dùng _decode_output GỐC (có bug) — để kiểm kết luận có phụ thuộc bản vá không")
    ap.add_argument("--captions", default=str(CAPTIONS),
                    help="file caption (ảnh → caption) do bước 06 sinh; mặc định runs/amanda-A/captions_patched.json")
    ap.add_argument("--spoke", choices=["cache", "live"], default="cache",
                    help="cache = chỉ đọc cache SPOKE của bước 07 (entity thiếu ⇒ rỗng); live = gọi SPOKE API cho entity thiếu")
    ap.add_argument("--tag", default="", help="hậu tố thư mục đầu ra, dùng để chạy LẶP LẠI cùng một arm")
    ap.add_argument("--expect", default="", help="kỳ vọng đăng ký TRƯỚC khi chạy")
    ap.add_argument("--wandb", action="store_true",
                    help="ghi lượt chạy lên Weights & Biases (WANDB_PROJECT, WANDB_ENTITY): điểm lũy kế mỗi 20 câu")
    ap.add_argument("--wandb-group", default="", help="nhóm W&B; mặc định suy từ arm (main / ablation / qbank)")
    a = ap.parse_args()

    if a.arm == "FQ":                          # ⛔ bank phải đóng băng TRƯỚC khi chạy
        frozen = QBANK.parent / "FROZEN"
        import hashlib
        h = hashlib.sha256(QBANK.read_bytes()).hexdigest()
        if not frozen.is_file() or frozen.read_text().split()[0] != h:
            sys.exit("⛔ arm FQ: question bank chưa đóng băng hoặc đã sửa sau khi đóng băng — "
                     "chạy scripts/qbank_graph.py --freeze (và đổi tên thành v2 nếu đã có lần chạy)")

    gt = json.loads((SLAKE / "test_en_amanda.json").read_text(encoding="utf8"))
    if a.limit:
        gt = gt[: a.limit]
    img_root = Path((SLAKE / "image_root.txt").read_text().strip())
    caps = json.loads(Path(a.captions).read_text(encoding="utf8"))

    sub = f"limit{a.limit}" if a.limit else "full"
    out_dir = REPO / "runs" / f"amanda-{a.arm}" / (f"{sub}-{a.tag}" if a.tag else sub)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    trace_path = out_dir / "trace.jsonl"
    if a.expect:
        (out_dir / "EXPECTATION.txt").write_text(a.expect, encoding="utf8")

    done = set()
    if pred_path.is_file():
        for line in pred_path.read_text(encoding="utf8").splitlines():
            if line.strip():
                done.add(str(json.loads(line)["question_id"]))
    todo = [g for g in gt if str(g["question_id"]) not in done]

    wb = running = None
    if a.wandb:
        from amanda_lab import wandb_log
        group = a.wandb_group or {"A1": "main", "A2": "ablation", "A3": "ablation", "A4": "ablation",
                                  **{f: "qbank" for f in FORCED}}.get(a.arm, "intervention")
        wb = wandb_log.start(f"{a.arm}-{a.tag or sub}", group, {
            **vars(a), "n_questions": len(gt), "max_iterations": 3, "confidence_threshold": 3,
            "max_sub_questions": 3, "fewshot": False, "temperature": 0.0, "mllm": MODEL.name,
            "git_sha": wandb_log.git_sha(REPO), "test_sha256": wandb_log.sha256(SLAKE / "test_en_amanda.json"),
            "captions_sha256": wandb_log.sha256(Path(a.captions))},
            tags=["live", a.arm], notes=a.expect)
        running = wandb_log.Running(gt)
        for line in (pred_path.read_text(encoding="utf8").splitlines() if pred_path.is_file() else []):
            if line.strip():
                running.add(json.loads(line))
    print(f"arm={a.arm}  n={len(gt)}  đã có {len(done)}  cần chạy {len(todo)}", flush=True)

    reference = None
    if a.arm == "B2":
        if not a.sham_reference:
            sys.exit("⛔ arm B2 cần --sham-reference trỏ tới predictions.jsonl của arm tham chiếu")
        reference = {}
        for line in Path(a.sham_reference).read_text(encoding="utf8").splitlines():
            if line.strip():
                r = json.loads(line)
                reference[str(r["question_id"])] = len((r.get("rag_context") or "").split())
        print(f"  SHAM tham chiếu: {sum(1 for v in reference.values() if v)} câu có context", flush=True)

    if todo:
        import torch
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        llm = ResilientLLM(OpenAIChatLLM(api_key="EMPTY", model=a.llm_model,
                                         base_url=a.llm_base_url, temperature=0.0, max_tokens=1024))
        t0 = time.time()
        Cls = LLaVAMedAdapter if a.raw_decode else PatchedLLaVAMedAdapter
        if a.raw_decode:
            print("  ⚠️ DÙNG BỘ GIẢI MÃ GỐC (CÓ BUG) — chỉ dùng cho đối chứng", flush=True)
        inner = Cls(model_path=MODEL, device=a.device,
                                       llava_repo_path=LLAVA_REPO, conv_mode=a.conv_mode,
                                       temperature=0.0)
        if a.arm == "C5":
            kg = {str(r["qid"]): r["context"]
                  for r in json.loads((REPO / "runs/amanda-C/SLAKE-native-loose.json")
                                      .read_text(encoding="utf8"))["results"] if r["context"]}
            print(f"  C5-EAGER: {len(kg)}/1061 câu có context SLAKE-KG nạp từ VÒNG 0", flush=True)
            mllm = EagerKGAdapter(inner, caps, kg)
        else:
            mllm = CaptionCachedAdapter(inner, caps)
        print(f"  nạp LLaVA-Med {time.time()-t0:.0f}s · VRAM {torch.cuda.memory_allocated()/2**30:.1f} GB",
              flush=True)

        retriever, spoke_client = build_retriever(a.arm, llm, reference, a.spoke)
        Pipe = ForcedExplorePipeline if a.arm in FORCED else AMANDAPipeline
        pipe = Pipe(mllm=mllm, llm=llm, retriever=retriever, dataset_name="slake",
                              settings=PipelineSettings(max_iterations=3, confidence_threshold=3,
                                                        max_sub_questions=3, fewshot_enabled=False))
        # ⛔ Bỏ agent bằng cách GHI ĐÈ thuộc tính sau khi dựng — third_party/AMANDA/ vẫn nguyên vẹn.
        if a.arm == "A3":
            pipe.explorer = NoExplorer()
            print("  ⚠️ ARM A3 — đã BỎ Explorer", flush=True)
        elif a.arm == "A4":
            pipe.perceiver = NoPerceiver()
            print("  ⚠️ ARM A4 — đã BỎ Perceiver (đối chứng DƯƠNG)", flush=True)
        elif a.arm == "F0":
            pipe.explorer = NoExplorer()
        elif a.arm == "FQ":
            pipe.explorer = QuestionBankExplorer(llm, max_sub_questions=3)
        if a.arm in FORCED:
            print(f"  ⭐ CHẾ ĐỘ ÉP — mọi câu qua ≥1 vòng Explorer · Retriever TẮT · explorer="
                  f"{type(pipe.explorer).__name__}", flush=True)

        t1 = time.time()
        wlock = threading.Lock()
        done_n = [0]

        def one(g):
            qid = str(g["question_id"])
            question = g["conversations"][0]["value"].replace("<image>\n", "")
            mllm.set_image_key(g["image"])
            if isinstance(mllm, EagerKGAdapter):
                mllm.set_qid(qid)
            if isinstance(retriever, ShamRetriever):
                retriever.set_qid(qid)
            sample = QuestionSample(question_id=qid, question=question,
                                    prompt=g["conversations"][0]["value"],
                                    image=g["image"], answer=g["conversations"][1]["value"],
                                    answer_type=g["answer_type"])
            res = pipe.run(sample, image=inner.load_image(img_root / g["image"]))
            pred = {"question_id": g["question_id"], "prompt": g["conversations"][0]["value"],
                    "text": res.final_answer, "answer_type": g["answer_type"],
                    "content_type": g.get("content_type"), "rag_context": res.rag_context,
                    "iterations": res.iteration}
            trace = {"question_id": g["question_id"], "caption": res.caption,
                     "initial_answer": res.initial_answer, "final_answer": res.final_answer,
                     "final_analysis": res.final_analysis, "rag_context": res.rag_context,
                     "iterations": res.iteration, "history": res.history}
            with wlock:                       # ⛔ ghi sau MỖI câu ⇒ resume được kể cả khi đứt giữa chừng
                fp.write(json.dumps(pred, ensure_ascii=False) + "\n"); fp.flush()
                ft.write(json.dumps(trace, ensure_ascii=False) + "\n"); ft.flush()
                done_n[0] += 1
                k = done_n[0]
                if running is not None:
                    running.add(pred)
                if k % 20 == 0 or k == len(todo):
                    el = time.time() - t1
                    if wb is not None:
                        wandb_log.log_progress(wb, running, {"speed/sec_per_question": el / k})
                    print(f"  {k}/{len(todo)}  {el/k:.2f}s/câu  còn ~{(len(todo)-k)*el/k/60:.0f} phút",
                          flush=True)

        with pred_path.open("a", encoding="utf8") as fp, trace_path.open("a", encoding="utf8") as ft:
            if a.workers <= 1:
                for g in todo:
                    one(g)
            else:
                with ThreadPoolExecutor(max_workers=a.workers) as ex:
                    list(ex.map(one, todo))

    preds = [json.loads(l) for l in pred_path.read_text(encoding="utf8").splitlines() if l.strip()]
    summary = evaluate_records(ground_truth=gt, predictions=preds)
    fired = sum(1 for p in preds if p.get("rag_context"))
    iters = [p.get("iterations", 0) for p in preds]
    print(f"\n{'='*60}\narm {a.arm} — n={len(preds)}")
    print(format_results_table(summary.metrics))
    print(f"  Retriever bắn : {fired}/{len(preds)} = {100*fired/len(preds):.1f}%")
    print(f"  vòng lặp TB   : {sum(iters)/len(iters):.2f}")
    tgt = PAPER.get(a.arm)
    if tgt:
        print(f"  🟩 đích bài báo (Table 1): Open {tgt['open']} · Closed {tgt['closed']}")
        print("  ⚠️ LLM thay thế + fewshot tắt ⇒ KHÔNG so trực tiếp với bài báo")
    (out_dir / "metrics.json").write_text(json.dumps(
        {"arm": a.arm, "n": len(preds), "metrics": summary.metrics, "paper_target": tgt,
         "retriever_fire": fired, "mean_iterations": round(sum(iters)/len(iters), 2),
         "llm_model": a.llm_model, "fewshot": False, "conv_mode": a.conv_mode,
         "workers": a.workers, "raw_decode": a.raw_decode, "captions": Path(a.captions).name,
         "forced_explore": a.arm in FORCED, "spoke": a.spoke,
         "spoke_stats": (getattr(spoke_client, "stats", None) or ({"cache_misses": spoke_client.misses} if hasattr(spoke_client, "misses") else None))
                        if todo and a.arm not in FORCED else None,
         "qbank_frozen": ((QBANK.parent / "FROZEN").read_text().split()[0]
                          if a.arm == "FQ" and (QBANK.parent / "FROZEN").is_file() else None),
         "qbank_stats": (pipe.explorer.stats if todo and a.arm == "FQ" else None)},
        indent=1, ensure_ascii=False), encoding="utf8")
    if wb is not None:
        wandb_log.finish(wb, out_dir, gt)


if __name__ == "__main__":
    main()
