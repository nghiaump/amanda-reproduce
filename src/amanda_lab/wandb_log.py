"""Ghi một lượt chạy AMANDA lên Weights & Biases.

Dùng chung cho hai đường:
- chạy trực tiếp: ``run_pipeline.py --wandb`` log điểm lũy kế mỗi 20 câu trong lúc chạy;
- upload sau: ``scripts/wandb_upload.py`` dựng lại đúng các đại lượng đó từ ``predictions.jsonl`` đã lưu.

Điểm luôn chấm bằng hàm của tác giả (``evaluate_records``, ``_closed_answer_is_correct``), giống ``metrics.json``.
Project và entity lấy từ biến môi trường ``WANDB_PROJECT`` và ``WANDB_ENTITY``; chế độ (online / offline / disabled)
do chính thư viện wandb đọc từ ``WANDB_MODE``.
"""
from __future__ import annotations

import hashlib, json, os, re, subprocess
from pathlib import Path
from typing import Any, Iterable

from amanda.evaluation.evaluate_medvqa import evaluate_records, _closed_answer_is_correct
from amanda.evaluation.metrics import calculate_f1score

PROJECT = os.environ.get("WANDB_PROJECT", "amanda-slake-reproduction")
ENTITY = os.environ.get("WANDB_ENTITY") or None
STEP = "questions_done"
MARKER = "WANDB_RUN.txt"            # ghi vào thư mục lượt chạy sau khi đã log xong


def git_sha(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                   # gói tái lập có thể không phải git repo
        return None


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_id(name: str) -> str:
    """id cố định theo tên ⇒ chạy lại hay upload lại thì ghi vào đúng run cũ, không sinh bản trùng."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", name)[:64]


def question_score(gt_item: dict, pred: dict) -> float:
    gold = gt_item["conversations"][1]["value"].lower()
    text = (pred.get("text") or "").lower()
    if str(gt_item.get("answer_type", "open")).lower() == "open":
        return float(calculate_f1score(text, gold)[2])
    return float(_closed_answer_is_correct(text, gold))


class Running:
    """Điểm lũy kế trên các câu đã chạy xong (cùng hàm chấm với metrics.json)."""

    def __init__(self, gt: list[dict]):
        self.gt = {str(g["question_id"]): g for g in gt}
        self.preds: list[dict] = []

    def add(self, pred: dict) -> None:
        self.preds.append(pred)

    def snapshot(self) -> dict[str, float]:
        sub = [self.gt[str(p["question_id"])] for p in self.preds if str(p["question_id"]) in self.gt]
        m = evaluate_records(ground_truth=sub, predictions=self.preds).metrics
        n = max(1, len(self.preds))
        return {"running/open_recall": m["Open Recall"], "running/closed_accuracy": m["Closed Accuracy"],
                "running/knowledge_in_prompt_pct": 100 * sum(1 for p in self.preds if p.get("rag_context")) / n,
                "running/mean_iterations": sum(p.get("iterations") or 0 for p in self.preds) / n}


def start(name: str, group: str, config: dict[str, Any], *, job_type: str = "run",
          tags: Iterable[str] = (), notes: str = ""):
    import wandb
    run = wandb.init(project=PROJECT, entity=ENTITY, name=name, id=run_id(name), resume="allow",
                     group=group, job_type=job_type, tags=list(tags), notes=notes, config=config)
    run.define_metric(STEP)
    run.define_metric("running/*", step_metric=STEP)
    run.define_metric("speed/*", step_metric=STEP)
    return run


def log_progress(run, running: Running, extra: dict[str, float] | None = None) -> None:
    run.log({STEP: len(running.preds), **running.snapshot(), **(extra or {})})


def replay(run, gt: list[dict], preds: list[dict], every: int = 20) -> None:
    """Upload sau: dựng lại đường cong lũy kế theo đúng thứ tự câu trong predictions.jsonl."""
    r = Running(gt)
    for i, p in enumerate(preds, 1):
        r.add(p)
        if i % every == 0 or i == len(preds):
            log_progress(run, r)


def finish(run, out_dir: Path, gt: list[dict], extra_files: Iterable[Path] = ()) -> None:
    """Summary = metrics.json; bảng dự đoán từng câu; artifact chứa mọi file đầu ra của lượt chạy."""
    import wandb
    out_dir = Path(out_dir)
    preds = [json.loads(l) for l in (out_dir / "predictions.jsonl").read_text(encoding="utf8").splitlines() if l.strip()]
    g = {str(x["question_id"]): x for x in gt}
    met = json.loads((out_dir / "metrics.json").read_text(encoding="utf8"))
    run.summary.update({"open_recall": met["metrics"]["Open Recall"],
                        "closed_accuracy": met["metrics"]["Closed Accuracy"], "n": met.get("n", len(preds)),
                        "knowledge_in_prompt": met.get("retriever_fire"),
                        "mean_iterations": met.get("mean_iterations"),
                        **({"spoke_stats": met["spoke_stats"]} if met.get("spoke_stats") else {})})
    cols = ["question_id", "answer_type", "content_type", "question", "ground_truth", "prediction", "score",
            "iterations", "knowledge_in_prompt"]
    rows = []
    for p in preds:
        gi = g.get(str(p["question_id"]))
        if gi is None:
            continue
        rows.append([p["question_id"], gi["answer_type"], gi.get("content_type"),
                     gi["conversations"][0]["value"].replace("<image>\n", ""), gi["conversations"][1]["value"],
                     p.get("text"), question_score(gi, p), p.get("iterations"), bool(p.get("rag_context"))])
    run.log({"predictions": wandb.Table(columns=cols, data=rows)})
    art = wandb.Artifact(run_id(run.name), type="run-output",
                         description=f"Đầu ra của lượt {run.name}: predictions, trace, metrics, kỳ vọng đăng ký trước")
    for f in ["predictions.jsonl", "trace.jsonl", "metrics.json", "EXPECTATION.txt", *map(str, extra_files)]:
        fp = Path(f) if Path(f).is_absolute() else out_dir / f
        if fp.is_file():
            art.add_file(str(fp), name=fp.name)
    run.log_artifact(art)
    # Đánh dấu lượt đã có trên W&B ⇒ scripts/wandb_upload.py (bước 15) không đưa lên lần nữa.
    (out_dir / MARKER).write_text(f"{run.id}\t{os.environ.get('WANDB_MODE', 'online')}\t{getattr(run, 'url', '') or ''}\n",
                                  encoding="utf8")
    run.finish()
