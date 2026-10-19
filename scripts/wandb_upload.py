#!/usr/bin/env python3
"""Đưa các lượt chạy ĐÃ XONG trong runs/ lên Weights & Biases từ file đã lưu (bước 15).

    PYTHONPATH=src .venv/bin/python scripts/wandb_upload.py                 # mọi lượt đã xong + phân tích
    PYTHONPATH=src .venv/bin/python scripts/wandb_upload.py --only A1-r1 A4-g1 analysis
    PYTHONPATH=src .venv/bin/python scripts/wandb_upload.py --list          # chỉ liệt kê, không upload
    PYTHONPATH=src .venv/bin/python scripts/wandb_upload.py --force         # upload cả lượt đã có WANDB_RUN.txt

Biến môi trường: WANDB_PROJECT (mặc định amanda-slake-reproduction), WANDB_ENTITY (trống = tài khoản mặc định),
WANDB_MODE (online / offline / disabled; offline ghi vào ./wandb để `wandb sync` sau).

Mỗi lượt thành một run W&B, cùng schema với ``run_pipeline.py --wandb``:
- đường cong điểm lũy kế dựng lại theo thứ tự câu trong ``predictions.jsonl``;
- bảng dự đoán từng câu;
- artifact gồm predictions, trace, metrics, EXPECTATION.txt (nếu có) và log của lượt.
Tên run = ``<arm>-<tag>`` (vd. ``A1-r2``), giống tên mà ``run_pipeline.py --wandb`` dùng. Lượt nào đã được log trực
tiếp (có file ``WANDB_RUN.txt`` trong thư mục lượt) thì bỏ qua, trừ khi có ``--force``.
Lệnh ghi trong config của run được dựng lại từ metrics.json, theo đúng dạng lệnh mà các bước 08–12 gọi.
"""
from __future__ import annotations

import argparse, datetime as dt, glob, json, os, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "third_party/AMANDA/src", REPO / "src", REPO / "scripts"):
    sys.path.insert(0, str(p))

from amanda_lab import wandb_log  # noqa: E402

R = REPO / "runs"
LOGS = REPO / "logs" / "runs"
N_FULL = 1061
FORCED = ("F0", "F1", "FQ")
GROUP = {"A0": "baseline", "A1": "main", "A2": "ablation", "A3": "ablation", "A4": "ablation",
         "C2": "graph", "B2": "graph", "C5": "graph",
         **{f: "qbank" for f in FORCED}}
DESC = {
    "A0": "LLaVA-Med trả lời một bước, không agent nào",
    "A1": "AMANDA đầy đủ, Retriever = SPOKE",
    "A2": "AMANDA bỏ Retriever",
    "A3": "AMANDA bỏ Explorer",
    "A4": "AMANDA bỏ Perceiver",
    "C2": "Retriever = knowledge graph của SLAKE",
    "B2": "Retriever SHAM: văn bản y khoa vô quan, cùng độ dài, bắn đúng các câu C2 cùng lượt đã bắn",
    "C5": "context SLAKE-KG nối vào caption từ vòng 0, Retriever trong vòng lặp tắt",
    "F0": "chế độ ép Explorer, Explorer rỗng",
    "F1": "chế độ ép Explorer, Explorer gốc",
    "FQ": "chế độ ép Explorer, question bank",
}


def complete(d: Path) -> bool:
    p = d / "predictions.jsonl"
    if not ((d / "metrics.json").is_file() and p.is_file()):
        return False
    need = int(d.name[5:].split("-")[0]) if d.name.startswith("limit") else N_FULL
    return sum(1 for l in p.open(encoding="utf8") if l.strip()) >= need


def split_dir(d: Path) -> tuple[str, str, int]:
    """runs/amanda-<arm>/<full|limitN>[-<tag>] → (arm, tag, limit)."""
    arm = d.parent.name.removeprefix("amanda-")
    sub, _, tag = d.name.partition("-")
    limit = int(sub[5:]) if sub.startswith("limit") else 0
    return arm, tag, limit


def command(d: Path, met: dict) -> str:
    """Dựng lại lệnh đã chạy, đúng dạng mà steps/*.sh (qua lib/common.sh) gọi."""
    arm, tag, limit = split_dir(d)
    if arm == "A0":
        return "python scripts/amanda_run.py --arm A0" + (f" --tag {tag}" if tag else "") + \
               (" --raw-decode" if met.get("raw_decode") else "")
    parts = ["python -m amanda_lab.run_pipeline", f"--arm {arm}"]
    if tag:
        parts.append(f"--tag {tag}")
    if limit:
        parts.append(f"--limit {limit}")
    parts += [f"--workers {met.get('workers')}", "--llm-base-url $AMANDA_LLM_URL",
              f"--llm-model {met.get('llm_model')}"]
    if met.get("captions"):
        parts.append(f"--captions runs/amanda-A/{met['captions']}")
    if arm not in FORCED and met.get("spoke"):
        parts.append(f"--spoke {met['spoke']}")
    if arm == "B2":
        parts.append(f"--sham-reference runs/amanda-C2/full-{tag}/predictions.jsonl")
    if met.get("raw_decode"):
        parts.append("--raw-decode")
    return " ".join(parts)


def run_log(arm: str, tag: str) -> Path | None:
    """Log mới nhất của lượt: logs/runs/<ngày-giờ>_<arm>-<tag>.log (lib/common.sh, steps/08_baseline.sh)."""
    hits = sorted(glob.glob(str(LOGS / f"*_{arm}-{tag or 'notag'}.log")))
    return Path(hits[-1]) if hits else None


def discover() -> dict[str, Path]:
    out = {}
    for d in sorted(Path(x) for x in glob.glob(str(R / "amanda-*" / "*"))):
        if d.is_dir() and d.parent.name != "amanda-A" and d.parent.name != "amanda-C" and complete(d):
            arm, tag, limit = split_dir(d)
            name = f"{arm}-{tag or ('limit%d' % limit if limit else 'full')}"
            out[name] = d
    return out


def when(d: Path) -> str:
    return dt.datetime.fromtimestamp((d / "predictions.jsonl").stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def speed(log: Path | None) -> float | None:
    if not log or not log.is_file():
        return None
    m = re.findall(r"(\d+(?:\.\d+)?)s/câu", log.read_text(encoding="utf8", errors="replace"))
    return float(m[-1]) if m else None


def upload_run(name: str, d: Path, gt: list[dict]) -> None:
    arm, tag, limit = split_dir(d)
    met = json.loads((d / "metrics.json").read_text(encoding="utf8"))
    preds = [json.loads(l) for l in (d / "predictions.jsonl").read_text(encoding="utf8").splitlines() if l.strip()]
    sub_gt = gt[:limit] if limit else gt
    logp = run_log(arm, tag)
    desc = DESC.get(arm, arm) + (" (bộ giải mã gốc của adapter)" if met.get("raw_decode") else "")
    cfg = {k: v for k, v in met.items() if k not in ("metrics", "spoke_stats", "qbank_stats")}
    cfg.update({"command": command(d, met), "run_dir": str(d.relative_to(REPO)), "finished_at": when(d),
                "sec_per_question": speed(logp), "max_iterations": 3, "confidence_threshold": 3,
                "max_sub_questions": 3, "temperature": 0.0,
                "test_sha256": wandb_log.sha256(REPO / "data/slake/test_en_amanda.json"), "description": desc})
    notes = (f"{desc}. Chạy xong lúc {when(d)}; đưa lên W&B ngày {dt.date.today():%Y-%m-%d} từ file đã lưu "
             f"(đường cong lũy kế dựng lại theo thứ tự câu).")
    run = wandb_log.start(name, GROUP.get(arm, "other"), cfg, tags=["post-hoc-upload", arm], notes=notes)
    wandb_log.replay(run, sub_gt, preds)
    wandb_log.finish(run, d, sub_gt, extra_files=[logp] if logp else [])
    print(f"✓ {name} ← {d.relative_to(REPO)}")


def flat(v):
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v


def upload_analysis() -> None:
    """Các file phân tích của bước 13 (nếu đã có): results/retriever_investigation.json, results/runs_summary.json."""
    import wandb
    inv_p, sum_p = REPO / "results/retriever_investigation.json", REPO / "results/runs_summary.json"

    def table(rows):
        cols = list(rows[0].keys())
        return wandb.Table(columns=cols, data=[[flat(r.get(c)) for c in cols] for r in rows])

    def art(run, name, files):
        a = wandb.Artifact(name, type="analysis")
        for f in files:
            a.add_file(str(f), name=Path(f).name)
        run.log_artifact(a)

    if inv_p.is_file():
        inv = json.loads(inv_p.read_text(encoding="utf8"))
        run = wandb_log.start("retriever-investigation", "analysis", {
            "source_runs": inv["summary"]["runs"], "script": "scripts/retriever_investigation.py"},
            job_type="analysis", tags=["post-hoc-upload", "retriever"],
            notes="Phát lại mọi lần gọi Retriever của các lượt A1: entity trích ra, SPOKE trả gì (cache và API thật), "
                  "bộ lọc cosine giữ lại gì.")
        run.summary.update({k: v for k, v in inv["summary"].items() if isinstance(v, (int, float))})
        ok = [r for r in inv["rows"] if "error" not in r]
        log = {"top_entities": wandb.Table(columns=["entity", "count"], data=inv["summary"]["top_entities"]),
               "top_missing_entities": wandb.Table(columns=["entity", "count"], data=inv["summary"]["top_missing"])}
        if ok:
            log["calls"] = table(ok)
        run.log(log)
        art(run, "retriever-investigation", [inv_p, REPO / "scripts/retriever_investigation.py"])
        run.finish()
        print("✓ analysis: retriever-investigation")
    else:
        print(f"· bỏ qua retriever-investigation (chưa có {inv_p.relative_to(REPO)})")

    if sum_p.is_file():
        S = json.loads(sum_p.read_text(encoding="utf8"))
        pl = S["pipeline"]
        run = wandb_log.start("knowledge-funnel", "analysis",
                              {"source_runs": S["runs"], "script": "scripts/summarize_runs.py"},
                              job_type="analysis", tags=["post-hoc-upload", "retriever"],
                              notes="Bao nhiêu câu dừng ở vòng 0, bao nhiêu đi tiếp, bao nhiêu có tri thức trong "
                                    "prompt; với mỗi câu có tri thức: đáp án đầu và cuối đúng không, Reasoner có dẫn không.")
        run.summary.update({**{f"pipeline/{k}": v for k, v in pl.items() if isinstance(v, (int, float))},
                            **{f"fired/{k}": v for k, v in S["fired"].items() if isinstance(v, (int, float))},
                            "A1/open": S["A1"]["open"], "A1/closed": S["A1"]["closed"],
                            "A1/sd_open": S["sd"]["open"], "A1/sd_closed": S["sd"]["closed"],
                            "A0/open": S["A0"]["open"], "A0/closed": S["A0"]["closed"],
                            "A4/open": S["A4"]["open"], "A4/closed": S["A4"]["closed"]})
        log = {"per_run": table(pl["per_run"])}
        if S["fired"]["rows"]:
            log["knowledge_rows"] = table(S["fired"]["rows"])
        run.log(log)
        art(run, "runs-summary", [sum_p, REPO / "scripts/summarize_runs.py"])
        run.finish()

        v = S["verbosity"]
        run = wandb_log.start("open-answer-length", "analysis",
                              {"modes": v["modes"], "script": "scripts/summarize_runs.py"},
                              job_type="analysis", tags=["post-hoc-upload", "open-recall"],
                              notes="Open recall của A0 và A1 khi giữ 5/10/20 từ đầu, câu đầu tiên, hoặc toàn bộ câu trả lời.")
        rows = [[m, v["A0"][i], v["A1_mean"][i], *[r[i] for r in v["A1_runs"]]] for i, m in enumerate(v["modes"])]
        run.log({"recall_by_length": wandb.Table(
            columns=["mode", "A0", "A1_mean", *[f"A1_r{i+1}" for i in range(len(v["A1_runs"]))]], data=rows)})
        run.finish()
        print("✓ analysis: knowledge-funnel, open-answer-length")
    else:
        print(f"· bỏ qua knowledge-funnel, open-answer-length (chưa có {sum_p.relative_to(REPO)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="chỉ upload các run này (tên <arm>-<tag>, hoặc 'analysis')")
    ap.add_argument("--list", action="store_true", help="chỉ liệt kê các lượt tìm thấy, không upload")
    ap.add_argument("--force", action="store_true", help="upload cả lượt đã có WANDB_RUN.txt")
    a = ap.parse_args()

    if os.environ.get("WANDB_MODE", "").lower() == "disabled" and not a.list:
        print("WANDB_MODE=disabled — không upload gì")
        return
    runs = discover()
    names = a.only or [*runs, "analysis"]
    if a.list:
        for n, d in runs.items():
            mark = "đã có trên W&B" if (d / wandb_log.MARKER).is_file() else "chưa"
            print(f"  {n:<22} {str(d.relative_to(REPO)):<40} {mark}")
        return
    gt = json.loads((REPO / "data/slake/test_en_amanda.json").read_text(encoding="utf8"))
    for n in names:
        if n == "analysis":
            upload_analysis()
            continue
        if n not in runs:
            print(f"✗ không có lượt nào đã xong tên {n} (xem --list)")
            continue
        if (runs[n] / wandb_log.MARKER).is_file() and not a.force:
            print(f"· {n} đã có trên W&B ({wandb_log.MARKER}) — bỏ qua")
            continue
        upload_run(n, runs[n], gt)
    ent = wandb_log.ENTITY or "<tài khoản>"
    print(f"project: https://wandb.ai/{ent}/{wandb_log.PROJECT}")


if __name__ == "__main__":
    main()
