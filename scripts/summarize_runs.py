#!/usr/bin/env python3
"""Tổng hợp các lượt A1 của bước 10 (cộng A0 của bước 08 và A4 của bước 09) thành một file JSON.

    PYTHONPATH=src .venv/bin/python scripts/summarize_runs.py
    A1_GLOB='full-r*' PYTHONPATH=src .venv/bin/python scripts/summarize_runs.py     # chọn tập lượt A1 khác

Đọc các lượt `runs/amanda-A1/<A1_GLOB>` đã đủ 1.061 câu, `runs/amanda-A0/full` và `runs/amanda-A4/full-g1` (nếu có),
rồi ghi `results/runs_summary.json`. Điểm được chấm lại từ predictions.jsonl bằng hàm chấm của tác giả
(`evaluate_records`). Các mục: điểm từng lượt, phễu pipeline (dừng ở vòng 0 / vào vòng bổ sung / có tri thức trong
prompt), các câu có tri thức trong prompt, câu mở khi cắt câu trả lời, bộ chấm chặt, thống kê câu hỏi của Explorer.
"""
from __future__ import annotations

import glob, json, os, re, statistics as st, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "third_party/AMANDA/src", REPO / "src", REPO / "scripts"):
    sys.path.insert(0, str(p))

from amanda.evaluation.evaluate_medvqa import evaluate_records, _closed_answer_is_correct  # noqa: E402
from amanda.evaluation.metrics import calculate_f1score                                  # noqa: E402
import verbosity_probe as vp                                                              # noqa: E402
import qbank_analyze as qa                                                                # noqa: E402
from closed_bias_audit import strict_ok                                                   # noqa: E402

R = REPO / "runs"
GT_LIST = json.loads((REPO / "data/slake/test_en_amanda.json").read_text(encoding="utf8"))
GT = {str(g["question_id"]): g for g in GT_LIST}
ANS = lambda k: GT[k]["conversations"][1]["value"]                                        # noqa: E731
CLOSED = [k for k, g in GT.items() if g["answer_type"] == "close"]


def load(d: Path, name: str) -> dict[str, dict]:
    return {str(json.loads(l)["question_id"]): json.loads(l) for l in open(d / name) if l.strip()}


def score(P: dict) -> dict:
    m = evaluate_records(ground_truth=GT_LIST, predictions=list(P.values())).metrics
    return {"open": m["Open Recall"], "closed": m["Closed Accuracy"]}


def main() -> None:
    pat = os.environ.get("A1_GLOB", "full-r*")                   # các lượt A1 của bước 10
    dirs = [Path(d) for d in sorted(glob.glob(str(R / f"amanda-A1/{pat}")))
            if (Path(d) / "predictions.jsonl").is_file() and sum(1 for _ in open(Path(d) / "predictions.jsonl")) >= len(GT)]
    if not dirs:
        sys.exit(f"chưa có lượt runs/amanda-A1/{pat} nào đủ {len(GT)} câu — chạy bước 10 trước")
    for need in ("amanda-A0/full", "amanda-A4/full-g1"):
        if not (R / need / "predictions.jsonl").is_file():
            sys.exit(f"thiếu runs/{need} — chạy bước 08 và 09 trước")
    P = [load(d, "predictions.jsonl") for d in dirs]
    T = [load(d, "trace.jsonl") for d in dirs]
    S = [score(p) for p in P]
    a0, a4 = score(load(R / "amanda-A0/full", "predictions.jsonl")), score(load(R / "amanda-A4/full-g1", "predictions.jsonl"))
    op, cl = [s["open"] for s in S], [s["closed"] for s in S]
    out = {"runs": [d.name for d in dirs], "per_run": S,
           "A1": {"open": st.mean(op), "closed": st.mean(cl)},
           "sd": {"open": st.stdev(op) if len(op) > 1 else None, "closed": st.stdev(cl) if len(cl) > 1 else None},
           "band": {"open": max(op) - min(op), "closed": max(cl) - min(cl)},
           "A0": a0, "gain": {"open": st.mean(op) - a0["open"], "closed": st.mean(cl) - a0["closed"]},
           "A4": a4, "no_perceiver_delta_closed": a4["closed"] - st.mean(cl)}
    # pipeline: câu dừng ở vòng 0, câu vào vòng bổ sung, tri thức vào prompt
    pf = []
    for p, t in zip(P, T):
        reach = sum(1 for x in p.values() if (x.get("iterations") or 0) > 0)
        fired = sum(1 for x in p.values() if x.get("rag_context"))
        stop0 = sum(1 for x in t.values() if x["iterations"] == 0)
        z = [k for k in CLOSED if t[k]["iterations"] == 0]
        wrong0 = sum(not _closed_answer_is_correct(p[k]["text"].lower(), ANS(k).lower()) for k in z)
        pf.append({"reach": reach, "fired": fired, "stop0": stop0, "closed_stop0": len(z), "closed_stop0_wrong": wrong0})
    n = len(GT)
    out["pipeline"] = {"stop0_pct": 100 * st.mean(x["stop0"] for x in pf) / n,
                       "reach_pct": 100 * st.mean(x["reach"] for x in pf) / n,
                       "fired_mean": st.mean(x["fired"] for x in pf),
                       "fired_pct_all": 100 * st.mean(x["fired"] for x in pf) / n,
                       "fired_pct_reach": 100 * st.mean(x["fired"] / x["reach"] for x in pf if x["reach"]),
                       "closed_stop0_wrong_pct": 100 * st.mean(x["closed_stop0_wrong"] / x["closed_stop0"] for x in pf),
                       "per_run": pf}
    # hai câu ví dụ (qid cố định): diễn biến qua các lượt
    def trace_summary(k):
        rows = []
        for p, t in zip(P, T):
            h = " ".join(t[k].get("history") or [])
            rows.append({"iterations": t[k]["iterations"], "initial": t[k]["initial_answer"][:120],
                         "final": p[k]["text"][:120], "correct": _closed_answer_is_correct(p[k]["text"].lower(), ANS(k).lower()),
                         "chambers": "chamber" in h.lower(), "rag": bool(t[k].get("rag_context") and t[k]["rag_context"] != "None"),
                         "analysis": (t[k].get("final_analysis") or "")[:160]})
        return rows
    out["trace_12113"], out["trace_11986"] = trace_summary("12113"), trace_summary("11986")
    # cận trên phần đóng góp của đồ thị cho Closed: số câu đóng có tri thức vào prompt / 416 (giả sử mọi câu đó đều được sửa đúng)
    fired_closed = [sum(1 for k in CLOSED if p[k].get("rag_context")) for p in P]
    out["pipeline"]["fired_closed_mean"] = st.mean(fired_closed)
    out["pipeline"]["graph_closed_upper_bound"] = 100 * st.mean(fired_closed) / len(CLOSED)
    # các câu có tri thức SPOKE vào prompt: đáp án đầu/cuối đúng không, Reasoner có dẫn tri thức không (từ khoá)
    def ok_any(k, txt):
        if GT[k]["answer_type"] == "close":
            return _closed_answer_is_correct(txt.lower(), ANS(k).lower())
        return calculate_f1score(txt.lower(), ANS(k).lower())[2] >= 0.99
    cite = re.compile(r"grounded|medical knowledge|knowledge graph|additional (?:medical )?(?:information|knowledge)|provided (?:medical )?knowledge", re.I)
    fired_rows = []
    for d, p, t in zip(dirs, P, T):
        for k, x in t.items():
            if x.get("rag_context") and x["rag_context"] != "None":
                an = " ".join(h for h in x.get("history") or [] if "analysis" in h.lower()) + " " + (x.get("final_analysis") or "")
                fired_rows.append({"run": d.name, "qid": k, "type": GT[k]["answer_type"], "init_ok": ok_any(k, x["initial_answer"]),
                                   "final_ok": ok_any(k, p[k]["text"]), "cited": bool(cite.search(an)), "facts": x["rag_context"][:200]})
    out["fired"] = {"n": len(fired_rows), "cited": sum(r["cited"] for r in fired_rows),
                    "cited_final_ok": sum(1 for r in fired_rows if r["cited"] and r["final_ok"]),
                    "wrong_to_right": sum(1 for r in fired_rows if not r["init_ok"] and r["final_ok"]),
                    "right_to_wrong": sum(1 for r in fired_rows if r["init_ok"] and not r["final_ok"]), "rows": fired_rows}
    # ví dụ Evaluator: câu đóng X-quang ngực, dừng ở vòng 0 (điểm ≥ 4) và sai ở MỌI run
    out["evaluator_examples"] = [k for k in CLOSED if GT[k]["modality"] == "X-Ray" and all(
        t[k]["iterations"] == 0 and not _closed_answer_is_correct(p[k]["text"].lower(), ANS(k).lower()) for p, t in zip(P, T))]
    # ví dụ câu hỏi phụ cần thông tin ngoài ảnh (Explorer gốc, caption đúng)
    outside_ex = []
    for t in T[:1]:
        for kk, r in t.items():
            for x in r.get("history") or []:
                if x.startswith("Sub-Question") and qa.OUTSIDE.search(x):
                    outside_ex.append((kk, x[:160]))
    out["outside_examples"] = outside_ex[:15]
    # câu mở: điểm khi cắt câu trả lời về cùng độ dài
    A0P = vp.load(R / "amanda-A0/full/predictions.jsonl")
    modes = ["5", "10", "20", "sent1", "full"]
    out["verbosity"] = {"modes": modes, "A0": [vp.score(A0P, m)[0] for m in modes],
                        "A1_runs": [[vp.score(vp.load(d / "predictions.jsonl"), m)[0] for m in modes] for d in dirs]}
    out["verbosity"]["A1_mean"] = [st.mean(r[i] for r in out["verbosity"]["A1_runs"]) for i in range(len(modes))]
    # ví dụ thước đo câu mở (qid 12495)
    k = "12495"; gold = ANS(k).lower(); a0t = load(R / "amanda-A0/full", "predictions.jsonl")[k]["text"]; a1t = P[0][k]["text"]
    _, p0, r0 = calculate_f1score(a0t.lower(), gold); _, p1, r1 = calculate_f1score(a1t.lower(), gold)
    out["metric_example"] = {"qid": k, "base_words": len(a0t.split()), "amanda_words": len(a1t.split()),
                             "base_recall": r0, "amanda_recall": r1, "amanda_precision": p1}
    # hàm chấm chặt (phụ lục B)
    strict = lambda p: 100 * st.mean(strict_ok(p[k]["text"], ANS(k), GT[k]["conversations"][0]["value"]) for k in CLOSED)  # noqa: E731
    out["strict"] = {"A1_author": st.mean(cl), "A1_strict": st.mean(strict(p) for p in P)}
    # Explorer gốc: câu hỏi phụ đầu tiên ở tầng 1, câu hỏi cần thông tin ngoài ảnh (từ khoá, như qbank_analyze)
    ex = []
    for d, t in zip(dirs, T):
        sq_all = n_first = t1 = outside = 0
        for kk, r in t.items():
            h = r.get("history") or []
            sq = [re.sub(r"^Sub-Question \d+:\s*", "", x) for x in h if x.startswith("Sub-Question")]
            if not sq:
                continue
            n_first += 1; t1 += bool(qa.GENERAL.search(sq[0]))
            sq_all += len(sq); outside += sum(bool(qa.OUTSIDE.search(q)) for q in sq)
        ex.append({"questions_with_explorer": n_first, "first_tier1": t1, "sub_questions": sq_all, "outside": outside})
    out["explorer"] = {"per_run": ex,
                       "first_tier1_pct": 100 * sum(x["first_tier1"] for x in ex) / max(1, sum(x["questions_with_explorer"] for x in ex)),
                       "outside_pct": 100 * sum(x["outside"] for x in ex) / max(1, sum(x["sub_questions"] for x in ex))}
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results/runs_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("trace_12113", "trace_11986", "verbosity")}, indent=1, ensure_ascii=False)[:3500])
    print("12113:", [(x["iterations"], x["correct"], x["chambers"], x["rag"]) for x in out["trace_12113"]])
    print("11986:", [(x["iterations"], x["correct"], x["initial"][:50]) for x in out["trace_11986"]])
    print("→ results/runs_summary.json")
    print("verbosity A0:", [round(x, 2) for x in out["verbosity"]["A0"]], "A1:", [round(x, 2) for x in out["verbosity"]["A1_mean"]])


if __name__ == "__main__":
    main()
