#!/usr/bin/env python3
"""Tổng hợp các arm cắt bỏ so với A1: trung bình, SD, hiệu, t với SD gộp.

    PYTHONPATH=src .venv/bin/python scripts/ablation_recompute.py

Đọc ``runs/amanda-{A1,A2,A3}/full-r*`` (bước 10) và ``runs/amanda-A4/full-g1`` (bước 09). Điểm chấm lại từ
predictions.jsonl bằng hàm chấm của tác giả. Ghi ``results/ablation_summary.json``.
t = hiệu trung bình / (SD gộp · sqrt(1/n1 + 1/n2)), df = n1 + n2 − 2.
"""
from __future__ import annotations

import glob, json, math, statistics as st, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "third_party/AMANDA/src", REPO / "src", REPO / "scripts"):
    sys.path.insert(0, str(p))

from summarize_runs import GT, load, score  # noqa: E402

R = REPO / "runs"
ARMS = {"A1": "amanda-A1/full-r*", "A2": "amanda-A2/full-r*", "A3": "amanda-A3/full-r*",
        "A4": "amanda-A4/full-g1"}


def complete(d: str) -> bool:
    f = Path(d) / "predictions.jsonl"
    return f.is_file() and sum(1 for _ in open(f)) >= len(GT) and (Path(d) / "metrics.json").is_file()


def arm_stats(pattern: str) -> dict | None:
    dirs = [Path(d) for d in sorted(glob.glob(str(R / pattern))) if complete(d)]
    if not dirs:
        return None
    S, P = [], []
    for d in dirs:
        p = load(d, "predictions.jsonl")
        S.append(score(p)); P.append(p)
    op, cl = [s["open"] for s in S], [s["closed"] for s in S]
    return {"runs": [d.name for d in dirs], "per_run": S, "n_runs": len(dirs),
            "open": st.mean(op), "closed": st.mean(cl),
            "sd_open": st.stdev(op) if len(op) > 1 else None, "sd_closed": st.stdev(cl) if len(cl) > 1 else None,
            "fired_mean": st.mean(sum(1 for x in p.values() if x.get("rag_context")) for p in P),
            "mean_iterations": st.mean(st.mean((x.get("iterations") or 0) for x in p.values()) for p in P),
            "stop0_pct": st.mean(100 * sum(1 for x in p.values() if not x.get("iterations")) / len(p) for p in P)}


def t_test(a: dict, b: dict, key: str) -> float | None:
    sa, sb = a.get(f"sd_{key}"), b.get(f"sd_{key}")
    if sa is None or sb is None:
        return None
    n1, n2 = a["n_runs"], b["n_runs"]
    sp = math.sqrt(((n1 - 1) * sa ** 2 + (n2 - 1) * sb ** 2) / (n1 + n2 - 2))
    return (a[key] - b[key]) / (sp * math.sqrt(1 / n1 + 1 / n2)) if sp else None


def main() -> None:
    out = {arm: arm_stats(pat) for arm, pat in ARMS.items()}
    a1 = out["A1"]
    if not a1:
        sys.exit("chưa có lượt runs/amanda-A1/full-r* nào đủ 1.061 câu — chạy bước 10 trước")
    for arm in ("A2", "A3", "A4"):
        x = out[arm]
        if x is None:
            continue
        x["delta"] = {k: x[k] - a1[k] for k in ("open", "closed")}
        x["t"] = {k: t_test(x, a1, k) for k in ("open", "closed")}
        x["df"] = x["n_runs"] + a1["n_runs"] - 2
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results/ablation_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")
    for arm, x in out.items():
        if x:
            d = x.get("delta", {})
            print(f"{arm}: {x['n_runs']} run · Open {x['open']:.2f} (SD {x['sd_open'] or 0:.2f}) · "
                  f"Closed {x['closed']:.2f} (SD {x['sd_closed'] or 0:.2f})"
                  + (f" · Δ {d['open']:+.2f}/{d['closed']:+.2f} · t {x['t']['open']}/{x['t']['closed']}" if d else ""))
        else:
            print(f"{arm}: chưa có lượt nào đủ 1.061 câu")


if __name__ == "__main__":
    main()
