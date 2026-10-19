#!/usr/bin/env python3
"""Tổng hợp mọi lượt chạy đã xong trong runs/ thành một bảng.

    .venv/bin/python scripts/summarize.py                  # đọc runs/, ghi results/summary.{md,json}

Mỗi nhóm arm là một hàng: trung bình ± độ lệch chuẩn qua các lượt lặp, cùng Open Recall và Closed Accuracy
chấm bằng chính `evaluate_records` của tác giả (đọc từ metrics.json).

So sánh giữa hai arm trong cùng một khối:
    SD gộp trong-arm (df = Σ(k−1)) ·  t = Δ / (SD·√(1/k₁+1/k₂)).
Khối "main" so với A1; khối "forced" so FQ với F1 và F1 với F0.
"""
from __future__ import annotations

import argparse, glob, json, statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
N_FULL = 1061

# (khoá, khối, mô tả, mẫu thư mục trong runs/)
GROUPS = [
    ("A0",        "baseline", "LLaVA-Med một bước (bộ giải mã đã vá)",         "amanda-A0/full"),
    ("A0-bug",    "baseline", "LLaVA-Med một bước (bộ giải mã gốc)",           "amanda-A0/full-BUG-rawdecode"),
    ("A4",        "gate",     "AMANDA bỏ Perceiver",                           "amanda-A4/full-g*"),
    ("A1",        "main",     "AMANDA đầy đủ, Retriever = SPOKE",              "amanda-A1/full-r*"),
    ("A2",        "main",     "bỏ Retriever",                                  "amanda-A2/full-r*"),
    ("A3",        "main",     "bỏ Explorer",                                   "amanda-A3/full-r*"),
    ("C2",        "main",     "Retriever = SLAKE-KG",                          "amanda-C2/full-r*"),
    ("B2",        "main",     "Retriever SHAM (văn bản vô quan, cùng độ dài)", "amanda-B2/full-r*"),
    ("C5",        "main",     "SLAKE-KG nạp từ vòng 0 (qua caption)",          "amanda-C5/full-r*"),
    ("F0",        "forced",   "ép vòng Explorer, Explorer rỗng",               "amanda-F0/full-r*"),
    ("F1",        "forced",   "ép vòng Explorer, Explorer gốc",                "amanda-F1/full-r*"),
    ("FQ",        "forced",   "ép vòng Explorer, question bank",               "amanda-FQ/full-r*"),
    ("A1-ctlfix", "decode",   "A1, n=200, bộ giải mã vá",                      "amanda-A1/limit200-ctlfix"),
    ("A1-ctlraw", "decode",   "A1, n=200, bộ giải mã gốc",                     "amanda-A1/limit200-ctlraw"),
    ("A2-ctlfix", "decode",   "A2, n=200, bộ giải mã vá",                      "amanda-A2/limit200-ctlfix"),
    ("A2-ctlraw", "decode",   "A2, n=200, bộ giải mã gốc",                     "amanda-A2/limit200-ctlraw"),
]
BLOCK_REF = {"main": ["A1"], "forced": ["F1", "F0"]}


def complete_runs(runs: Path, pattern: str) -> list[Path]:
    out = []
    for d in sorted(glob.glob(str(runs / pattern))):
        d = Path(d)
        m, p = d / "metrics.json", d / "predictions.jsonl"
        if not (m.is_file() and p.is_file()):
            continue
        need = 200 if "limit200" in d.name else N_FULL
        if sum(1 for l in p.open(encoding="utf8") if l.strip()) >= need:
            out.append(d)
    return out


def collect(runs: Path) -> dict:
    res = {}
    for key, block, desc, pat in GROUPS:
        dirs = complete_runs(runs, pat)
        if not dirs:
            continue
        per = []
        for d in dirs:
            m = json.loads((d / "metrics.json").read_text(encoding="utf8"))
            per.append({"dir": d.name, "open": m["metrics"]["Open Recall"],
                        "closed": m["metrics"]["Closed Accuracy"],
                        "fired": m.get("retriever_fire"), "iters": m.get("mean_iterations")})
        res[key] = {"block": block, "desc": desc, "k": len(per), "runs": per,
                    "open": st.mean(r["open"] for r in per), "closed": st.mean(r["closed"] for r in per),
                    "open_sd": st.stdev(r["open"] for r in per) if len(per) > 1 else None,
                    "closed_sd": st.stdev(r["closed"] for r in per) if len(per) > 1 else None}
    return res


def pooled_sd(vals: list[list[float]]) -> tuple[float, int]:
    ss = sum(sum((x - st.mean(v)) ** 2 for x in v) for v in vals if len(v) > 1)
    df = sum(len(v) - 1 for v in vals if len(v) > 1)
    return ((ss / df) ** 0.5 if df else float("nan")), df


def tests(res: dict) -> list[dict]:
    out = []
    for block, refs in BLOCK_REF.items():
        arms = [k for k, v in res.items() if v["block"] == block]
        for metric in ("open", "closed"):
            by = {k: [r[metric] for r in res[k]["runs"]] for k in arms}
            sd, df = pooled_sd(list(by.values()))
            for ref in refs:
                if ref not in by:
                    continue
                for k in arms:
                    if k == ref or (block == "forced" and k != {"F1": "FQ", "F0": "F1"}[ref]):
                        continue            # khối forced: chỉ FQ − F1 và F1 − F0, đúng hai giả thuyết đã đăng ký
                    delta = st.mean(by[k]) - st.mean(by[ref])
                    se = sd * (1 / len(by[k]) + 1 / len(by[ref])) ** 0.5 if df else float("nan")
                    t = delta / se if se == se and se > 0 else float("nan")
                    out.append({"block": block, "metric": metric, "arm": k, "ref": ref,
                                "delta": delta, "t": t, "sd_pooled": sd, "df": df})
    return out


def fmt(x, sd=None, d=2):
    if x is None:
        return "—"
    return f"{x:.{d}f}" + (f" ± {sd:.2f}" if sd is not None else "")


def render(res: dict, tt: list[dict]) -> str:
    L = ["# Tổng hợp các lượt chạy (n = 1.061, trừ khối decode n = 200)", "",
         "Open = Open Recall, Closed = Closed Accuracy, chấm bằng bộ chấm của tác giả (metrics.json).", "",
         "| arm | mô tả | k | Open | Closed |", "|---|---|---|---|---|"]
    for key, block, desc, _ in GROUPS:
        r = res.get(key)
        if not r:
            continue
        L.append(f"| {key} | {desc} | {r['k']} | "
                 f"{fmt(r['open'], r['open_sd'])} | {fmt(r['closed'], r['closed_sd'])} |")
    L.append("")
    if tt:
        L += ["## So sánh trong khối", "",
              "| khối | chỉ số | arm − mốc | Δ | t | SD gộp (df) |", "|---|---|---|---|---|---|"]
        for x in tt:
            L.append(f"| {x['block']} | {x['metric']} | {x['arm']} − {x['ref']} | {x['delta']:+.2f} | "
                     f"{x['t']:+.2f} | {x['sd_pooled']:.2f} ({x['df']}) |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "runs"))
    a = ap.parse_args()

    res = collect(Path(a.runs))
    tt = tests(res)
    md = render(res, tt)
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "summary.md").write_text(md, encoding="utf8")
    (REPO / "results" / "summary.json").write_text(
        json.dumps({"groups": res, "tests": tt}, indent=1, ensure_ascii=False), encoding="utf8")
    print(md)
    print("→ results/summary.md · results/summary.json")


if __name__ == "__main__":
    main()
