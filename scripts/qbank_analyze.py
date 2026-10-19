#!/usr/bin/env python3
"""Phân tích thí nghiệm question bank — đúng luật quyết định đăng ký trước (|t| > 2).

    .venv/bin/python scripts/qbank_analyze.py            # bộ đầy đủ: runs/amanda-{F0,F1,FQ}/full-r*
    .venv/bin/python scripts/qbank_analyze.py --pilot    # pilot: runs/amanda-{F1,FQ}/limit200-pilot
    .venv/bin/python scripts/qbank_analyze.py --pilot --fq-tag=pilot11   # FQ lấy pilot Explorer v1.1

In ra:
  A. điểm từng lượt: Closed (bộ chấm tác giả + bộ chấm chặt) · Open recall (nguyên văn + cắt còn câu đầu) · độ dài
  B. ⭐ luật quyết định: SD gộp trong-arm (df = Σ(k−1)), t = Δ / (SD·√(1/k₁+1/k₂)), |t| > 2 mới gọi là khác
  C. chẩn đoán Explorer từ trace: độ phủ, tầng của sub-question đầu, Yes:No, câu đòi thông tin ngoài ảnh,
     tỉ lệ rơi (FQ), ĐỘ CHÍNH XÁC ĐỊNH TUYẾN so với metadata SLAKE (chỉ để báo cáo — router không thấy metadata)
  D. Closed theo vùng (route thật theo metadata), FQ − F1 — kiểm giả thuyết phụ K8
"""
from __future__ import annotations

import collections, glob, json, re, statistics as st, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "AMANDA" / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from amanda.evaluation.evaluate_medvqa import _closed_answer_is_correct  # noqa: E402
from amanda.evaluation.metrics import calculate_f1score  # noqa: E402
from closed_bias_audit import strict_ok  # noqa: E402
from verbosity_probe import truncate  # noqa: E402
from amanda_explorer_audit import GENERAL, YESNO  # noqa: E402
# ⛔ Dùng regex OUTSIDE của CỔNG BANK (qbank_graph.py), KHÔNG dùng bản của amanda_explorer_audit.py: bản kia đếm cả
#    T1/T2/FLAIR, mà câu "ảnh NÀY là T1 hay T2?" là hợp lệ ⇒ đếm nhầm (bản đầu cho FQ 215 câu "ngoài ảnh").
from qbank_graph import OUTSIDE  # noqa: E402
from amanda_lab.qbank_explorer import QBank  # noqa: E402

PILOT = "--pilot" in sys.argv
ARMS = ("F1", "FQ") if PILOT else ("F0", "F1", "FQ")
PAT = "limit200-pilot" if PILOT else "full-r*"
GT = {str(r["question_id"]): r for r in json.load(open(REPO / "data/slake/test_en_amanda.json"))}
MOD = {"X-Ray": "xray", "CT": "ct", "MRI": "mri"}
REG = {"Lung": "chest", "Chest_mediastinal": "chest", "Abdomen": "abdomen_pelvis", "Pelvic Cavity": "abdomen_pelvis",
       "Brain": "head", "Brain_Tissue": "head", "Brain_Face": "head", "Neck": "neck"}
ans = lambda k: GT[k]["conversations"][1]["value"]                       # noqa: E731
ques = lambda k: GT[k]["conversations"][0]["value"]                      # noqa: E731


FQ_TAG = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--fq-tag=")), None)


def runs(arm: str) -> list[Path]:
    """--fq-tag=pilot11 ⇒ FQ lấy thư mục limit200-pilot11 (Explorer v1.1), các arm khác giữ nguyên."""
    pat = f"limit200-{FQ_TAG}" if (arm == "FQ" and FQ_TAG) else PAT
    return [Path(d) for d in sorted(glob.glob(str(REPO / f"runs/amanda-{arm}/{pat}")))
            if (Path(d) / "metrics.json").is_file()]


def load(d: Path, name: str) -> dict[str, dict]:
    return {str(json.loads(l)["question_id"]): json.loads(l) for l in open(d / name) if l.strip()}


# ⭐ Endpoint CHÍNH (đăng ký bổ sung 25/09, trước khi có số bộ đầy đủ): tập con X-quang ngực. Metadata SLAKE chỉ dùng
#    để CHỌN TẬP ĐÁNH GIÁ, không bao giờ vào mô hình.
CXR = {k for k, g in GT.items() if g["modality"] == "X-Ray" and g["location"] == "Lung"}


def scores(d: Path, keys: set[str] | None = None) -> dict[str, float]:
    P = load(d, "predictions.jsonl")
    if keys is not None:
        P = {k: v for k, v in P.items() if k in keys}
    cl = [k for k in P if GT[k]["answer_type"] == "close"]
    op = [k for k in P if GT[k]["answer_type"] == "open"]
    return {
        "closed": 100 * st.mean(_closed_answer_is_correct(P[k]["text"].lower(), ans(k).lower()) for k in cl),
        "closed_strict": 100 * st.mean(strict_ok(P[k]["text"], ans(k), ques(k)) for k in cl),
        "open": 100 * st.mean(calculate_f1score(P[k]["text"].lower(), ans(k).lower())[2] for k in op),
        "open_sent1": 100 * st.mean(calculate_f1score(truncate(P[k]["text"], "sent1").lower(),
                                                      ans(k).lower())[2] for k in op),
        "words": st.mean(len(P[k]["text"].split()) for k in P),
    }


def decide(S: dict, title: str) -> None:
    print("\n" + title)
    for metric in ("closed", "closed_strict", "open", "open_sent1"):
        by = {a: [s[metric] for s in S[a]] for a in ARMS}
        sd, df = pooled_sd(by)
        line = f"  {metric:13} " + "  ".join(f"{a} {st.mean(v):6.2f}" for a, v in by.items())
        line += f"  | SD gộp {sd:.2f} (df={df})"
        for x, y in (("FQ", "F1"), ("F1", "F0"), ("FQ", "F0")):
            if x in by and y in by:
                delta = st.mean(by[x]) - st.mean(by[y])
                se = sd * (1 / len(by[x]) + 1 / len(by[y])) ** 0.5 if df else float("nan")
                t = delta / se if se == se and se > 0 else float("nan")
                line += f"  | {x}−{y} {delta:+.2f} t={t:+.2f}{' ✱' if abs(t) > 2 else ''}"
        print(line)


def pooled_sd(by_arm: dict[str, list[float]]) -> tuple[float, int]:
    ss = sum(sum((x - st.mean(v)) ** 2 for x in v) for v in by_arm.values() if len(v) > 1)
    df = sum(len(v) - 1 for v in by_arm.values() if len(v) > 1)
    return ((ss / df) ** 0.5 if df else float("nan")), df


def diagnostics(arm: str, d: Path, qb: QBank) -> dict:
    T = load(d, "trace.jsonl")
    text2ids = collections.defaultdict(set)
    for qid, q in qb.questions.items():
        text2ids[q["en_open"]].add(qid)
        text2ids[q["en_yn"]].add(qid)
    cov = first_t1 = yes = no = outside = route_ok_m = route_ok_r = routed = 0
    for k, r in T.items():
        h = r.get("history") or []
        sq = [re.sub(r"^Sub-Question \d+:\s*", "", x) for x in h if x.startswith("Sub-Question")]
        sa = [re.sub(r"^Sub-Answer \d+:\s*", "", x) for x in h if x.startswith("Sub-Answer")]
        if not sq:
            continue
        cov += 1
        if arm == "FQ":
            first_t1 += any(qb.questions[i]["tier"] == 1 for i in text2ids.get(sq[0], ()))
            refl = [x for x in h if x.startswith("Reflection: route=")]
            if refl:
                route = re.search(r"route=(\w+)", refl[-1]).group(1)
                routed += 1
                m_pred, _, r_pred = route.partition("_")
                route_ok_m += m_pred == MOD[GT[k]["modality"]]
                route_ok_r += r_pred == REG[GT[k]["location"]]
        else:
            first_t1 += bool(GENERAL.search(sq[0]))
        for q, a in zip(sq, sa):
            outside += bool(OUTSIDE.search(q))
            if YESNO.match(q):
                yes += bool(re.match(r"^\W*yes\b", a, re.I))
                no += bool(re.match(r"^\W*no\b", a, re.I))
    m = json.loads((d / "metrics.json").read_text())
    stats = m.get("qbank_stats") or {}
    return {"coverage": cov / len(T), "first_tier1": first_t1 / max(cov, 1), "yes": yes, "no": no,
            "outside": outside, "route_mod": route_ok_m / routed if routed else None,
            "route_reg": route_ok_r / routed if routed else None,
            "fallback": stats.get("fallback", 0) / stats["calls"] if stats.get("calls") else None}


def vn(x: float, d: int = 2, sign: bool = False) -> str:
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace(".", "{,}").replace("-", "$-$").replace("+", "$+$")


def write_tex(S: dict, D: dict, S_cxr: dict) -> None:
    """Sinh results/qbank_numbers.tex (macro LaTeX chứa các con số, để không phải gõ tay)."""
    m = lambda a, k: st.mean(s[k] for s in S[a])                            # noqa: E731
    out = {"qbReady": "1", "qbNruns": str(min(len(S[a]) for a in ARMS))}
    for a, key in (("F0", "Fz"), ("F1", "Fo"), ("FQ", "Fq")):
        out[f"qb{key}Closed"] = vn(m(a, "closed"))
        out[f"qb{key}Open"] = vn(m(a, "open"))
        out[f"qb{key}OpenS"] = vn(m(a, "open_sent1"))
    for metric, tag in (("closed", "Closed"), ("open_sent1", "OpenS")):
        by = {a: [s[metric] for s in S[a]] for a in ARMS}
        sd, _ = pooled_sd(by)
        if tag == "Closed":
            out["qbSDclosed"] = vn(sd)
        for x, y, k in (("FQ", "F1", "qo"), ("F1", "F0", "oz")):
            if tag == "OpenS" and k == "oz":
                continue
            delta = st.mean(by[x]) - st.mean(by[y])
            t = delta / (sd * (1 / len(by[x]) + 1 / len(by[y])) ** 0.5)
            out[f"qbD{k}{tag}"] = vn(delta, sign=True)
            out[f"qbT{k}{tag}"] = vn(t, sign=True)
    out["qbSDopen"] = vn(pooled_sd({a: [s["open"] for s in S[a]] for a in ARMS})[0])
    # ⭐ endpoint chính: X-quang ngực
    mc = lambda a, k: st.mean(s[k] for s in S_cxr[a])                         # noqa: E731
    for a, key in (("F0", "Fz"), ("F1", "Fo"), ("FQ", "Fq")):
        out[f"qbCxr{key}Closed"] = vn(mc(a, "closed"))
        out[f"qbCxr{key}OpenS"] = vn(mc(a, "open_sent1"))
    for metric, tag in (("closed", "Closed"), ("open_sent1", "OpenS")):
        by = {a: [s[metric] for s in S_cxr[a]] for a in ARMS}
        sd, _ = pooled_sd(by)
        for x, y, k in (("FQ", "F1", "qo"), ("F1", "F0", "oz")):
            delta = st.mean(by[x]) - st.mean(by[y])
            t = delta / (sd * (1 / len(by[x]) + 1 / len(by[y])) ** 0.5)
            out[f"qbCxrD{k}{tag}"] = vn(delta, sign=True)
            out[f"qbCxrT{k}{tag}"] = vn(t, sign=True)
    pct = lambda v: "---" if v is None else vn(100 * v, 1) + "\\,\\%"         # noqa: E731
    avg = lambda a, k: (st.mean(x[k] for x in D[a] if x[k] is not None)       # noqa: E731
                        if any(x[k] is not None for x in D[a]) else None)
    out["qbFoTierOne"], out["qbFqTierOne"] = pct(avg("F1", "first_tier1")), pct(avg("FQ", "first_tier1"))
    out["qbFoOutside"] = str(round(st.mean(x["outside"] for x in D["F1"])))
    out["qbFqOutside"] = str(round(st.mean(x["outside"] for x in D["FQ"])))
    out["qbFqFallback"] = pct(avg("FQ", "fallback"))
    out["qbRouteMod"], out["qbRouteReg"] = pct(avg("FQ", "route_mod")), pct(avg("FQ", "route_reg"))
    lines = ["% ⛔ SINH TỰ ĐỘNG bởi `.venv/bin/python scripts/qbank_analyze.py --tex` — KHÔNG sửa tay."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in out.items()]
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "qbank_numbers.tex").write_text("\n".join(lines) + "\n")
    # bản không định dạng dấu thập phân
    plain = {k: (v.replace("{,}", ".").replace("$+$", "+").replace("$-$", "-").replace("\\,\\%", "")
                 if k != "qbReady" else v) for k, v in out.items()}
    lines = ["% ⛔ SINH TỰ ĐỘNG bởi `scripts/qbank_analyze.py --tex` — dấu chấm thập phân."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in plain.items()]
    (REPO / "results" / "qbank_numbers_plain.tex").write_text("\n".join(lines) + "\n")
    print("\n→ results/qbank_numbers.tex · results/qbank_numbers_plain.tex")


def main() -> int:
    qb = QBank.load()
    R = {a: runs(a) for a in ARMS}
    if not all(R.values()):
        print("chưa đủ lượt chạy:", {a: len(v) for a, v in R.items()})
        return 1
    print(f"== A. điểm từng lượt ({'PILOT 200 câu' if PILOT else 'đầy đủ 1.061 câu'})")
    S = {a: [scores(d) for d in R[a]] for a in ARMS}
    for a in ARMS:
        for d, s in zip(R[a], S[a]):
            print(f"  {a} {d.name:15} Closed {s['closed']:6.2f} (chặt {s['closed_strict']:6.2f})  "
                  f"Open {s['open']:6.2f} (câu đầu {s['open_sent1']:6.2f})  {s['words']:6.1f} từ")

    S_cxr = {a: [scores(d, CXR) for d in R[a]] for a in ARMS}
    n_cl = sum(1 for k in CXR if GT[k]["answer_type"] == "close")
    decide(S_cxr, f"== B1. ⭐ ENDPOINT CHÍNH — X-quang ngực ({len(CXR)} câu, {n_cl} câu đóng) — |t| > 2 mới gọi là khác")
    decide(S, "== B2. phụ — toàn tập — |t| > 2 mới gọi là khác")
    if PILOT:
        print("  ⚠️ PILOT: 1 lượt mỗi arm ⇒ không có SD gộp, KHÔNG kết luận gì về hiệu ứng.")

    print("\n== C. chẩn đoán Explorer (trung bình qua các lượt)")
    for a in ARMS:
        D = [diagnostics(a, d, qb) for d in R[a]]
        f = lambda k: st.mean(x[k] for x in D if x[k] is not None) if any(x[k] is not None for x in D) else None  # noqa: E731
        fmt = lambda v, p=True: "—" if v is None else (f"{100*v:.1f}%" if p else f"{v:.1f}")  # noqa: E731
        print(f"  {a}: phủ {fmt(f('coverage'))} · sub-q đầu ở tầng 1 {fmt(f('first_tier1'))} · "
              f"Yes:No {sum(x['yes'] for x in D)}:{sum(x['no'] for x in D)} · ngoài-ảnh "
              f"{sum(x['outside'] for x in D)} · rơi {fmt(f('fallback'))} · định tuyến đúng phương thức "
              f"{fmt(f('route_mod'))} / vùng {fmt(f('route_reg'))}")

    if "--tex" in sys.argv and not PILOT:
        write_tex(S, {a: [diagnostics(a, d, qb) for d in R[a]] for a in ARMS}, S_cxr)

    print("\n== D. Closed theo vùng (metadata SLAKE, chỉ để phân tích), trung bình qua các lượt")
    groups = collections.defaultdict(list)
    for k, g in GT.items():
        if g["answer_type"] == "close":
            groups[f"{MOD[g['modality']]}_{REG[g['location']]}"].append(k)
    for grp, ks in sorted(groups.items(), key=lambda x: -len(x[1])):
        row = []
        for a in ARMS:
            vals = []
            for d in R[a]:
                P = load(d, "predictions.jsonl")
                kk = [k for k in ks if k in P]
                if kk:
                    vals.append(100 * st.mean(_closed_answer_is_correct(P[k]["text"].lower(), ans(k).lower())
                                              for k in kk))
            row.append(f"{a} {st.mean(vals):6.2f}" if vals else f"{a}    —")
        n = len([k for k in ks if k in load(R[ARMS[0]][0], 'predictions.jsonl')])
        print(f"  {grp:20} n={n:3}  " + "  ".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
