#!/usr/bin/env python3
"""Kiểm toán câu đóng: nền hằng số, bộ chấm chặt, nhiễm dữ liệu (proxy), lật đáp án A0 → A1.

    .venv/bin/python scripts/closed_bias_audit.py

Đọc predictions sẵn có, KHÔNG chạy mô hình. Bốn phép thử:

  1. Nền hằng số: luôn đáp Yes / luôn đáp No được bao nhiêu điểm Closed?
  2. Kẽ hở của bộ chấm: với câu đóng KHÔNG phải yes/no ("liver or spleen?"), code tác giả
     (`_closed_answer_is_correct`) chỉ kiểm CHUỖI CON ⇒ câu trả lời nhắc cả hai phương án vẫn được
     tính đúng. Bộ chấm CHẶT ở đây lấy phương án được nhắc ĐẦU TIÊN. Bài báo (Phụ lục C) chỉ mô tả
     luật "từ yes/no đầu tiên", không nhắc nhánh chuỗi con.
  3. Nhiễm dữ liệu của LLM agent (proxy yếu): A4 = bỏ Perceiver, agent không thấy ảnh. Nếu agent
     thuộc lòng SLAKE thì trên các câu nó CHỊU đáp, độ chính xác phải vượt xa nền hằng số.
  4. Lật đáp án A0 → A1 trên từng câu đóng: ✗→✓ so với ✓→✗, qua mọi lần chạy A1.

⛔ Bộ chấm chặt là heuristic (tách phương án bằng regex) — dùng để đo CHIỀU và CỠ của kẽ hở,
không dùng thay số chính thức.
"""
from __future__ import annotations

import collections, glob, json, re, statistics as st, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "AMANDA" / "src"))
from amanda.evaluation.evaluate_medvqa import _closed_answer_is_correct, _first_yes_no  # noqa: E402
from amanda.evaluation.metrics import normalize_word  # noqa: E402

GT = {str(r["question_id"]): r for r in json.load(open(REPO / "data/slake/test_en_amanda.json"))}
CLOSED = {k: r for k, r in GT.items() if r["answer_type"].lower() != "open"}
ans = lambda r: r["conversations"][1]["value"]                         # noqa: E731
ques = lambda r: r["conversations"][0]["value"]                        # noqa: E731
YN = {k for k, r in CLOSED.items() if normalize_word(ans(r)).strip() in ("yes", "no")}

RUNS = [("A0", "runs/amanda-A0/full/"), ("A0-bug", "runs/amanda-A0/full-BUG-rawdecode/"),
        ("A1", "runs/amanda-A1/full*/"),
        ("A2", "runs/amanda-A2/full*/"), ("A3", "runs/amanda-A3/full*/"),
        ("A4", "runs/amanda-A4/full*/"), ("B2", "runs/amanda-B2/full*/"),
        ("C2", "runs/amanda-C2/full*/"), ("C5", "runs/amanda-C5/full*/")]


def done_dirs(pat: str) -> list[str]:
    """Chỉ lấy lượt chạy đã đủ 1.061 câu (có metrics.json) — lượt đang dở bị bỏ qua."""
    return [str(Path(d).relative_to(REPO)) for d in sorted(glob.glob(str(REPO / pat)))
            if (Path(d) / "metrics.json").is_file()
            and sum(1 for _ in open(Path(d) / "predictions.jsonl")) >= len(GT)]


def load(d: str) -> dict[str, str]:
    return {str(json.loads(l)["question_id"]): json.loads(l)["text"]
            for l in open(REPO / d / "predictions.jsonl")}


def options(q: str) -> list[str]:
    q = q.replace("<image>", "").lower()
    tail = q.split(",", 1)[1] if "," in q else q
    tail = re.sub(r"^.*?\b(is this an?|is the abnormality|is the brain enhancing tumor|is this)\b", "", tail)
    return [p.strip() for p in re.split(r",| or ", tail.strip(" ?")) if p.strip()]


def strict_ok(pred: str, gt: str, q: str) -> bool:
    g = normalize_word(gt).strip()
    if g in ("yes", "no"):
        return _closed_answer_is_correct(pred.lower(), gt.lower())
    p = normalize_word(pred.lower())
    hits = [(m.start(), o) for o in options(q)
            for m in [re.search(r"\b" + re.escape(normalize_word(o).split()[0]) + r"\b", p)] if m]
    if not hits:
        return g in p
    return normalize_word(min(hits)[1]).split()[0] == g.split()[0]


def closed_scores(P: dict[str, str]) -> tuple[float, float]:
    lax = sum(_closed_answer_is_correct(P[k].lower(), ans(r).lower()) for k, r in CLOSED.items())
    strict = sum(strict_ok(P[k], ans(r), ques(r)) for k, r in CLOSED.items())
    return 100 * lax / len(CLOSED), 100 * strict / len(CLOSED)


def main() -> None:
    n_yes = sum(normalize_word(ans(CLOSED[k])).strip() == "yes" for k in YN)
    print(f"== 1. nền hằng số · closed {len(CLOSED)} = yes/no {len(YN)} + khác {len(CLOSED) - len(YN)}")
    print(f"   luôn Yes {100 * n_yes / len(CLOSED):.2f} · luôn No {100 * (len(YN) - n_yes) / len(CLOSED):.2f}"
          f"   (trên riêng yes/no: {100 * n_yes / len(YN):.1f} / {100 * (len(YN) - n_yes) / len(YN):.1f})")

    print("\n== 2. bộ chấm tác giả (lỏng) vs chặt")
    for name, pat in RUNS:
        rs = [closed_scores(load(d)) for d in done_dirs(pat)]
        if not rs:
            print(f"   {name:8} (chưa có lượt chạy)"); continue
        print(f"   {name:8} n={len(rs)}  lỏng {st.mean(r[0] for r in rs):6.2f}"
              f"  chặt {st.mean(r[1] for r in rs):6.2f}  Δ {st.mean(r[1] - r[0] for r in rs):+.2f}")

    if not done_dirs("runs/amanda-A4/full-g1/") or not done_dirs("runs/amanda-A0/full/"):
        print("\n(bỏ qua mục 3–4: cần A4/full-g1 và A0/full)"); return
    a4 = load("runs/amanda-A4/full-g1/")
    answered = [k for k in YN if _first_yes_no(normalize_word(a4[k]))]
    acc = sum(_closed_answer_is_correct(a4[k].lower(), ans(CLOSED[k]).lower()) for k in answered)
    yes = sum(normalize_word(ans(CLOSED[k])).strip() == "yes" for k in answered)
    print(f"\n== 3. A4 (không ảnh): chịu đáp {len(answered)}/{len(YN)} câu yes/no"
          f" · đúng {100 * acc / len(answered):.1f}% · luôn-Yes trên cùng tập {100 * yes / len(answered):.1f}%")

    print("\n== 4. lật đáp án A0 → A1 (câu đóng, bộ chấm tác giả)")
    a0 = load("runs/amanda-A0/full/")
    for d in done_dirs("runs/amanda-A1/full*/"):
        a1 = load(d)
        f = collections.Counter()
        for k, r in CLOSED.items():
            f[(_closed_answer_is_correct(a0[k].lower(), ans(r).lower()),
               _closed_answer_is_correct(a1[k].lower(), ans(r).lower()))] += 1
        print(f"   {Path(d).name:10} giữ✓ {f[(1, 1)]:3} · giữ✗ {f[(0, 0)]:3} · ✗→✓ {f[(0, 1)]:3} · ✓→✗ {f[(1, 0)]:3}"
              f" · ròng {f[(0, 1)] - f[(1, 0)]:+d}")


if __name__ == "__main__":
    main()
