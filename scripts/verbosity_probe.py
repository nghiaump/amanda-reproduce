#!/usr/bin/env python3
"""⭐ Thước câu mở của Med-VQA là RECALL theo token ⇒ trả lời dài là tự động ăn điểm.

    .venv/bin/python scripts/verbosity_probe.py

Câu hỏi: phần tăng Open của AMANDA là **đúng hơn** hay chỉ là **dài hơn**?

Cách phân xử: **cắt bớt đáp án rồi chấm lại**. Nếu recall giữ nguyên khi cắt ⇒ thông tin đúng nằm ở
đầu câu, phần đuôi chỉ là nước. Nếu recall sụp theo độ dài ⇒ điểm đến từ việc rải token cầu may.

⛔ Chấm bằng CHÍNH `calculate_f1score` của tác giả, không tự viết lại thước.
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "AMANDA" / "src"))
from amanda.evaluation.metrics import calculate_f1score   # noqa: E402

GT = {str(g["question_id"]): g
      for g in json.loads((REPO / "data/slake/test_en_amanda.json").read_text(encoding="utf8"))}


def load(p: Path) -> dict:
    return {str(json.loads(l)["question_id"]): json.loads(l)
            for l in p.read_text(encoding="utf8").splitlines() if l.strip()}


def truncate(text: str, mode: str) -> str:
    if mode == "full":
        return text
    if mode == "sent1":
        m = re.split(r"(?<=[.!?])\s+", text.strip())
        return m[0] if m else text
    return " ".join(text.split()[: int(mode)])


def score(preds: dict, mode: str) -> tuple[float, float, float, float]:
    f1s = ps = rs = 0.0
    n = 0
    L = 0
    for q, g in GT.items():
        if g["answer_type"] != "open":
            continue
        t = truncate(preds[q]["text"], mode)
        f, p, r = calculate_f1score(t, g["conversations"][1]["value"])
        f1s += f; ps += p; rs += r; n += 1; L += len(t.split())
    return 100 * rs / n, 100 * ps / n, 100 * f1s / n, L / n


if __name__ == "__main__":
    # A0 không có hậu tố; các arm pipeline lấy lượt r1
    path = lambda a: REPO / (f"runs/amanda-{a}/full/predictions.jsonl" if a == "A0"      # noqa: E731
                             else f"runs/amanda-{a}/full-r1/predictions.jsonl")
    arms = [(n, load(path(a)))
            for n, a in (("A0 baseline", "A0"), ("A1 SPOKE", "A1"),
                         ("A2 -Retr", "A2"), ("A3 -Expl", "A3"), ("C2 SLAKE-KG", "C2"), ("B2 SHAM", "B2"))
            if path(a).is_file() and len(path(a).read_text().splitlines()) >= 1061]
    if not arms or arms[0][0] != "A0 baseline":
        sys.exit("cần runs/amanda-A0/full và ít nhất một arm pipeline ở full-r1")

    print("Câu MỞ (n=645) · thước = calculate_f1score của tác giả · cột Recall là thứ bài báo báo cáo\n")
    print(f"{'cắt còn':<12}" + "".join(f"{n:>26}" for n, _ in arms))
    print(f"{'':<12}" + "".join(f"{'Recall':>8}{'Prec':>7}{'F1':>6}{'từ':>5}" for _ in arms))
    print("-" * (12 + 26 * len(arms)))
    rows = {}
    for mode, label in (("full", "nguyên"), ("sent1", "câu đầu"),
                        ("30", "30 từ"), ("20", "20 từ"), ("10", "10 từ"), ("5", "5 từ")):
        line = f"{label:<12}"
        rows[label] = {}
        for name, D in arms:
            r, p, f, L = score(D, mode)
            rows[label][name] = r
            line += f"{r:>8.2f}{p:>7.2f}{f:>6.2f}{L:>5.0f}"
        print(line)

    print("\n⭐ Khoảng cách so với A0, ở nguyên văn vs khi cắt còn câu đầu:")
    base_f, base_s = rows["nguyên"]["A0 baseline"], rows["câu đầu"]["A0 baseline"]
    for name, _ in arms[1:]:
        f, sn = rows["nguyên"][name], rows["câu đầu"][name]
        print(f"  {name:<14} nguyên {f - base_f:+6.2f}   cắt câu đầu {sn - base_s:+6.2f}   "
              f"⇒ {'CÒN LẠI khi cùng độ dài' if sn - base_s > 0 else 'BỐC HƠI khi cùng độ dài'}")
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results/verbosity.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf8")
