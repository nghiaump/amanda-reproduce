#!/usr/bin/env python3
"""Explorer (Intrinsic Med-KA) THẬT SỰ hỏi gì? — đọc trace, không chạy mô hình.

    .venv/bin/python scripts/amanda_explorer_audit.py                 # các lần chạy mặc định
    .venv/bin/python scripts/amanda_explorer_audit.py path/trace.jsonl # một trace bất kỳ

Câu hỏi: bài báo (§3.3, Table 12) nói Explorer hỏi theo ba tầng *General observation →
Anatomical analysis → Detailed findings*. Trace có cho thấy trình tự đó không?

Đo năm thứ, tất cả trên `history` của trace:
  1. tỉ lệ câu có gọi Explorer, số lượt gọi, số sub-question mỗi lượt
  2. sub-question ĐẦU TIÊN có ở tầng "tổng quát" không   (🟨 heuristic từ khoá — xem GENERAL)
  3. tỉ lệ sub-question dạng yes/no, và Med-MLLM đáp Yes / No bao nhiêu
  4. sub-question hỏi thông tin KHÔNG THỂ có trong một ảnh đơn (ADC, chuỗi xung khác, phim cũ,
     bệnh sử...) và Med-MLLM có thừa nhận "không thấy" không
  5. vài cặp hỏi–đáp mẫu để đọc bằng mắt

⛔ Phép phân loại (2) và (4) là từ khoá ⇒ chỉ là CHỈ BÁO, phải đọc tay mẫu trước khi trích.
"""
from __future__ import annotations

import json, random, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT = [
    "runs/amanda-A1/full-r1/trace.jsonl",         # Explorer gốc, cấu hình gốc
    "runs/amanda-F1/full-r1/trace.jsonl",         # Explorer gốc, chế độ ép
    "runs/amanda-FQ/full-r1/trace.jsonl",         # question bank, chế độ ép
]

GENERAL = re.compile(r"\b(overall|general|entire|whole image|global|what (imaging )?modality"
                     r"|what type of (image|scan))\b", re.I)
YESNO = re.compile(r"^(is|are|does|do|was|were|can|could|has|have|did)\b", re.I)
OUTSIDE = re.compile(r"\b(ADC|prior|previous|comparison|clinical history|patient.s (history|symptoms|age)"
                     r"|laborator|other sequences?|different sequences|T1|T2|FLAIR|lateral view"
                     r"|additional views?|biopsy|symptom)", re.I)
CANT = re.compile(r"(cannot|can't|not possible|unable|not (visible|provided|available|shown)"
                  r"|no information|does not (show|provide|include))", re.I)


def strip(x: str) -> str:
    return re.sub(r"^Sub-(Question|Answer) \d+:\s*", "", x).strip()


def audit(path: Path, n_show: int = 4) -> None:
    rows = [json.loads(l) for l in path.open()]
    items = calls = 0
    first, qs, ans = [], [], []
    for r in rows:
        h = r.get("history") or []
        if not any(x.startswith("Sub-Question") for x in h):
            continue
        items += 1
        calls += r["iterations"]          # mỗi vòng tinh chỉnh = một lượt Explorer
        # số thứ tự sub-question CHẠY TIẾP qua các vòng (4, 5, 6...) ⇒ "Sub-Question 1" = lượt đầu
        first.append(strip(next(x for x in h if x.startswith("Sub-Question 1:"))))
        qs += [strip(x) for x in h if x.startswith("Sub-Question")]
        ans += [strip(x) for x in h if x.startswith("Sub-Answer")]
    pairs = list(zip(qs, ans))
    yn = [(q, a) for q, a in pairs if YESNO.match(q)]
    yes = sum(bool(re.match(r"^\W*yes\b", a, re.I)) for _, a in yn)
    no = sum(bool(re.match(r"^\W*no\b", a, re.I)) for _, a in yn)
    out = [(q, a) for q, a in pairs if OUTSIDE.search(q)]
    gen = sum(bool(GENERAL.search(q)) for q in first)

    print(f"== {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")
    print(f"  câu có gọi Explorer   {items}/{len(rows)} ({100 * items / len(rows):.1f}%)"
          f" · lượt gọi {calls} · sub-question {len(qs)} ({len(qs) / max(calls, 1):.2f}/lượt)")
    print(f"  sub-q đầu ở tầng tổng quát (từ khoá)  {gen}/{len(first)} ({100 * gen / len(first):.1f}%)")
    print(f"  sub-q dạng yes/no     {len(yn)}/{len(qs)} ({100 * len(yn) / len(qs):.0f}%)"
          f" · đáp Yes {yes} · No {no} · khác {len(yn) - yes - no}")
    print(f"  sub-q đòi thông tin ngoài ảnh đơn  {len(out)}"
          f" · trong đó Med-MLLM thừa nhận không thấy: {sum(bool(CANT.search(a)) for _, a in out)}")
    rnd = random.Random(0)
    print("  -- sub-q đầu, mẫu ngẫu nhiên:")
    for q in rnd.sample(first, min(n_show, len(first))):
        print("     ·", q[:150])
    print("  -- cặp hỏi–đáp đòi thông tin ngoài ảnh, mẫu ngẫu nhiên:")
    for q, a in rnd.sample(out, min(n_show, len(out))):
        print("     Q:", q[:150])
        print("     A:", a[:150])
    print()


if __name__ == "__main__":
    for p in (sys.argv[1:] or [d for d in DEFAULT if (REPO / d).is_file()]):
        audit((REPO / p) if not Path(p).is_absolute() else Path(p))
