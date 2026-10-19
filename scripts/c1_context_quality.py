#!/usr/bin/env python3
"""C1 — context truy hồi được có DÙNG ĐƯỢC không? Thuần CPU.

    .venv/bin/python scripts/c1_context_quality.py

⚠️ Tỉ lệ bắn cao **không phải** thắng lợi. Câu hỏi thật là: trong các ca bắn được, context có
chứa thông tin trả lời được câu hỏi không.

Thước dùng ở đây: **đáp án gold có xuất hiện trong context không** (khớp theo token, đã chuẩn hoá).
🟨 Đây là **cận trên** của phần đóng góp mà truy hồi có thể mang lại — context không chứa đáp án thì
mô hình chỉ có thể đoán đúng nhờ ảnh, không nhờ đồ thị. Không phải thước hoàn hảo (context có thể
hữu ích gián tiếp), nên có kèm mẫu để đọc tay.
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "runs/amanda-C"
GT = {str(g["question_id"]): g
      for g in json.loads((REPO / "data/slake/test_en_amanda.json").read_text(encoding="utf8"))}
STOP = {"the", "a", "an", "of", "in", "on", "is", "are", "and", "or", "to"}


def toks(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP}


def score(path: Path) -> dict:
    """⚠️ Hai thước, và thước thứ hai mới là thước thật.

    `full`      — context chứa đủ token của đáp án gold. **Thổi phồng nặng.**
    `informative` — context chứa đáp án VÀ đáp án KHÔNG có sẵn trong câu hỏi.

    Vì sao phải có thước thứ hai: câu *"Which organ is abnormal, heart or lung?"* có gold *"Heart"*,
    mà chữ `heart` vốn đã nằm trong câu hỏi. Context nhắc lại `Organ Heart four chambers …` sẽ được
    tính là "chứa đáp án" dù nó **không nói gì** về chuyện bất thường. Đếm kiểu đó là tự lừa mình.
    """
    d = json.loads(path.read_text(encoding="utf8"))
    fired = [r for r in d["results"] if r["context"]]
    hit = part = info = 0
    for r in fired:
        g = GT[str(r["qid"])]
        gold = toks(g["conversations"][1]["value"])
        ctx = toks(r["context"])
        qtoks = toks(g["conversations"][0]["value"])
        if gold and gold <= ctx:
            hit += 1
            if not (gold <= qtoks):        # đáp án CHƯA có sẵn trong câu hỏi
                info += 1
        elif gold & ctx:
            part += 1
    n = len(fired)
    return {"arm": d["arm"], "fired": n,
            "full": hit, "full_pct": round(100 * hit / n, 1) if n else 0.0,
            "informative": info, "informative_pct": round(100 * info / n, 1) if n else 0.0,
            "partial": part, "partial_pct": round(100 * part / n, 1) if n else 0.0,
            "none": n - hit - part}


if __name__ == "__main__":
    files = [("SPOKE · whitelist gốc", "SPOKE-whitelistgốc.json"),
             ("SLAKE · strict", "SLAKE-strict-f.gốc.json"),
             ("SLAKE · native · loose", "SLAKE-native-loose.json")]
    print("C1 — trong các ca Retriever BẮN, context có chứa đáp án gold không?\n")
    print(f"{'nhánh':<24}{'bắn':>6}{'chứa ĐỦ':>13}{'⭐ CÓ THÔNG TIN':>19}{'một phần':>13}")
    print("  (⭐ = chứa đáp án VÀ đáp án không có sẵn trong câu hỏi — thước thật)")
    print("─" * 76)
    rows = []
    for label, fname in files:
        p = OUT / fname
        if not p.is_file():
            print(f"  ⚠️ thiếu {fname}"); continue
        s = score(p)
        rows.append(s)
        print(f"{label:<24}{s['fired']:>6}{s['full']:>7} ({s['full_pct']:4.1f}%)"
              f"{s['informative']:>11} ({s['informative_pct']:4.1f}%)"
              f"{s['partial']:>7} ({s['partial_pct']:4.1f}%)")
    (OUT / "C1_summary.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf8")

    print("\n⭐ Mẫu đọc TAY — ca CÓ THÔNG TIN THẬT (đáp án nằm trong context, KHÔNG nằm sẵn trong câu hỏi):")
    d = json.loads((OUT / "SLAKE-native-loose.json").read_text(encoding="utf8"))
    shown = 0
    for r in d["results"]:
        if not r["context"]:
            continue
        g = GT[str(r["qid"])]
        gold = g["conversations"][1]["value"]
        qt = toks(g["conversations"][0]["value"])
        if not (toks(gold) and toks(gold) <= toks(r["context"]) and not toks(gold) <= qt):
            continue
        print(f"\n  qid {r['qid']} [{r['content_type']}] {g['conversations'][0]['value'].replace(chr(10),' ')[:70]}")
        print(f"    gold    : {gold}")
        print(f"    entity  : {r['matched']}")
        print(f"    context : {r['context'][:190]}")
        shown += 1
        if shown >= 4:
            break

    print("\n⚠️ Mẫu đọc TAY — ca bắn nhưng context KHÔNG chứa đáp án (nhiễu, phải thừa nhận):")
    shown = 0
    for r in d["results"]:
        if not r["context"]:
            continue
        g = GT[str(r["qid"])]
        gold = g["conversations"][1]["value"]
        if toks(gold) & toks(r["context"]):
            continue
        print(f"\n  qid {r['qid']} [{r['content_type']}] {g['conversations'][0]['value'].replace(chr(10),' ')[:70]}")
        print(f"    gold    : {gold}")
        print(f"    context : {r['context'][:190]}")
        shown += 1
        if shown >= 3:
            break
