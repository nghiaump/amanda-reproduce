#!/usr/bin/env python3
"""Kiểm question bank (đồ thị câu hỏi đọc phim) TRƯỚC khi đóng băng — và in thống kê.

    .venv/bin/python scripts/qbank_graph.py            # kiểm + thống kê
    .venv/bin/python scripts/qbank_graph.py --freeze   # kiểm; nếu xanh thì ghi sha256 vào data/qbank/FROZEN

Cổng 3 cần văn bản nguồn trong runs/qbank-src/ (trích từ kho tài liệu học tập tiếng Việt, truy cập riêng; không kèm
theo gói). Trong gói này, file được dùng làm thư viện: qbank_analyze.py nhập regex OUTSIDE từ đây.

Năm cổng, cổng nào đỏ thì thoát mã 1:
  1. schema + đồ thị (`QBank.validate`): id duy nhất, cạnh refines hợp lệ, không chu trình, tầng khớp loại nút,
     mỗi route đủ 3 tầng.
  2. ⛔ luật loại trừ §14: không câu tiếng Anh nào đòi thông tin NGOÀI một ảnh đơn (regex OUTSIDE dùng lại từ
     scripts/amanda_explorer_audit.py, bỏ các từ chỉ chuỗi xung vì hỏi "ảnh NÀY là T1 hay T2" là hợp lệ).
  3. ⛔ trích dẫn có thật: mỗi `vi_src` phải khớp text của đúng chunk `src` đã trích (runs/qbank-src/, do
     scripts/qbank_extract.py tạo). So bằng tỉ lệ token KHÔNG DẤU ≥ 0,6, vì OCR hỏng dấu (vd. "mơ" ↔ "mô").
     Chunk bị OCR cụt thì đối chiếu thêm bản gốc kb_ump (`*.raw.txt`).
     Nguồn mã hoá VNI (yhoc8003:20239 chunk 0) được giải mã tay ⇒ chỉ cảnh báo, không đánh đỏ.
  4. ⛔ chống rò rỉ: không câu nào trùng nguyên văn câu hỏi test SLAKE (so chuỗi đã chuẩn hoá).
  5. dạng yes/no: mọi `en_yn` phải bắt đầu bằng trợ động từ (Is/Are/Does/Do).
"""
from __future__ import annotations

import collections, hashlib, json, re, sys, unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from amanda_lab.qbank_explorer import QBANK, QBank  # noqa: E402

SRC = REPO / "runs" / "qbank-src"
VNI_OK = {"yhoc8003:20239#0"}
OUTSIDE = re.compile(r"\b(ADC|prior|previous|comparison|clinical history|patient.s (history|symptoms|age)"
                     r"|laborator|other sequences?|different sequences|lateral view|additional views?|biopsy"
                     r"|symptom|scout|age\b|sex\b|gender)", re.I)


def unaccent(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", unaccent(s))


def chunk_text(src: str) -> str | None:
    ref, _, ord_ = src.partition("#")
    f = SRC / (ref.replace(":", "_") + ".txt")
    if not f.is_file():
        return None
    parts = re.split(r"^### chunk ord=(\d+).*$", f.read_text(), flags=re.M)
    chunks = {int(parts[i]): parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    return chunks.get(int(ord_ or 0))


def raw_text(src: str) -> str | None:
    """Bản gốc kb_ump cho tài liệu mà chunk trong kho tài liệu nguồn bị OCR cụt."""
    f = SRC / (src.partition("#")[0].replace(":", "_") + ".raw.txt")
    return f.read_text() if f.is_file() else None


def cite_score(vi_src: str, text: str) -> float:
    body = re.sub(r"\((áp dụng|đậm độ →)[^)]*\)", " ", vi_src)          # chú thích của tôi, không phải trích
    want = [t for t in tokens(body) if len(t) > 1]
    have = set(tokens(text))
    return sum(t in have for t in want) / max(1, len(want))


def main() -> int:
    qb = QBank.load()
    red = 0

    errs = qb.validate()
    print(f"[1] schema + đồ thị: {'XANH' if not errs else 'ĐỎ'}")
    for e in errs:
        print("    ·", e)
    red += bool(errs)

    bad = [(q["id"], k) for q in qb.raw["questions"] for k in ("en_open", "en_yn") if OUTSIDE.search(q[k])]
    print(f"[2] câu đòi thông tin ngoài ảnh: {len(bad)} {'XANH' if not bad else 'ĐỎ'}")
    for b in bad:
        print("    ·", b)
    red += bool(bad)

    low, warn, missing = [], [], []
    for q in qb.raw["questions"]:
        srcs = [q["src"]] + list(q.get("also") or [])
        texts = [chunk_text(s_) for s_ in srcs]
        if any(t is None for t in texts):
            missing.append(q["id"])
            continue
        texts += [r for r in (raw_text(q["src"]),) if r]
        s = cite_score(q["vi_src"], "\n".join(texts))
        # ⛔ số chunk phải đúng: chunk được trích phải là chunk khớp NHẤT của tài liệu (lệch ≤ 0,1)
        ref = q["src"].partition("#")[0]
        per = {o: cite_score(q["vi_src"], chunk_text(f"{ref}#{o}") or "") for o in range(0, 400)
               if chunk_text(f"{ref}#{o}") is not None}
        best = max(per, key=per.get)
        cited = int(q["src"].partition("#")[2] or 0)
        if per[best] - per[cited] > 0.1 and not q.get("also") and q["src"] not in VNI_OK:
            low.append((q["id"], q["src"], f"ord lệch — khớp nhất là #{best} ({per[best]:.2f} > {per[cited]:.2f})"))
        elif s < 0.6:
            (warn if q["src"] in VNI_OK else low).append((q["id"], q["src"], round(s, 2)))
    print(f"[3] trích dẫn khớp chunk nguồn: {len(qb.raw['questions']) - len(low) - len(missing)}"
          f"/{len(qb.raw['questions'])} {'XANH' if not (low or missing) else 'ĐỎ'}"
          f" (cảnh báo VNI giải mã tay: {len(warn)})")
    for x in low + [(m, 'thiếu file nguồn — chạy scripts/qbank_extract.py', 0) for m in missing]:
        print("    ·", x)
    red += bool(low or missing)

    gt = json.load(open(REPO / "data/slake/test_en_amanda.json"))
    norm = lambda s: " ".join(tokens(s))                                   # noqa: E731
    test_q = {norm(r["conversations"][0]["value"].replace("<image>", "")) for r in gt}
    leak = [q["id"] for q in qb.raw["questions"] for k in ("en_open", "en_yn") if norm(q[k]) in test_q]
    print(f"[4] trùng nguyên văn câu hỏi test SLAKE: {len(leak)} {'XANH' if not leak else 'ĐỎ'} {leak}")
    red += bool(leak)

    yn = [q["id"] for q in qb.raw["questions"] if not re.match(r"^(Is|Are|Does|Do)\b", q["en_yn"])]
    print(f"[5] en_yn đúng dạng yes/no: {'XANH' if not yn else 'ĐỎ'} {yn}")
    red += bool(yn)

    # ── thống kê ──
    print("\n── thống kê ──")
    tiers = collections.Counter(q["tier"] for q in qb.raw["questions"])
    types = collections.Counter(n["type"] for n in qb.raw["nodes"])
    edges = sum(1 for n in qb.raw["nodes"] if n.get("parent"))
    print(f"nút {len(qb.nodes)} {dict(types)} · cạnh refines {edges} · câu hỏi {len(qb.questions)} {dict(sorted(tiers.items()))}")
    print(f"nguồn {len(qb.raw['sources'])} tài liệu · mục bị loại (ngoài ảnh) {len(qb.raw['excluded'])}")
    for route, meta in qb.raw["routes"].items():
        root = qb.route_root(route)
        c = [len(qb.questions_in(root, t)) for t in (1, 2, 3)]
        print(f"  {route:18} derived={meta['derived']!s:5}  tầng1 {c[0]:2}  tầng2 {c[1]:2}  tầng3 {c[2]:2}")

    if "--md" in sys.argv:
        out = REPO / "docs" / "repro" / "qbank_table.md"
        L = ["# Bảng song ngữ question bank v1 — SINH TỰ ĐỘNG",
             "",
             "⛔ Không sửa tay. Sinh bằng `.venv/bin/python scripts/qbank_graph.py --md` từ `data/qbank/qbank_v1.yaml`.",
             "Cột *trích* là nguyên văn tiếng Việt (đã đối chiếu máy với chunk nguồn, cổng [3]).", ""]
        for route, meta in qb.raw["routes"].items():
            root = qb.route_root(route)
            L += [f"## {route}" + (" — ⚠️ SUY RA (derived)" if meta["derived"] else ""), "",
                  "| tầng | id | trích tiếng Việt | EN (câu mở) | EN (yes/no) | nguồn |", "|---|---|---|---|---|---|"]
            for t in (1, 2, 3):
                for qid in qb.questions_in(root, t):
                    q = qb.questions[qid]
                    cell = lambda s: s.replace("|", "\\|")                          # noqa: E731
                    src = q["src"] + "".join(f" + {a}" for a in q.get("also") or [])
                    L.append(f"| {t} | `{qid}` | {cell(q['vi_src'])} | {cell(q['en_open'])} | {cell(q['en_yn'])} | `{src}` |")
            L.append("")
        L += ["## Mục bị loại — cần thông tin ngoài một ảnh đơn", "", "| trích | nguồn | lý do |", "|---|---|---|"]
        L += [f"| {e['vi_src']} | `{e['src']}` | {e['reason']} |" for e in qb.raw["excluded"]]
        out.write_text("\n".join(L) + "\n")
        print(f"\n→ {out.relative_to(REPO)}")

    if "--freeze" in sys.argv:
        if red:
            print("\n⛔ không đóng băng: còn cổng đỏ")
            return 1
        h = hashlib.sha256(QBANK.read_bytes()).hexdigest()
        (QBANK.parent / "FROZEN").write_text(f"{h}  {QBANK.name}\n")
        print(f"\n✅ đóng băng: sha256 {h}")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
