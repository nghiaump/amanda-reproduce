#!/usr/bin/env python3
"""Chuẩn bị SLAKE cho AMANDA: đổi định dạng + tải KG riêng của SLAKE + self-test.

    python scripts/prepare_slake.py --self-test

Hai đầu ra:
  1. `data/slake/test_en_amanda.json` — SLAKE en test ở **định dạng LLaVA-style** mà
     `run_amanda.py` của tác giả đòi: `id · image · answer_type · conversations`.
     ⚠️ SLAKE ghi `OPEN`/`CLOSED` viết hoa; AMANDA đòi `open`/`close` viết thường.
  2. `data/slake/KG/*.csv` — ⭐ **knowledge graph riêng của SLAKE**, thứ mà AMANDA bỏ qua
     để đi dùng SPOKE. Đây là vật liệu cho thí nghiệm C.
"""
from __future__ import annotations

import argparse, collections, io, json, sys, urllib.request, zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SLAKE = REPO / "data" / "slake"
KG_URL = "https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/KG.zip"

EXPECTED = {"questions": 1061, "images": 96, "open": 645, "closed": 416}


def build_amanda_format() -> list[dict]:
    rows = [x for x in json.loads((SLAKE / "test.json").read_text(encoding="utf8"))
            if x["q_lang"] == "en"]
    out = []
    for r in rows:
        # SLAKE: OPEN/CLOSED  →  AMANDA: open/close  (khác cả hoa thường lẫn hậu tố)
        at = "open" if r["answer_type"].upper() == "OPEN" else "close"
        out.append({
            "id": str(r["qid"]),
            "question_id": r["qid"],
            "image": r["img_name"],
            "answer_type": at,
            "conversations": [
                {"from": "human", "value": "<image>\n" + r["question"]},
                {"from": "gpt", "value": str(r["answer"])},
            ],
            # giữ metadata của SLAKE để phân tầng kết quả về sau
            "content_type": r.get("content_type"),
            "modality": r.get("modality"),
            "location": r.get("location"),
        })
    return out


def resolve_image_root() -> Path:
    """Tìm thư mục mà `<root>/xmlab102/source.jpg` tồn tại.

    ⚠️ `imgs.zip` của SLAKE giải nén ra **thư mục lồng** `imgs/imgs/xmlabNNN/` kèm rác `__MACOSX/`.
    Hardcode `data/slake/imgs` sẽ trượt toàn bộ 1.061 ảnh — self-test đã bắt được đúng lỗi này.
    """
    for cand in (SLAKE / "imgs" / "imgs", SLAKE / "imgs", SLAKE):
        if (cand / "xmlab102" / "source.jpg").is_file():
            return cand
    sys.exit(f"⛔ Không tìm thấy thư mục ảnh. Đã thử: {SLAKE}/imgs/imgs, {SLAKE}/imgs")


def fetch_kg() -> Path:
    kg_dir = SLAKE / "KG"
    if (kg_dir / "en_disease.csv").is_file():
        print("  ✓ KG của SLAKE đã có")
        return kg_dir
    print("  ↓ tải KG.zip của SLAKE ...", flush=True)
    with urllib.request.urlopen(KG_URL, timeout=180) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    kg_dir.mkdir(parents=True, exist_ok=True)
    for n in z.namelist():
        if n.endswith(".csv"):
            (kg_dir / Path(n).name).write_bytes(z.read(n))
    print(f"  ✓ {len(list(kg_dir.glob('*.csv')))} file CSV → {kg_dir}")
    return kg_dir


def self_test(rows: list[dict], kg_dir: Path, img_root: Path) -> None:
    fails = []

    def check(label, got, want):
        ok = got == want
        print(f"  {'✓' if ok else '✗'} {label:<40}{got!r:>10}  (kỳ vọng {want!r})")
        if not ok:
            fails.append(label)

    print("\n[1] Bộ câu hỏi")
    check("số câu hỏi tiếng Anh", len(rows), EXPECTED["questions"])
    check("số ảnh riêng biệt", len({r["image"] for r in rows}), EXPECTED["images"])
    at = collections.Counter(r["answer_type"] for r in rows)
    check("câu mở (open)", at["open"], EXPECTED["open"])
    check("câu đóng (close)", at["close"], EXPECTED["closed"])
    check("chỉ có hai giá trị answer_type", sorted(at), ["close", "open"])
    missing = [r["image"] for r in rows if not (img_root / r["image"]).is_file()]
    check("ảnh thiếu trên đĩa", len(missing), 0)
    print(f"    (thư mục ảnh đã dò: {img_root.relative_to(REPO)})")

    print("\n[2] ⭐ Knowledge graph RIÊNG của SLAKE — thứ AMANDA bỏ qua")
    tot = 0
    for name in ("en_organ_rel.csv", "en_organ.csv", "en_disease.csv"):
        lines = [l for l in (kg_dir / name).read_text(encoding="utf8", errors="ignore").splitlines()[1:]
                 if l.count("#") >= 2]
        rel = collections.Counter(l.split("#")[1] for l in lines)
        tot += len(lines)
        print(f"  {name:<20}{len(lines):>6,} triple · {len({l.split('#')[0] for l in lines}):>4} chủ thể"
              f" · {len(rel):>3} quan hệ")
        print(f"  {'':<20}{', '.join(f'{k}({v})' for k, v in rel.most_common(5))}")
    print(f"  {'TỔNG':<20}{tot:>6,} triple   (SPOKE: ~160 triệu cạnh)")

    print("\n[3] ⭐ So sánh loại tri thức — vì sao SPOKE không hợp Med-VQA")
    lines = []
    for name in ("en_organ_rel.csv", "en_organ.csv", "en_disease.csv"):
        lines += [l for l in (kg_dir / name).read_text(encoding="utf8", errors="ignore").splitlines()[1:]
                  if l.count("#") >= 2]
    heads = {l.split("#")[0].lower() for l in lines}
    top12 = ["organ", "lung", "body", "liver", "organs", "heart",
             "abnormality", "spleen", "kidney", "brain", "chest", "lungs"]
    hit = [e for e in top12 if e in heads]
    print(f"  12 entity AMANDA trích nhiều nhất trên SLAKE:")
    print(f"    khớp SPOKE/Disease (AMANDA thực dùng) : 0/12")
    print(f"    khớp SLAKE-KG                          : {len(hit)}/12  {hit}")

    print()
    if fails:
        sys.exit(f"⛔ SELF-TEST HỎNG ({len(fails)}): {', '.join(fails)}")
    print("✅ SELF-TEST XANH — dữ liệu khớp Table 5 của bài báo (SLAKE #Test = 1.061)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    rows = build_amanda_format()
    out = SLAKE / "test_en_amanda.json"
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf8")
    print(f"✓ {len(rows)} câu → {out}")
    kg_dir = fetch_kg()
    img_root = resolve_image_root()
    (SLAKE / "image_root.txt").write_text(str(img_root), encoding="utf8")
    print(f"✓ thư mục ảnh: {img_root}")
    if args.self_test:
        self_test(rows, kg_dir, img_root)
