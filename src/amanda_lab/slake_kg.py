#!/usr/bin/env python3
"""`SlakeKGClient` — cùng giao diện `SPOKEClient`, nhưng đọc KG RIÊNG của SLAKE.

    python -m amanda_lab.slake_kg --self-test          (chạy từ src/)

## Vì sao có file này

AMANDA truy hồi từ **SPOKE** — knowledge graph y sinh cấp **phân tử** (gene, protein, compound,
pathway). Trong khi đó SLAKE **tự đóng gói** một
KG 2.600 triple đúng loại tri thức mà Med-VQA cần: `location` · `symptom` · `position` · `component`.

⛔ **Không sửa một byte nào trong `third_party/AMANDA/`.** File này chỉ thay **nguồn tri thức**; còn
`SpokeRetriever`, `filter_relevant_triples`, `triples_from_spoke_items`, `ContextFilter`,
`triples_to_contexts` đều là **code gốc của tác giả, dùng lại nguyên vẹn**.

## Hợp đồng phải khớp (đọc từ `triple_templates.triples_from_spoke_items`)

Client trả về danh sách item kiểu SPOKE API. Bộ chuyển của tác giả phân biệt **node** với **cạnh**
bằng đúng một quy tắc: `neo4j_type` **có dấu `_` ⇒ cạnh**, không có ⇒ node.

    node : {"data": {"neo4j_type": "Disease", "id": …, "properties": {"name": …}}}
    cạnh : {"data": {"neo4j_type": "HAS_location_SLAKE", "source": id, "target": id,
                     "properties": {…}}}

`_edge_predicate` cắt `X_pred_Y` lấy phần giữa ⇒ `HAS_location_SLAKE` → vị từ `location`.
`_node_display_name` ghép `f"{node_type} {name}"` ⇒ chuỗi cuối là `Disease Pneumothorax location
Organ Lung.` — đúng khuôn `TRIPLE_TEXT_TEMPLATE` của tác giả.

## ⚠️ Cạm bẫy đã lường: whitelist vị từ

`DEFAULT_ALLOWED_PREDICATES` của AMANDA chỉ nhận **6** vị từ, tất cả là tên riêng schema SPOKE:
`isa · isa_did · localizes · localizes_dla · resembles · resembles_drd`.
Vị từ SLAKE (`location`, `symptom`, …) **không nằm trong đó** ⇒ để nguyên config thì lọc ra **rỗng**.

⇒ Đổi KG **bắt buộc** kéo theo đổi từ vựng vị từ. Không giấu chuyện này; chạy **hai biến thể**:
  * `strict` — ánh xạ sang whitelist gốc chỗ nào tương đương (`location` → `localizes`,
    `belong to` → `isa`), giữ nguyên config tác giả;
  * `native` — giữ vị từ gốc của SLAKE, nới whitelist đúng bằng từ vựng của SLAKE.
Và phải chạy thêm **SPOKE + whitelist nới** để biết phần tăng đến từ *đổi KG* hay từ *nới bộ lọc*.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parents[2]
KG_DIR = REPO / "data" / "slake" / "KG"

# 3 file tiếng Anh. Loại node của CHỦ THỂ suy ra từ chính file — SLAKE không ghi loại node.
FILES: tuple[tuple[str, str], ...] = (
    ("en_disease.csv", "Disease"),
    ("en_organ.csv", "Organ"),
    ("en_organ_rel.csv", "Organ"),
)

# Vị từ SLAKE ↔ whitelist gốc của AMANDA. Chỉ ánh xạ chỗ NGHĨA thật sự trùng.
# ⛔ `symptom`, `component`, `function`, `shape` không có tương đương ⇒ ở chế độ `strict` chúng BỊ BỎ,
#    và đó chính là điều cần đo: whitelist của AMANDA vứt đi bao nhiêu tri thức đúng loại.
STRICT_MAP: Mapping[str, str] = {
    "location": "localizes",
    "position": "localizes",
    "belong to": "isa",
}

# Giá trị tự do dài dòng (`description`, `prevention`, `treatment`) không phải tri thức đồ thị —
# chúng là đoạn văn. Giữ chúng lại sẽ biến "truy hồi đồ thị" thành "tra từ điển", làm hỏng so sánh.
PROSE_PREDICATES = frozenset({"description", "definition", "prevention", "treatment"})


class SlakeKGClient:
    """Đọc 3 CSV của SLAKE, trả lân cận 1-hop theo đúng định dạng item của SPOKE API."""

    def __init__(
        self,
        kg_dir: Path = KG_DIR,
        *,
        mode: str = "native",
        include_reverse: bool = True,
        drop_prose: bool = True,
        loose_match: bool = False,
    ) -> None:
        if mode not in ("native", "strict"):
            raise ValueError(f"mode phải là native|strict, nhận {mode!r}")
        self.mode = mode
        self.include_reverse = include_reverse
        self.drop_prose = drop_prose
        self.loose_match = loose_match

        self.triples: list[tuple[str, str, str, str]] = []   # (head, pred, tail, head_type)
        for fname, head_type in FILES:
            path = kg_dir / fname
            if not path.is_file():
                raise FileNotFoundError(f"⛔ thiếu {path} — chạy scripts/prepare_slake.py trước")
            for line in path.read_text(encoding="utf8", errors="ignore").splitlines()[1:]:
                parts = line.split("#")
                if len(parts) < 3:
                    continue
                head, pred, tail = parts[0].strip(), parts[1].strip(), "#".join(parts[2:]).strip()
                if not (head and pred and tail):
                    continue
                if self.drop_prose and pred.lower() in PROSE_PREDICATES:
                    continue
                self.triples.append((head, pred, tail, head_type))

        # Loại node của ĐUÔI: nếu đuôi cũng xuất hiện làm chủ thể ở đâu đó thì lấy loại đó,
        # còn không thì `Attribute` (⛔ không được trùng blocked_node_types của AMANDA).
        self._head_type: dict[str, str] = {h.lower(): t for h, _, _, t in self.triples}
        self._index: dict[str, list[int]] = collections.defaultdict(list)
        for i, (head, _, tail, _) in enumerate(self.triples):
            self._index[self._key(head)].append(i)
            if self.include_reverse:
                self._index[self._key(tail)].append(i)

    # ── khớp tên ───────────────────────────────────────────────────────────────────
    def _key(self, name: str) -> str:
        k = name.strip().lower()
        if self.loose_match and len(k) > 3 and k.endswith("s") and not k.endswith("ss"):
            k = k[:-1]                       # lungs → lung. ⚠️ đây là MỘT thay đổi nữa, phải đo riêng.
        return k

    def _node_type(self, name: str, default: str) -> str:
        return self._head_type.get(name.strip().lower(), default)

    def _predicate(self, pred: str) -> str | None:
        if self.mode == "strict":
            return STRICT_MAP.get(pred.lower())      # None ⇒ bỏ triple này
        return pred.lower()

    @property
    def native_predicates(self) -> list[str]:
        """Từ vựng vị từ để nới `allowed_predicates` ở chế độ native."""
        return sorted({p.lower() for _, p, _, _ in self.triples})

    # ── đúng giao diện SPOKEClient ────────────────────────────────────────────────
    def get_neighborhood(self, node_value: str) -> list[Mapping[str, Any]]:
        idxs = self._index.get(self._key(node_value), [])
        nodes: dict[str, Mapping[str, Any]] = {}
        edges: list[Mapping[str, Any]] = []

        for i in idxs:
            head, pred, tail, head_type = self.triples[i]
            predicate = self._predicate(pred)
            if predicate is None:
                continue
            h_id, t_id = f"h{i}", f"t{i}"
            nodes[h_id] = {"neo4j_type": self._node_type(head, head_type),
                           "id": h_id, "properties": {"name": head}}
            nodes[t_id] = {"neo4j_type": self._node_type(tail, "Attribute"),
                           "id": t_id, "properties": {"name": tail}}
            # `X_pred_Y`: `_edge_predicate` lấy phần GIỮA ⇒ đúng vị từ ta muốn.
            edges.append({"neo4j_type": f"HAS_{predicate}_SLAKE", "id": f"e{i}",
                          "source": h_id, "target": t_id,
                          "properties": {"sources": ["SLAKE-KG"]}})

        return [{"data": d} for d in list(nodes.values()) + edges]


# ── self-test ─────────────────────────────────────────────────────────────────────
def _self_test() -> None:
    sys.path.insert(0, str(REPO / "third_party" / "AMANDA" / "src"))
    from amanda.retrieval.spoke.retriever import (
        DEFAULT_ALLOWED_PREDICATES, SpokeRetrieverConfig, filter_relevant_triples)
    from amanda.retrieval.spoke.triple_templates import (
        triples_from_spoke_items, triples_to_contexts)

    fails: list[str] = []

    def check(label: str, got: Any, want: Any) -> None:
        ok = got == want
        print(f"  {'✓' if ok else '✗'} {label:<52}{str(got):>8}  (kỳ vọng {want})")
        if not ok:
            fails.append(label)

    print("\n[1] Nạp KG")
    c = SlakeKGClient()
    print(f"  {len(c.triples):,} triple (đã bỏ {sorted(PROSE_PREDICATES)}) · "
          f"{len(c._head_type):,} chủ thể · {len(c.native_predicates)} vị từ")
    print(f"  vị từ: {', '.join(c.native_predicates)}")
    check("có triple", len(c.triples) > 500, True)

    print("\n[2] ⭐ Cổng: hợp đồng item có khớp bộ chuyển CỦA TÁC GIẢ không")
    items = c.get_neighborhood("Pneumothorax")
    tri = triples_from_spoke_items(list(items))
    check("get_neighborhood('Pneumothorax') ra item", len(items) > 0, True)
    check("triples_from_spoke_items dịch được", len(tri) > 0, True)
    for t in tri[:4]:
        print(f"      {t.source} —{t.predicate}→ {t.target}")

    print("\n[3] ⭐⭐ Whitelist gốc của AMANDA vứt đi bao nhiêu")
    cfg_orig = SpokeRetrieverConfig()
    kept = filter_relevant_triples(tri, cfg_orig)
    print(f"  allowed_predicates gốc = {DEFAULT_ALLOWED_PREDICATES}")
    check("native + whitelist GỐC ⇒ còn lại", len(kept), 0)
    print("    ⇒ đúng như dự đoán: đổi KG mà không nới whitelist thì lọc ra RỖNG")

    cfg_wide = SpokeRetrieverConfig(allowed_predicates=tuple(c.native_predicates))
    kept_wide = filter_relevant_triples(tri, cfg_wide)
    check("native + whitelist NỚI ⇒ còn lại", len(kept_wide) == len(tri), True)

    cs = SlakeKGClient(mode="strict")
    tri_s = triples_from_spoke_items(list(cs.get_neighborhood("Pneumothorax")))
    kept_s = filter_relevant_triples(tri_s, cfg_orig)
    check("strict + whitelist GỐC ⇒ còn lại", len(kept_s) > 0, True)
    for t in kept_s[:3]:
        print(f"      {t.source} —{t.predicate}→ {t.target}")

    print("\n[4] ⭐ Ca thật qid 12268 — 'What part of the lung is the pneumothorax located in?'")
    print("    (SPOKE trả về: 'tuberculous pneumothorax isa_did pneumothorax' — phân loại con,")
    print("     KHÔNG nói vị trí. Xem SLAKE-KG cho gì:)")
    ctxs = triples_to_contexts(filter_relevant_triples(tri, cfg_wide), include_provenance=False)
    for s in ctxs[:6]:
        print(f"      {s}")
    check("có context nhắc tới Lung", any("lung" in s.lower() for s in ctxs), True)

    print("\n[5] Chế độ khớp tên lỏng (số nhiều)")
    check("exact : 'lungs' bắn", len(SlakeKGClient().get_neighborhood("lungs")) > 0, False)
    check("loose : 'lungs' bắn", len(SlakeKGClient(loose_match=True).get_neighborhood("lungs")) > 0, True)

    print()
    if fails:
        sys.exit(f"⛔ SELF-TEST HỎNG ({len(fails)}): {', '.join(fails)}")
    print("✅ SELF-TEST XANH — SlakeKGClient cắm vừa pipeline gốc của tác giả")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        print(json.dumps(SlakeKGClient().get_neighborhood("Pneumothorax"), indent=1)[:1200])
