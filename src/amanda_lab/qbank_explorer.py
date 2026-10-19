"""⭐ Explorer đi trên ĐỒ THỊ CÂU HỎI đọc phim (question bank tiếng Việt → tiếng Anh) — arm FQ.

Thay cho Explorer gốc của AMANDA (LLM tự sinh câu hỏi theo 3 tầng *gợi ý*). Giữ đúng khung 3 tầng của bài báo
(§3.3, Table 12): General observation → Anatomical analysis → Detailed findings, nhưng câu hỏi lấy từ
`data/qbank/qbank_v1.yaml` — trích từ checklist/bài giảng đọc phim của ĐHYD TP.HCM, ĐHYK Phạm Ngọc Thạch.

Cơ chế, mỗi lần `generate()`:
  1. **Định tuyến** — một lời gọi LLM phân loại (modality, region) từ caption + câu hỏi vào tập nhãn ĐÓNG.
     ⛔ KHÔNG đọc metadata SLAKE (modality/location/content_type): làm thế là rò đáp án cho câu hỏi Modality/Organ.
  2. **Đi trên đồ thị** — vòng 0: LLM chọn một ĐƯỜNG ĐI root → structure → finding (một câu mỗi tầng), trả id.
     Vòng ≥1: đi SÂU hơn — chọn finding/descriptor chưa thăm (ưu tiên con của các structure đã thăm).
     Đường đi phải hợp lệ trên cạnh `refines`; sai ⇒ rơi về đường đi mặc định và ĐẾM số lần rơi.
  3. Câu đóng (`answer_type == "close"`) dùng `en_yn`, câu mở dùng `en_open` — vì Perceiver tự nối
     "Answer yes or no…" vào MỌI sub-question của câu đóng (`perceiver.py:13`).
Kết quả định tuyến và id được ghi vào `reflection` (`route=…; ids=…; fallback=…`) để trace kiểm được.

⛔ Không sửa `third_party/AMANDA/`: harness ghi đè `pipe.explorer` sau khi dựng, như arm A3.
"""
from __future__ import annotations

import json, re, threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

REPO = Path(__file__).resolve().parents[2]
QBANK = REPO / "data" / "qbank" / "qbank_v1.yaml"

MODALITIES = ("xray", "ct", "mri")
REGIONS = ("chest", "abdomen_pelvis", "head", "neck")
TIER_OF_TYPE = {"root": 1, "structure": 2, "finding": 3, "descriptor": 3}


@dataclass
class QBank:
    raw: dict
    nodes: dict[str, dict] = field(default_factory=dict)
    questions: dict[str, dict] = field(default_factory=dict)
    by_node: dict[str, list[str]] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = QBANK) -> "QBank":
        raw = yaml.safe_load(path.read_text())
        qb = cls(raw)
        for n in raw["nodes"]:
            qb.nodes[n["id"]] = n
            qb.children.setdefault(n["id"], [])
        for n in raw["nodes"]:
            if n.get("parent"):
                qb.children[n["parent"]].append(n["id"])
        for q in raw["questions"]:
            qb.questions[q["id"]] = q
            qb.by_node.setdefault(q["node"], []).append(q["id"])
        return qb

    # ── đồ thị ────────────────────────────────────────────────────────────────
    def root_of(self, node_id: str) -> str:
        seen = set()
        while self.nodes[node_id].get("parent"):
            if node_id in seen:
                raise ValueError(f"chu trình tại {node_id}")
            seen.add(node_id)
            node_id = self.nodes[node_id]["parent"]
        return node_id

    def route_root(self, route: str) -> str:
        route = self.raw.get("aliases", {}).get(route, route)
        if route not in self.raw["routes"]:
            route = "generic"
        return next(n["id"] for n in self.raw["nodes"] if n["type"] == "root" and n["route"] == route)

    def subgraph(self, root: str) -> list[str]:
        out, stack = [], [root]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(reversed(self.children[n]))
        return out

    def questions_in(self, root: str, tier: int) -> list[str]:
        return [q for n in self.subgraph(root) for q in self.by_node.get(n, [])
                if self.questions[q]["tier"] == tier]

    def validate(self) -> list[str]:
        """Trả danh sách lỗi (rỗng = hợp lệ). Kiểm: id duy nhất, cha tồn tại, không chu trình,
        tầng khớp loại nút, mỗi route có ≥1 câu mỗi tầng, mỗi finding có đường về root, trường đủ."""
        errs = []
        ids = [n["id"] for n in self.raw["nodes"]]
        if len(ids) != len(set(ids)):
            errs.append("id nút trùng")
        qids = [q["id"] for q in self.raw["questions"]]
        if len(qids) != len(set(qids)):
            errs.append("id câu hỏi trùng")
        for n in self.raw["nodes"]:
            if n.get("parent") and n["parent"] not in self.nodes:
                errs.append(f"{n['id']}: cha {n['parent']} không tồn tại")
            if n["type"] not in TIER_OF_TYPE:
                errs.append(f"{n['id']}: loại nút lạ {n['type']}")
            try:
                r = self.root_of(n["id"])
                if self.nodes[r]["type"] != "root":
                    errs.append(f"{n['id']}: không về được root")
            except ValueError as e:
                errs.append(str(e))
            if n["type"] == "finding" and self.nodes.get(n.get("parent"), {}).get("type") != "structure":
                errs.append(f"{n['id']}: finding phải là con của structure (refines)")
        for q in self.raw["questions"]:
            for k in ("id", "node", "tier", "vi_src", "src", "en_open", "en_yn"):
                if not q.get(k):
                    errs.append(f"{q.get('id')}: thiếu {k}")
            node = self.nodes.get(q["node"])
            if node is None:
                errs.append(f"{q['id']}: nút {q['node']} không tồn tại")
            elif TIER_OF_TYPE[node["type"]] != q["tier"]:
                errs.append(f"{q['id']}: tier {q['tier']} ≠ tầng của nút {node['type']}")
            if q.get("src", "").split("#")[0] not in self.raw["sources"]:
                errs.append(f"{q['id']}: nguồn {q.get('src')} chưa khai trong `sources`")
        for route in self.raw["routes"]:
            root = self.route_root(route)
            for t in (1, 2, 3):
                if not self.questions_in(root, t):
                    errs.append(f"route {route}: không có câu tầng {t}")
        return errs


# ── LLM ───────────────────────────────────────────────────────────────────────
# v1.1 (24/09, sau pilot): v1 cho phép "unknown" ⇒ caption hiếm khi nêu phương thức (29/96 ảnh) nên 50/96 ảnh
# rơi về generic. v1.1 buộc ĐOÁN phương thức theo manh mối ⇒ đo offline trên 96 caption: phương thức đúng
# 46 → 78, vùng 85 → 85, rơi generic 50 → 3. Vẫn ⛔ không đọc metadata SLAKE.
ROUTER_SYSTEM = (
    "You classify a medical image from its text description and the question asked about it. Reply with JSON only: "
    '{"modality": one of ["xray","ct","mri"], "region": one of ["chest","abdomen_pelvis","head","neck","unknown"]}. '
    "ALWAYS pick the most likely modality — never leave it empty. Cues: radiograph, X-ray, PA/AP view, film -> xray; "
    "slice, axial/cross-sectional, attenuation, density, Hounsfield, contrast phase, window -> ct; "
    "T1, T2, FLAIR, DWI, ADC, signal intensity, hyperintense/hypointense -> mri. "
    "If there is no explicit cue, choose the modality most commonly used for that body region and that kind of question. "
    "Use \"unknown\" for region only if the description gives no anatomical clue."
)

PATH_SYSTEM = (
    "You help a radiologist read an image systematically, from coarse to fine, like a clinical reading checklist. "
    "You are given a question bank organised as a graph. Choose question IDs ONLY from the lists given. "
    "Reply with JSON only."
)


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


class QuestionBankExplorer:
    """Cùng giao diện với `amanda.agents.explorer.Explorer.generate` ⇒ thay được bằng `pipe.explorer = …`."""

    def __init__(self, llm, *, max_sub_questions: int = 3, bank: QBank | None = None) -> None:
        self.llm = llm
        self.max_sub_questions = max_sub_questions
        self.bank = bank or QBank.load()
        errs = self.bank.validate()
        if errs:
            raise ValueError("question bank không hợp lệ:\n  " + "\n  ".join(errs))
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "route_unknown": 0, "fallback": 0, "repaired": 0}
        # Trạng thái theo từng mẫu. v1.1: khoá thêm theo LUỒNG và reset ở vòng 0 — v1 khoá bằng (caption, question)
        # nên hai ảnh khác nhau có caption giống hệt dùng chung `visited` ⇒ đường đi hợp lệ bị đánh rơi oan.
        # Harness chạy trọn một mẫu trong một luồng (ThreadPoolExecutor.map) ⇒ khoá này không va nhau.
        self._state: dict[tuple[str, str], dict[str, Any]] = {}

    # -- định tuyến ----------------------------------------------------------------
    def _route(self, caption: str, question: str, answer_type) -> tuple[str, str]:
        out = _json(self.llm.chat([
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": f"Image description: {caption}\nQuestion: {question}"},
        ], answer_type=answer_type))
        mod = out.get("modality") if out.get("modality") in MODALITIES else None
        reg = out.get("region") if out.get("region") in REGIONS else None
        route = f"{mod}_{reg}" if mod and reg else "generic"
        return route, self.bank.route_root(route)

    # -- chọn đường đi ------------------------------------------------------------
    def _menu(self, ids: Sequence[str], answer_type) -> str:
        key = "en_yn" if answer_type == "close" else "en_open"
        return "\n".join(f"  {i}: {self.bank.questions[i][key]}" for i in ids)

    def _tree(self, root: str, answer_type) -> str:
        """Tầng 2 kèm các tầng 3 là con của nó (cạnh `refines`), rồi các câu mô tả tổn thương của route."""
        key = "en_yn" if answer_type == "close" else "en_open"
        lines = []
        for s in self.bank.children[root]:
            if self.bank.nodes[s]["type"] != "structure":
                continue
            for q2 in self.bank.by_node.get(s, []):
                lines.append(f"  {q2}: {self.bank.questions[q2][key]}")
            for f in self.bank.children[s]:
                for q3 in self.bank.by_node.get(f, []):
                    lines.append(f"      {q3}: {self.bank.questions[q3][key]}")
        desc = [q for d in self.bank.children[root] if self.bank.nodes[d]["type"] == "descriptor"
                for q in self.bank.by_node.get(d, [])]
        if desc:
            lines.append("  Lesion description (allowed as level 3 after any level-2 question):")
            lines += [f"      {q}: {self.bank.questions[q][key]}" for q in desc]
        return "\n".join(lines)

    def _repair(self, ids: list[str], root: str, visited: set[str]) -> list[str] | None:
        """v1.1: giữ ý định của LLM thay vì rơi về đường đi mặc định. Lấy câu tầng 1 và câu tầng 3 LLM đã chọn,
        rồi thay câu tầng 2 bằng câu gắn vào CHA THẬT của nút tầng 3 (cạnh refines). Không sửa được ⇒ None."""
        qs = self.bank.questions
        ids = [i for i in ids if i in qs and i not in visited and self.bank.root_of(qs[i]["node"]) == root]
        t1 = next((i for i in ids if qs[i]["tier"] == 1), None)
        t3 = next((i for i in reversed(ids) if qs[i]["tier"] == 3), None)
        if not (t1 and t3):
            return None
        f = self.bank.nodes[qs[t3]["node"]]
        if f["type"] == "descriptor":
            t2 = next((i for i in ids if qs[i]["tier"] == 2), None)
        else:
            t2 = next((i for i in self.bank.by_node.get(f["parent"], []) if i not in visited), None)
        path = [t1, t2, t3] if t2 else None
        return path if path and self._valid_path(path, root, True) else None

    def _default_path(self, root: str, visited: set[str]) -> list[str]:
        t1 = [q for q in self.bank.questions_in(root, 1) if q not in visited]
        t2 = [q for q in self.bank.questions_in(root, 2) if q not in visited]
        path = t1[:1] + t2[:1]
        if t2:
            s_node = self.bank.questions[t2[0]]["node"]
            t3 = [q for c in self.bank.children[s_node] for q in self.bank.by_node.get(c, [])
                  if q not in visited]
            path += t3[:1]
        return path

    def _valid_path(self, ids: list[str], root: str, first_round: bool) -> bool:
        qs = self.bank.questions
        if not ids or any(i not in qs for i in ids):
            return False
        if any(self.bank.root_of(qs[i]["node"]) != root for i in ids):
            return False
        if first_round:
            if [qs[i]["tier"] for i in ids] != [1, 2, 3]:
                return False
            s_node, f_node = qs[ids[1]]["node"], qs[ids[2]]["node"]
            f = self.bank.nodes[f_node]
            # tầng 3 phải refines từ đúng structure đã chọn, hoặc là descriptor của route
            return f.get("parent") == s_node or f["type"] == "descriptor"
        return True

    def generate(self, *, caption: str, question: str, history: str, iteration: int,
                 answer_type) -> tuple[str, list[str]]:
        key = (threading.get_ident(), caption, question)
        with self._lock:
            self.stats["calls"] += 1
            if iteration == 0:
                self._state[key] = {"visited": set()}
            st = self._state.setdefault(key, {"visited": set()})
        if iteration == 0 or "root" not in st:
            route, root = self._route(caption, question, answer_type)
            st.update(route=route, root=root)
            if route == "generic":
                with self._lock:
                    self.stats["route_unknown"] += 1
        root, visited = st["root"], st["visited"]

        if iteration == 0:
            prompt = (
                f"Image description: {caption}\nMain question: {question}\n\n"
                "Pick a coarse-to-fine reading path that helps answer the main question: ONE level-1 question, then ONE "
                "level-2 question, then ONE level-3 question listed UNDER that level-2 question (or a lesion "
                "description).\n"
                f"Level 1 — general observation:\n{self._menu(self.bank.questions_in(root, 1), answer_type)}\n"
                f"Level 2 — anatomical analysis, each followed by its level-3 detailed findings:\n"
                f"{self._tree(root, answer_type)}\n"
                'Reply: {"path": ["<level-1 id>", "<level-2 id>", "<level-3 id>"]}'
            )
        else:
            done_structs = {self.bank.questions[q]["node"] for q in visited
                            if self.bank.questions[q]["tier"] == 2}
            cand = [q for q in self.bank.questions_in(root, 3) if q not in visited]
            near = [q for q in cand if self.bank.nodes[self.bank.questions[q]["node"]].get("parent") in done_structs
                    or self.bank.nodes[self.bank.questions[q]["node"]]["type"] == "descriptor"]
            cand = near or cand or [q for q in self.bank.questions_in(root, 2) if q not in visited]
            prompt = (
                f"Image description: {caption}\nMain question: {question}\nPrevious Q&A history: {history}\n\n"
                f"Go deeper: pick up to {self.max_sub_questions} NEW detailed questions that best resolve the remaining "
                f"uncertainty.\n{self._menu(cand, answer_type)}\n"
                'Reply: {"path": ["<id>", ...]}'
            )

        out = _json(self.llm.chat([{"role": "system", "content": PATH_SYSTEM},
                                   {"role": "user", "content": prompt}], answer_type=answer_type))
        ids = [i for i in (out.get("path") or []) if isinstance(i, str)][: self.max_sub_questions]
        ok = self._valid_path(ids, root, iteration == 0) and not (set(ids) & visited)
        if not ok and iteration == 0:
            fixed = self._repair(ids, root, visited)
            if fixed:
                ids, ok = fixed, True
                with self._lock:
                    self.stats["repaired"] += 1
        if not ok:
            with self._lock:
                self.stats["fallback"] += 1
            ids = (self._default_path(root, visited) if iteration == 0 else
                   [q for q in self.bank.questions_in(root, 3) if q not in visited][: self.max_sub_questions])
        visited.update(ids)
        field = "en_yn" if answer_type == "close" else "en_open"
        questions = [self.bank.questions[i][field] for i in ids]
        reflection = f"route={st['route']}; ids={','.join(ids)}; fallback={int(not ok)}"
        return reflection, questions
