#!/usr/bin/env python3
"""Chạy khô arm FQ — pipeline THẬT của tác giả + QuestionBankExplorer, LLM và LLaVA-Med GIẢ. Không cần GPU.

    .venv/bin/python scripts/qbank_dryrun.py

Kiểm (in PASS/FAIL, thoát mã 1 nếu có FAIL):
  1. chế độ ép: MỌI câu có ≥1 lượt Explorer (ở cấu hình gốc chỉ ~17%)
  2. vòng 0 luôn là đường đi 3 câu, đúng thứ tự tầng 1 → 2 → 3, tầng 3 là con (refines) của tầng 2 hoặc descriptor
  3. mỗi lượt ≤ max_sub_questions câu, không lặp câu đã hỏi trong cùng một mẫu
  4. câu đóng dùng `en_yn`, câu mở dùng `en_open`
  5. đường đi SAI do LLM trả (cố ý bơm ~30%) bị bắt và rơi về mặc định — số lần rơi khớp số lần bơm
  6. reflection ghi `route=…; ids=…; fallback=…`
  7. ⛔ `qbank_explorer.py` không đọc metadata SLAKE (modality / location / content_type)
"""
from __future__ import annotations

import json, random, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "AMANDA" / "src"))
sys.path.insert(0, str(REPO / "src"))

from amanda.pipeline import PipelineSettings, QuestionSample                     # noqa: E402
from amanda_lab.qbank_explorer import PATH_SYSTEM, ROUTER_SYSTEM, QuestionBankExplorer  # noqa: E402
from amanda_lab.run_pipeline import ForcedExplorePipeline, NullRetriever            # noqa: E402

rng = random.Random(0)
INJECT = 0.3


class StubLLM:
    """Router: đoán theo từ khoá caption. Chọn đường: hợp lệ, hoặc cố ý SAI với xác suất INJECT.
    Reasoner: trả 'Answer: yes'. Evaluator: điểm 2 ⇒ đi tiếp một vòng nữa sau vòng ép."""
    def __init__(self) -> None:
        self.injected = 0

    def chat(self, messages, *, answer_type):
        sys_, user = messages[0]["content"], messages[-1]["content"]
        if sys_ == ROUTER_SYSTEM:
            t = user.lower()
            mod = "mri" if "mri" in t or "t2" in t else "ct" if re.search(r"\bct\b|computed", t) else "xray"
            reg = ("head" if re.search(r"brain|cerebr|skull", t) else "abdomen_pelvis"
                   if re.search(r"abdom|liver|kidney|pelvi", t) else "chest")
            return json.dumps({"modality": mod, "region": reg})
        if sys_ == PATH_SYSTEM:
            ids = re.findall(r"^\s+([a-z]{2}\.[a-z]\.[\w.]+):", user, re.M)
            if rng.random() < INJECT:
                self.injected += 1
                return json.dumps({"path": ["xx.bogus", ids[0] if ids else "yy"]})
            if "Level 1" in user:
                t1 = [i for i in ids if ".t." in i]
                t2 = [i for i in ids if ".s." in i]
                s = rng.choice(t2)
                # con của structure s = các dòng thụt sâu ngay sau s
                block = user.split(f"  {s}:")[1].split("\n  ")[0] if f"  {s}:" in user else ""
                kids = re.findall(r"^\s{6}([a-z]{2}\.[fd]\.[\w.]+):", block, re.M)
                desc = [i for i in ids if ".d." in i]
                return json.dumps({"path": [t1[0], s, (kids or desc)[0]]})
            return json.dumps({"path": ids[:2]})
        if "Score" in sys_ or "score" in sys_.lower():
            return "Score: 2\nExplanation: stub"
        return "Analysis: stub\nAnswer: yes"


class StubMLLM:
    def __init__(self, caps):
        self.caps, self.key = caps, None

    def caption(self, image, prompt=None):
        return self.caps[image]

    def answer(self, image, question, answer_type):
        return "Yes." if answer_type == "close" else "It appears normal."


def main() -> int:
    gt = json.loads((REPO / "data/slake/test_en_amanda.json").read_text())
    caps = json.loads((REPO / "runs/amanda-A/captions_patched.json").read_text())
    by = {}
    for g in gt:
        by.setdefault((g["modality"], g["answer_type"]), []).append(g)
    samples = [g for k in sorted(by) for g in rng.sample(by[k], min(4, len(by[k])))][:20]

    llm = StubLLM()
    exp = QuestionBankExplorer(llm, max_sub_questions=3)
    pipe = ForcedExplorePipeline(mllm=StubMLLM(caps), llm=llm, retriever=NullRetriever(), dataset_name="slake",
                                 settings=PipelineSettings(max_iterations=3, confidence_threshold=3,
                                                           max_sub_questions=3, fewshot_enabled=False))
    pipe.explorer = exp
    fails = []
    n_forced = 0
    for g in samples:
        q = g["conversations"][0]["value"].replace("<image>\n", "")
        s = QuestionSample(question_id=str(g["question_id"]), question=q, prompt=q, image=g["image"],
                           answer=g["conversations"][1]["value"], answer_type=g["answer_type"])
        res = pipe.run(s, image=g["image"])
        h = res.history
        refl = [x for x in h if x.startswith("Reflection:")]
        sq = [re.sub(r"^Sub-Question \d+:\s*", "", x) for x in h if x.startswith("Sub-Question")]
        n_forced += bool(sq)
        field = "en_yn" if g["answer_type"] == "close" else "en_open"
        allowed = {qq[field] for qq in exp.bank.questions.values()}
        if any(x not in allowed for x in sq):
            fails.append((g["question_id"], "4: dạng câu sai"))
        if len(sq) != len(set(sq)):
            fails.append((g["question_id"], "3: lặp câu"))
        if not refl or not refl[-1].startswith("Reflection: route="):
            fails.append((g["question_id"], "6: history thiếu dòng Reflection: route=…"))
    # kiểm vòng 0 bằng cách gọi thẳng generate trên mẫu mới (không bơm lỗi)
    global INJECT
    INJECT = 0.0
    for g in samples:
        q = g["conversations"][0]["value"].replace("<image>\n", "")
        e2 = QuestionBankExplorer(StubLLM(), max_sub_questions=3)
        refl, qs = e2.generate(caption=caps[g["image"]], question=q, history="", iteration=0,
                               answer_type=g["answer_type"])
        ids = re.search(r"ids=([^;]+)", refl).group(1).split(",")
        tiers = [e2.bank.questions[i]["tier"] for i in ids]
        f = e2.bank.nodes[e2.bank.questions[ids[2]]["node"]]
        ok_ref = f.get("parent") == e2.bank.questions[ids[1]]["node"] or f["type"] == "descriptor"
        if tiers != [1, 2, 3] or not ok_ref or len(qs) != 3:
            fails.append((g["question_id"], f"2: đường đi vòng 0 sai {ids}"))
        if not re.match(r"route=\w+; ids=.+; fallback=[01]$", refl):
            fails.append((g["question_id"], f"6: reflection sai {refl!r}"))

    # [7] (a) chữ ký generate() = đúng 5 tham số của Explorer gốc ⇒ về cấu trúc KHÔNG nhận được bản ghi dataset;
    #     (b) code không nhắc location/content_type; (c) "modality" chỉ đọc từ `out` = JSON router trả về.
    import inspect
    from amanda.agents.explorer import Explorer
    sig_ok = (list(inspect.signature(QuestionBankExplorer.generate).parameters)
              == list(inspect.signature(Explorer.generate).parameters))
    import ast
    tree = ast.parse((REPO / "src/amanda_lab/qbank_explorer.py").read_text())
    for node in ast.walk(tree):                       # bỏ docstring (module/class/hàm) — chỉ quét CODE
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:] or [ast.Pass()]
    code = ast.unparse(tree)                          # unparse cũng bỏ comment
    leak = re.findall(r"location|content_type", code)
    leak += [m for m in re.findall(r"(\w+)\.get\([\"']modality", code) if m != "out"]
    leak += [] if sig_ok else ["chữ ký generate() khác Explorer gốc"]

    print(f"[1] câu có ≥1 lượt Explorer (chế độ ép): {n_forced}/{len(samples)}",
          "PASS" if n_forced == len(samples) else "FAIL")
    print(f"[2–4,6] lỗi đường đi/dạng câu/reflection: {len(fails)}", "PASS" if not fails else "FAIL")
    for x in fails[:10]:
        print("    ·", x)
    print(f"[5] đường đi sai được bơm {llm.injected} · bị bắt và rơi về mặc định {exp.stats['fallback']}",
          "PASS" if llm.injected == exp.stats["fallback"] and llm.injected > 0 else "FAIL")
    print(f"[7] qbank_explorer.py đọc metadata SLAKE: {len(leak)}", "PASS" if not leak else "FAIL")
    print(f"    thống kê explorer: {exp.stats}")
    ok = n_forced == len(samples) and not fails and llm.injected == exp.stats["fallback"] > 0 and not leak
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
