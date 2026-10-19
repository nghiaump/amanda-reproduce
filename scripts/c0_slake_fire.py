#!/usr/bin/env python3
"""C0 — tỉ lệ bắn của Retriever khi đổi SPOKE → SLAKE-KG. Thuần CPU, không gọi LLM, không gọi mạng.

    .venv/bin/python scripts/c0_slake_fire.py

⭐ **Một biến đổi duy nhất.** Dùng lại **đúng bộ entity** LLM đã trích ở A1
(`runs/amanda-A/A1.json`) thay vì trích lại ⇒ khác biệt đo được **chỉ** đến từ nguồn tri thức.
`filter_relevant_triples`, `triples_from_spoke_items`, `triples_to_contexts`, `ContextFilter`
đều là code GỐC của tác giả. `ContextFilter` là cosine token thuần ⇒ chạy offline được.

Nhánh cuối là **đối chứng chống nhiễu**: nếu SPOKE cũng vọt lên khi nới whitelist thì phần tăng
không đến từ việc đổi đồ thị, và phải báo cáo đúng như vậy.
"""
from __future__ import annotations

import json, os, sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "AMANDA" / "src"))
sys.path.insert(0, str(REPO / "src"))

from amanda.retrieval.spoke.retriever import (                    # noqa: E402
    SpokeRetrieverConfig, build_retrieval_query, filter_relevant_triples)
from amanda.retrieval.spoke.context_filter import ContextFilter   # noqa: E402
from amanda.retrieval.spoke.triple_templates import (             # noqa: E402
    triples_from_spoke_items, triples_to_contexts)
from amanda_lab.slake_kg import SlakeKGClient                     # noqa: E402

# AMANDA_A_DIR: thư mục chứa A1.json, file caption, spoke_cache (mặc định runs/amanda-A)
# AMANDA_CAPTIONS: tên file caption trong thư mục đó (mặc định captions_patched.json, do bước 06 sinh)
ADIR = Path(os.environ.get("AMANDA_A_DIR", REPO / "runs/amanda-A"))
A1 = json.loads((ADIR / "A1.json").read_text(encoding="utf8"))
GT = {str(g["question_id"]): g
      for g in json.loads((REPO / "data/slake/test_en_amanda.json").read_text(encoding="utf8"))}
CAPS = json.loads((ADIR / os.environ.get("AMANDA_CAPTIONS", "captions_patched.json")).read_text(encoding="utf8"))
SPOKE_CACHE = ADIR / "spoke_cache"
OUT = REPO / "runs/amanda-C"


class CachedSpokeReplay:
    """Phát lại cache SPOKE đã tải ở B0 — cùng quy tắc đặt tên file như `amanda_retriever_probe.py`.

    ⛔ Không gọi mạng. Entity chưa có trong cache ⇒ đếm vào `misses` và coi như rỗng; con số này
    được in ra để không âm thầm báo cáo thiếu.
    """
    def __init__(self) -> None:
        self.misses = 0

    def get_neighborhood(self, node_value: str):
        key = "".join(c if c.isalnum() else "_" for c in node_value)[:120] or "_empty"
        p = SPOKE_CACHE / f"{key}.json"
        if p.is_file():
            return json.loads(p.read_text())
        self.misses += 1
        return []


def run(name: str, client, cfg: SpokeRetrieverConfig, *, use_caption: bool = True) -> dict:
    cf = ContextFilter(cfg.filter_config)
    rows, fired, ctx_lens = [], 0, []
    for r in A1["results"]:
        ents = r["entities"][: cfg.max_entities]
        g = GT[str(r["qid"])]
        question = g["conversations"][0]["value"].replace("<image>\n", "")
        query = build_retrieval_query(question, caption=CAPS.get(g["image"]) if use_caption else None)
        contexts: list[str] = []
        matched: list[str] = []
        if ents:
            per = max(1, cfg.context_volume // max(1, len(ents)))
            for e in ents:
                tri = filter_relevant_triples(
                    triples_from_spoke_items(list(client.get_neighborhood(e))), cfg)
                cs = triples_to_contexts(tri, include_provenance=cfg.include_provenance)
                if cs:
                    matched.append(e)
                contexts.extend(cs[:per])
        keep = cf.filter(query, contexts) if contexts else []
        ctx = " ".join(k if k.endswith(".") else k + "." for k in keep) if keep else None
        if ctx:
            fired += 1
            ctx_lens.append(len(ctx.split()))
        rows.append({"qid": r["qid"], "content_type": r["content_type"],
                     "answer_type": r["answer_type"], "entities": ents,
                     "matched": matched, "context": ctx})
    n = len(rows)
    return {"arm": name, "use_caption": use_caption, "n": n, "fired": fired, "fire_rate": round(100 * fired / n, 2),
            "mean_ctx_words": round(sum(ctx_lens) / len(ctx_lens), 1) if ctx_lens else 0.0,
            "results": rows}


if __name__ == "__main__":
    native = SlakeKGClient()
    wide = tuple(native.native_predicates)

    ARMS = [
        # (nhãn,                      client,                              config)
        ("SPOKE · whitelist gốc",     CachedSpokeReplay(),                 SpokeRetrieverConfig()),
        ("SPOKE · whitelist nới",     CachedSpokeReplay(),                 SpokeRetrieverConfig(allowed_predicates=())),
        ("SLAKE · strict · f.gốc",    SlakeKGClient(mode="strict"),        SpokeRetrieverConfig()),
        ("SLAKE · native · exact",    native,                              SpokeRetrieverConfig(allowed_predicates=wide)),
        ("SLAKE · native · loose",    SlakeKGClient(loose_match=True),     SpokeRetrieverConfig(allowed_predicates=wide)),
    ]

    print("C0 — tỉ lệ Retriever bắn trên n=1.061 câu SLAKE, dùng lại đúng bộ entity của A1")
    print("⛔ Mọi nhánh dùng NGUYÊN code lọc/dựng-context của tác giả; chỉ đổi nguồn tri thức.\n")
    print(f"{'nhánh':<24}{'KHÔNG caption':>18}{'CÓ caption':>18}{'từ/ctx':>9}")
    print("─" * 69)
    summary = []
    for label, client, cfg in ARMS:
        a = run(label, client, cfg, use_caption=False)
        b = run(label, client, cfg, use_caption=True)
        print(f"{label:<24}{a['fired']:>7} ({a['fire_rate']:5.2f}%){b['fired']:>7} ({b['fire_rate']:5.2f}%)"
              f"{b['mean_ctx_words']:>9}")
        key = label.replace(" ", "").replace("·", "-")
        (OUT / f"{key}.json").write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf8")
        summary.append({"arm": label, "fired_nocap": a["fired"], "rate_nocap": a["fire_rate"],
                        "fired_cap": b["fired"], "rate_cap": b["fire_rate"],
                        "mean_ctx_words": b["mean_ctx_words"]})
    (OUT / "C0_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf8")

    print("\n⭐ Đối chứng chống nhiễu: SPOKE nới whitelist KHÔNG khá lên ⇒ phần tăng đến từ ĐỔI ĐỒ THỊ,")
    print("   không từ nới bộ lọc. (Nới còn hơi hại: thêm triple nhiễu làm ContextFilter đội ngưỡng.)")

    best = json.loads((OUT / "SLAKE-native-loose.json").read_text())
    print(f"\n⭐ Bắn theo content_type (SLAKE·native·loose, có caption) — SPOKE bắn 0/253 nhóm Organ:")
    tot = Counter(r["content_type"] for r in best["results"])
    hit = Counter(r["content_type"] for r in best["results"] if r["context"])
    for ct, t in tot.most_common():
        bar = "█" * round(20 * hit[ct] / t)
        print(f"    {str(ct):<14}{hit[ct]:>5}/{t:<5} = {100*hit[ct]/t:5.1f}%  {bar}")
    print("    ⚠️ Modality và Plane bắn 0% là ĐÚNG: SLAKE-KG không chứa tri thức về")
    print("       phương thức chụp (CT/MRI/X-quang) hay mặt cắt — hai nhóm này phải trả lời từ ẢNH.")
