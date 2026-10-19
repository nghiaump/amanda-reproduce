#!/usr/bin/env python3
"""Phát lại mọi lời gọi Retriever trong các lượt A1 của bước 10, với bốn cách lấy tri thức (cần server LLM).

    PYTHONPATH=src .venv/bin/python scripts/retriever_investigation.py
    A1_GLOB='full-r*' PYTHONPATH=src .venv/bin/python scripts/retriever_investigation.py

Dựng lại lời gọi `SpokeRetriever.retrieve(question, caption, subquestions, history)` cho mọi câu đã vào vòng bổ sung
trong các lượt `runs/amanda-A1/<A1_GLOB>`, dùng EntityExtractor + ContextFilter của tác giả và server LLM ở
AMANDA_LLM_URL / AMANDA_LLM_MODEL, rồi ghi bốn trường hợp cho mỗi lời gọi:

  trace      : Retriever trong lượt chạy thật có trả tri thức không (rag_context)
  cache      : chỉ đọc runs/amanda-A/spoke_cache (bước 07), entity không có trong cache ⇒ rỗng (không gọi mạng)
  live       : gọi SPOKE API cho entity thiếu (lưu vào runs/amanda-A/spoke_cache_live)
  live_short : như live nhưng query của bộ lọc chỉ gồm câu hỏi + sub-question + caption (bỏ lịch sử)

Lịch sử được dựng lại từ trace.jsonl. Đầu ra: results/retriever_investigation.json.
"""
from __future__ import annotations

import glob, json, os, re, sys, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "third_party/AMANDA/src", REPO / "src", REPO / "scripts"):
    sys.path.insert(0, str(p))

from amanda.llm.openai_chat import OpenAIChatLLM                                   # noqa: E402
from amanda.retrieval.spoke.entity_extractor import EntityExtractor                # noqa: E402
from amanda.retrieval.spoke.spoke_client import SPOKEClient, SpokeAPIConfig        # noqa: E402
from amanda.retrieval.spoke.context_filter import ContextFilter                    # noqa: E402
from amanda.retrieval.spoke.retriever import (                                     # noqa: E402
    SpokeRetrieverConfig, build_retrieval_query, filter_relevant_triples, triples_from_spoke_items, triples_to_contexts)

R = REPO / "runs"
OLD = R / "amanda-A/spoke_cache"
LIVE = R / "amanda-A/spoke_cache_live"
LIVE.mkdir(parents=True, exist_ok=True)
GT = {str(g["question_id"]): g for g in json.loads((REPO / "data/slake/test_en_amanda.json").read_text(encoding="utf8"))}
CAPS = json.loads((R / "amanda-A" / os.environ.get("AMANDA_CAPTIONS", "captions_patched.json")).read_text(encoding="utf8"))
SPOKE_URL = os.environ.get("SPOKE_URL", "https://spoke.rbvi.ucsf.edu")
A1_GLOB = os.environ.get("A1_GLOB", "full-r*")
key_of = lambda e: "".join(c if c.isalnum() else "_" for c in e)[:120] or "_empty"   # noqa: E731
LOCK = threading.Lock()
STAT = Counter()


def neighborhood(entity: str, live: bool) -> list:
    f = OLD / f"{key_of(entity)}.json"
    if f.is_file():
        return json.loads(f.read_text())
    if not live:
        return []
    g = LIVE / f"{key_of(entity)}.json"
    if g.is_file():
        return json.loads(g.read_text())
    try:
        out = SPOKEClient(SpokeAPIConfig(base_uri=SPOKE_URL, timeout_seconds=60, depth=1)).get_neighborhood(entity)
        g.write_text(json.dumps(out))
        with LOCK: STAT["live_fetch_ok"] += 1
    except Exception as ex:                      # ghi lỗi, không cache lỗi
        with LOCK: STAT["live_fetch_err"] += 1; STAT["err:" + type(ex).__name__] += 1
        out = []
    return out


def rounds(history: list[str]) -> list[tuple[list[str], str]]:
    """Mỗi vòng bổ sung: (sub-question của vòng, lịch sử tại thời điểm gọi Retriever)."""
    out = []
    for i, h in enumerate(history):
        if not h.startswith("Reflection:"):
            continue
        j = i + 1; sq = []
        while j < len(history) and history[j].startswith(("Sub-Question", "Sub-Answer")):
            if history[j].startswith("Sub-Question"):
                sq.append(re.sub(r"^Sub-Question \d+:\s*", "", history[j]))
            j += 1
        out.append((sq, "\n".join(history[:j])))
    return out


def main() -> None:
    cfg = SpokeRetrieverConfig(); cf = ContextFilter(cfg.filter_config)
    llm = OpenAIChatLLM(api_key="EMPTY", model=os.environ.get("AMANDA_LLM_MODEL", "gemma-4-26b"),
                        base_url=os.environ.get("AMANDA_LLM_URL", "http://127.0.0.1:4001/v1"), temperature=0.0, max_tokens=512)
    ex = EntityExtractor(llm, max_entities=cfg.max_entities)
    jobs = []
    for d in sorted(glob.glob(str(R / f"amanda-A1/{A1_GLOB}"))):
        for l in open(Path(d) / "trace.jsonl"):
            t = json.loads(l)
            if t["iterations"] and t["iterations"] > 0:
                for k, (sq, hist) in enumerate(rounds(t["history"])):
                    jobs.append((Path(d).name, str(t["question_id"]), k + 1, sq, hist, bool(t.get("rag_context") and t["rag_context"] != "None")))

    def one(job):
        run, qid, rnd, sq, hist, fired_trace = job
        q = GT[qid]["conversations"][0]["value"].replace("<image>\n", "")
        cap = CAPS[GT[qid]["image"]]
        query = build_retrieval_query(q, caption=cap, subquestions=sq, history=hist)
        short = build_retrieval_query(q, caption=cap, subquestions=sq)
        try:
            ents = ex.extract(query)[: cfg.max_entities]
        except Exception as e:
            return {"run": run, "qid": qid, "round": rnd, "error": str(e)[:100]}
        res = {"run": run, "qid": qid, "round": rnd, "entities": ents, "fired_trace": fired_trace, "query_words": len(query.split()),
               "cache_hits": [(OLD / f"{key_of(e)}.json").is_file() for e in ents]}
        per = max(1, cfg.context_volume // max(1, len(ents))) if ents else 1
        for mode in ("cache", "live"):
            ctx = []
            for e in ents:
                tri = filter_relevant_triples(triples_from_spoke_items(list(neighborhood(e, mode == "live"))), cfg)
                ctx += triples_to_contexts(tri, include_provenance=cfg.include_provenance)[:per]
            res[f"{mode}_contexts"] = len(ctx)
            res[f"{mode}_kept"] = len(cf.filter(query, ctx)) if ctx else 0
            if mode == "live":
                res["live_short_kept"] = len(cf.filter(short, ctx)) if ctx else 0
                res["live_kept_facts"] = cf.filter(query, ctx)[:3] if ctx else []
        return res

    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(one, jobs))
    ok = [r for r in rows if "error" not in r]
    n_ent = sum(len(r["entities"]) for r in ok); n_hit = sum(sum(r["cache_hits"]) for r in ok)
    summ = {"calls": len(rows), "errors": len(rows) - len(ok), "runs": sorted({r["run"] for r in rows}),
            "calls_with_entities": sum(1 for r in ok if r["entities"]),
            "entities": n_ent, "entity_cache_hits": n_hit,
            "calls_with_contexts_cache": sum(1 for r in ok if r["cache_contexts"]),
            "calls_with_contexts_live": sum(1 for r in ok if r["live_contexts"]),
            "fired_trace": sum(1 for r in ok if r["fired_trace"]),
            "fired_cache": sum(1 for r in ok if r["cache_kept"]),
            "fired_live": sum(1 for r in ok if r["live_kept"]),
            "fired_live_short": sum(1 for r in ok if r["live_short_kept"]),
            "median_query_words": sorted(r["query_words"] for r in ok)[len(ok) // 2] if ok else None,
            "spoke": dict(STAT),
            "top_entities": Counter(e for r in ok for e in r["entities"]).most_common(25),
            "top_missing": Counter(e for r in ok for e, h in zip(r["entities"], r["cache_hits"]) if not h).most_common(25)}
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results/retriever_investigation.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
