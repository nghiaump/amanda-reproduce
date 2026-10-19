#!/usr/bin/env python3
"""Thí nghiệm A — đo tỉ lệ bắn của Retriever (extrinsic Med-KA) trong AMANDA.

    python scripts/amanda_retriever_probe.py --arm A1
    python scripts/amanda_retriever_probe.py --arm A2

⭐ Dùng ĐÚNG EntityExtractor + SpokeRetriever + ContextFilter của AMANDA (`third_party/AMANDA/src`),
không mô phỏng lại. Chỉ thay LLM engine (GPT-4o → gemma-4-26b cục bộ) và thiếu caption
(không đủ VRAM chạy LLaVA-Med) ⇒ **reproduction có thay thế**, khai báo rõ.

Hai nhánh chặn trên–dưới cho việc thiếu caption:
  A1 = chỉ câu hỏi            → đầu vào nghèo nhất, cận DƯỚI
  A2 = câu hỏi + đáp án gold  → thay cho caption hoàn hảo, cận TRÊN

⛔ Cache mọi phản hồi SPOKE xuống đĩa ngay lần đầu (mất cache là mất cả buổi).
"""
from __future__ import annotations

import argparse, json, os, sys, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "AMANDA" / "src"))

from amanda.retrieval.spoke.entity_extractor import EntityExtractor           # noqa: E402
from amanda.retrieval.spoke.spoke_client import SPOKEClient, SpokeAPIConfig   # noqa: E402
from amanda.retrieval.spoke.retriever import (                                # noqa: E402
    SpokeRetriever, SpokeRetrieverConfig, build_retrieval_query)

CACHE = REPO / "runs" / "amanda-A" / "spoke_cache"
# Cùng server LLM mà pipeline dùng (config.env → AMANDA_LLM_URL / AMANDA_LLM_MODEL).
LLM_URL = os.environ.get("AMANDA_LLM_URL", "http://127.0.0.1:4001/v1").rstrip("/") + "/chat/completions"
LLM_MODEL = os.environ.get("AMANDA_LLM_MODEL", "gemma-4-26b")


class LocalLLM:
    """Khớp giao thức EntityExtractionLLM của AMANDA."""
    def __init__(self):
        self.calls = 0; self.fails = 0; self._lk = threading.Lock()

    def chat(self, messages, *, answer_type: str) -> str:
        import urllib.request
        body = json.dumps({"model": LLM_MODEL, "messages": list(messages),
                           "max_tokens": 2000, "temperature": 0}).encode()
        req = urllib.request.Request(LLM_URL, data=body, headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    out = json.load(r)["choices"][0]["message"].get("content") or ""
                with self._lk: self.calls += 1
                return out
            except Exception:
                if attempt == 2:
                    with self._lk: self.fails += 1
                    return ""
                time.sleep(2 * (attempt + 1))
        return ""


class CachedSPOKE(SPOKEClient):
    """SPOKEClient của AMANDA + cache đĩa. Không đổi logic truy vấn."""
    def __init__(self, cfg):
        super().__init__(cfg)
        CACHE.mkdir(parents=True, exist_ok=True)
        self._types = None; self._lk = threading.Lock()
        self.hits = self.miss = self.err = 0

    def get_types(self):
        if self._types is None:
            f = CACHE / "_types.json"
            self._types = json.loads(f.read_text()) if f.is_file() else super().get_types()
            f.write_text(json.dumps(self._types))
        return self._types

    def get_neighborhood(self, node_value: str):
        key = "".join(c if c.isalnum() else "_" for c in node_value)[:120] or "_empty"
        f = CACHE / f"{key}.json"
        if f.is_file():
            with self._lk: self.hits += 1
            return json.loads(f.read_text())
        try:
            out = super().get_neighborhood(node_value)
        except Exception:
            # Lỗi mạng / SPOKE trả 5xx: KHÔNG ghi cache, để lần chạy sau (khi SPOKE sống lại) hỏi lại.
            with self._lk: self.err += 1
            return []
        f.write_text(json.dumps(out))
        with self._lk: self.miss += 1
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A1", "A2"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    rows = [x for x in json.loads((REPO / "data/slake/test.json").read_text(encoding="utf8"))
            if x["q_lang"] == "en"]
    if a.limit: rows = rows[:a.limit]

    llm = LocalLLM()
    client = CachedSPOKE(SpokeAPIConfig(base_uri="https://spoke.rbvi.ucsf.edu", timeout_seconds=60, depth=1))
    retr = SpokeRetriever(entity_extractor=EntityExtractor(llm, max_entities=3),
                          client=client, config=SpokeRetrieverConfig())

    print(f"arm={a.arm}  n={len(rows)}  LLM={LLM_MODEL}  workers={a.workers}", flush=True)
    done = [0]; lk = threading.Lock()

    def one(r):
        q = r["question"] if a.arm == "A1" else f"{r['question']} Answer: {r['answer']}"
        try:
            ents = retr.entity_extractor.extract(build_retrieval_query(q))
        except Exception:
            ents = []
        matched, triples = [], 0
        for e in ents[:3]:
            items = client.get_neighborhood(e)
            if items:
                matched.append(e); triples += len(items)
        ctx = None
        if matched:
            try: ctx = retr.retrieve(question=q)
            except Exception: ctx = None
        with lk:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  {done[0]}/{len(rows)}  cache {client.hits}h/{client.miss}m", flush=True)
        return {"qid": r["qid"], "content_type": r["content_type"], "answer_type": r["answer_type"],
                "entities": ents, "matched": matched, "n_spoke_items": triples,
                "context": ctx, "fired": bool(ctx)}

    # Resume: mỗi câu xong được ghi ngay vào file .partial.jsonl; chạy lại thì bỏ qua câu đã có.
    partial = REPO / "runs" / "amanda-A" / f"{a.arm}.partial.jsonl"
    partial.parent.mkdir(parents=True, exist_ok=True)
    have = {}
    if partial.is_file():
        for line in partial.read_text(encoding="utf8").splitlines():
            if line.strip():
                r = json.loads(line); have[r["qid"]] = r
    todo = [r for r in rows if r["qid"] not in have]
    print(f"  đã có {len(have)} câu từ lần chạy trước · cần chạy {len(todo)}", flush=True)

    t0 = time.time()
    with partial.open("a", encoding="utf8") as fp:
        def one_and_save(r):
            out = one(r)
            with lk:
                fp.write(json.dumps(out, ensure_ascii=False) + "\n"); fp.flush()
            return out
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for out in ex.map(one_and_save, todo):
                have[out["qid"]] = out
    res = [have[r["qid"]] for r in rows]          # giữ đúng thứ tự của test.json

    out = REPO / "runs" / "amanda-A" / f"{a.arm}.json"
    out.write_text(json.dumps({"arm": a.arm, "llm": LLM_MODEL, "n": len(res),
                               "runtime_sec": round(time.time() - t0, 1),
                               "llm_calls": llm.calls, "llm_fails": llm.fails,
                               "spoke_cache_hits": client.hits, "spoke_queries": client.miss,
                               "spoke_errors": client.err,
                               "results": res}, indent=1, ensure_ascii=False), encoding="utf8")

    n = len(res)
    print(f"\n{'='*62}\narm {a.arm} — {time.time()-t0:.0f}s · LLM {llm.calls} lượt ({llm.fails} lỗi)")
    print(f"  SPOKE: {client.hits} lượt trúng cache · {client.miss} lượt hỏi mạng · {client.err} lượt LỖI"
          + ("  ⚠ entity bị lỗi được coi như rỗng — chạy lại khi SPOKE hoạt động" if client.err else ""))
    print(f"  trích được ≥1 entity : {sum(1 for r in res if r['entities']):>5}/{n} = {100*sum(1 for r in res if r['entities'])/n:.1f}%")
    print(f"  neo được ≥1 vào SPOKE: {sum(1 for r in res if r['matched']):>5}/{n} = {100*sum(1 for r in res if r['matched'])/n:.1f}%")
    print(f"  ⭐ BẮN (context ≠ rỗng): {sum(1 for r in res if r['fired']):>5}/{n} = {100*sum(1 for r in res if r['fired'])/n:.1f}%")
    print(f"\n  theo content_type:")
    g = defaultdict(lambda: [0, 0])
    for r in res:
        g[r["content_type"]][0] += r["fired"]; g[r["content_type"]][1] += 1
    for k, (f, t) in sorted(g.items(), key=lambda x: -x[1][0]/x[1][1]):
        print(f"    {k:<16}{f:>4}/{t:<5} {100*f/t:>5.1f}%")
    ents = Counter(e for r in res for e in r["entities"])
    print(f"\n  entity hay gặp nhất: {', '.join(f'{e}({c})' for e,c in ents.most_common(8))}")


if __name__ == "__main__":
    main()
