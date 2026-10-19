#!/usr/bin/env python3
"""Chạy AMANDA / baseline trên SLAKE, dùng ĐÚNG agent + adapter + thước của tác giả.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python scripts/amanda_run.py --arm A0 --limit 20
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python scripts/amanda_run.py --arm A0

Arm hiện có:
  A0  baseline — LLaVA-Med v1.5 trả lời MỘT BƯỚC, không agent nào khác.
      Đây là **CỔNG**: phải rơi gần Table 1 (SLAKE Open 41,74 · Closed 44,95) thì các arm sau mới đáng tin.

Đầu ra JSONL khớp hợp đồng của `amanda.evaluation.evaluate_medvqa`:
  `question_id` · `prompt` (= conversations[0].value, tức "<image>\\n"+câu hỏi) · `text`
⇒ chấm bằng chính bộ chấm của tác giả, không tự viết lại thước.

⛔ Ghi JSONL sau MỖI câu ⇒ mất điện/OOM vẫn resume được (bài học từ lần OOM 18/08).
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "AMANDA" / "src"))

from amanda.agents.perceiver import Perceiver                       # noqa: E402
from amanda.mllm.adapters.llava_med_adapter import LLaVAMedAdapter  # noqa: E402
sys.path.insert(0, str(REPO / "src"))
from amanda_lab.patched_adapter import PatchedLLaVAMedAdapter  # noqa: E402
from amanda.evaluation.evaluate_medvqa import (                     # noqa: E402
    evaluate_records, format_results_table)

SLAKE = REPO / "data" / "slake"
MODEL = REPO / "models" / "llava-med-v1.5-mistral-7b"
LLAVA_REPO = REPO / "third_party" / "LLaVA-Med"
# Đích của tác giả, Table 1 hàng LLaVA-Med-v1.5 cột SLAKE — CHỈ để đối chiếu.
PAPER = {"A0": {"open": 41.74, "closed": 44.95}, "A1": {"open": 54.39, "closed": 70.43}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A0", choices=["A0"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conv-mode", default="mistral_instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--raw-decode", action="store_true", help="dùng _decode_output GỐC (có bug) để tái hiện A0 đỏ")
    ap.add_argument("--expect", default="", help="kỳ vọng đăng ký TRƯỚC khi chạy")
    ap.add_argument("--tag", default="", help="hậu tố thư mục đầu ra, vd. r1 hoặc BUG-rawdecode")
    a = ap.parse_args()

    gt = json.loads((SLAKE / "test_en_amanda.json").read_text(encoding="utf8"))
    if a.limit:
        gt = gt[: a.limit]
    img_root = Path((SLAKE / "image_root.txt").read_text().strip())

    sub = f"limit{a.limit}" if a.limit else "full"
    out_dir = REPO / "runs" / f"amanda-{a.arm}" / (f"{sub}-{a.tag}" if a.tag else sub)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"

    done = {}
    if pred_path.is_file():
        for line in pred_path.read_text(encoding="utf8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[str(r["question_id"])] = r
    todo = [g for g in gt if str(g["question_id"]) not in done]
    print(f"arm={a.arm}  n={len(gt)}  đã có {len(done)}  cần chạy {len(todo)}", flush=True)
    if a.expect:
        (out_dir / "EXPECTATION.txt").write_text(a.expect, encoding="utf8")
        print(f"kỳ vọng đăng ký trước: {a.expect}", flush=True)

    if todo:
        import torch
        t0 = time.time()
        Cls = LLaVAMedAdapter if a.raw_decode else PatchedLLaVAMedAdapter
        adapter = Cls(model_path=MODEL, device=a.device, llava_repo_path=LLAVA_REPO,
                                  conv_mode=a.conv_mode, temperature=0.0)
        perceiver = Perceiver(adapter)
        print(f"  nạp model {time.time()-t0:.0f}s · VRAM {torch.cuda.memory_allocated()/2**30:.1f} GB",
              flush=True)

        t1 = time.time()
        with pred_path.open("a", encoding="utf8") as fh:
            for k, g in enumerate(todo, 1):
                img = adapter.load_image(img_root / g["image"])
                question = g["conversations"][0]["value"].replace("<image>\n", "")
                # A0 = đúng bước "initial_answer" của Perceiver, không Reasoner/Evaluator/Explorer/Retriever
                ans = perceiver.initial_answer(img, question, g["answer_type"])
                fh.write(json.dumps({"question_id": g["question_id"],
                                     "prompt": g["conversations"][0]["value"],
                                     "text": ans, "answer_type": g["answer_type"],
                                     "content_type": g.get("content_type")},
                                    ensure_ascii=False) + "\n")
                fh.flush()
                if k % 20 == 0 or k == len(todo):
                    el = time.time() - t1
                    print(f"  {k}/{len(todo)}  {el/k:.2f}s/câu  còn ~{(len(todo)-k)*el/k/60:.0f} phút",
                          flush=True)

    preds = [json.loads(l) for l in pred_path.read_text(encoding="utf8").splitlines() if l.strip()]
    summary = evaluate_records(ground_truth=gt, predictions=preds)
    print(f"\n{'='*58}\narm {a.arm} — n={len(preds)}")
    print(format_results_table(summary.metrics))
    tgt = PAPER.get(a.arm)
    if tgt:
        print(f"\n  đích của tác giả (Table 1, SLAKE): Open {tgt['open']} · Closed {tgt['closed']}")
        print("  ⚠️ LLM engine và môi trường khác bài gốc ⇒ reproduction có thay thế, không phải replication")
    (out_dir / "metrics.json").write_text(
        json.dumps({"arm": a.arm, "n": len(preds), "metrics": summary.metrics,
                    "paper_target": tgt, "conv_mode": a.conv_mode, "raw_decode": a.raw_decode},
                   indent=1, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
