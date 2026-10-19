#!/usr/bin/env python3
"""Sinh caption cho ảnh SLAKE bằng ĐÚNG Perceiver + LLaVAMedAdapter của AMANDA.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/amanda_captions.py \
        --patched-decode --out runs/amanda-A/captions_patched.json          # đúng như bước 06

⭐ Vì sao đáng làm riêng một bước: SLAKE en test có **1.061 câu hỏi nhưng chỉ 96 ẢNH riêng biệt**.
   Caption phụ thuộc ẢNH chứ không phụ thuộc câu hỏi ⇒ chỉ cần 96 lượt forward, không phải 1.061.
   Cache lại thì mọi arm sau dùng chung (run_pipeline.py --captions), không phải sinh lại.

⛔ Dùng `random.seed` cố định vì Perceiver bốc NGẪU NHIÊN 1 trong 16 prompt caption
   (Table 11 của bài báo) ⇒ không cố định thì hai lần chạy ra hai caption khác nhau.
"""
from __future__ import annotations

import argparse, json, random, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "AMANDA" / "src"))
sys.path.insert(0, str(REPO / "src"))

from amanda.agents.perceiver import Perceiver                       # noqa: E402
from amanda.mllm.adapters.llava_med_adapter import LLaVAMedAdapter  # noqa: E402

MODEL = REPO / "models" / "llava-med-v1.5-mistral-7b"
LLAVA_REPO = REPO / "third_party" / "LLaVA-Med"
IMG_ROOT = Path((REPO / "data/slake/image_root.txt").read_text().strip())
OUT = REPO / "runs" / "amanda-A" / "captions.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    # ⚠️ configs/amanda.example.yaml của tác giả ghi conv_mode: "simple" — mode đó KHÔNG TỒN TẠI
    #    trong microsoft/LLaVA-Med @30697ca. Model là llava_mistral ⇒ dùng mistral_instruct,
    #    đúng như chính LLaVA-Med hướng dẫn (--conv-mode mistral_instruct).
    ap.add_argument("--conv-mode", default="mistral_instruct")
    # --patched-decode: dùng PatchedLLaVAMedAdapter (src/amanda_lab/patched_adapter.py) thay cho adapter gốc.
    # --out: file đầu ra (mặc định runs/amanda-A/captions.json; bước 06 dùng runs/amanda-A/captions_patched.json).
    ap.add_argument("--patched-decode", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)

    rows = [x for x in json.loads((REPO / "data/slake/test.json").read_text(encoding="utf8"))
            if x["q_lang"] == "en"]
    images = sorted({x["img_name"] for x in rows})
    if a.limit:
        images = images[:a.limit]

    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text(encoding="utf8")) if out.is_file() else {}
    todo = [i for i in images if i not in done]
    print(f"ảnh: {len(images)} riêng biệt (từ {len(rows)} câu hỏi) · đã có {len(done)} · cần sinh {len(todo)}",
          flush=True)
    if not todo:
        print("✅ đã đủ caption, không phải sinh lại"); return

    random.seed(a.seed)
    t0 = time.time()
    if a.patched_decode:
        from amanda_lab.patched_adapter import PatchedLLaVAMedAdapter as Adapter
    else:
        Adapter = LLaVAMedAdapter
    adapter = Adapter(model_path=MODEL, device=a.device, llava_repo_path=LLAVA_REPO,
                              conv_mode=a.conv_mode, temperature=0.0, caption_max_new_tokens=256)
    perceiver = Perceiver(adapter)
    print(f"  nạp model xong sau {time.time()-t0:.0f}s", flush=True)

    t1 = time.time()
    for k, name in enumerate(todo, 1):
        path = IMG_ROOT / name
        if not path.is_file():
            print(f"  ⚠️ thiếu ảnh {name}"); continue
        cap = perceiver.caption(adapter.load_image(path))
        done[name] = cap
        if k % 5 == 0 or k == len(todo):
            el = time.time() - t1
            print(f"  {k}/{len(todo)}  {el/k:.1f}s/ảnh  còn ~{(len(todo)-k)*el/k/60:.0f} phút", flush=True)
            out.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf8")

    out.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf8")
    print(f"\n✅ {len(done)} caption · {time.time()-t1:.0f}s · → {out}")
    k0 = todo[0]
    print(f"\nVí dụ ({k0}):\n  {done[k0][:400]}")


if __name__ == "__main__":
    main()
