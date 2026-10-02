#!/usr/bin/env python3
"""Ada (local GPU) half of the change-map test, fully independent of the API run.

    cd ~/changemap && sbatch ada_check.sbatch

Qwen3-VL-8B (4-bit, as in the haze screen) gets before, after and change_map.png with the
teammate's prompt, then the same model marks its own caption (text only) as
land_use / seasonal / none / unclear. Output: results_ada_8b.jsonl (resumable).
Expects next to it: pairs.json, maps/<patch_id>/{change_map.png,prompt.txt}, pairs/<patch_id>/{before,after}.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MARK = ("Below is a caption written about two satellite images. Classify what it concludes. "
        "Reply with one word only: land_use (it reports something lasting changed, e.g. new buildings, roads, "
        "construction, ponds, clearing), seasonal (only seasonal/vegetation/water differences), "
        "none (no change), or unclear.\n\nCaption:\n")


def load():
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.float16,
                                                            quantization_config=q).eval()
    return model, AutoProcessor.from_pretrained(MODEL)


@torch.inference_mode()
def ask(model, processor, content, images, max_new_tokens):
    msgs = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images or None, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def caption_text(raw: str) -> str:
    try:
        j = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        return f"{j.get('short_caption', '')} {j.get('detailed_caption', '')}".strip()
    except ValueError:
        return raw


def main() -> None:
    out = HERE / "results_ada_8b.jsonl"
    done = {json.loads(l)["patch_id"] for l in open(out)} if out.exists() else set()
    model, processor = load()
    for p in json.loads((HERE / "pairs.json").read_text()):
        pid = p["patch_id"]
        if pid in done:
            continue
        t0 = time.time()
        imgs = [Image.open(HERE / "pairs" / pid / "before.png").convert("RGB").resize((512, 512)),
                Image.open(HERE / "pairs" / pid / "after.png").convert("RGB").resize((512, 512)),
                Image.open(HERE / "maps" / pid / "change_map.png").convert("RGB")]
        content = [{"type": "text", "text": "Image 1 (BEFORE):"}, {"type": "image"},
                   {"type": "text", "text": "Image 2 (AFTER):"}, {"type": "image"},
                   {"type": "text", "text": "Image 3 (CHANGE MAP):"}, {"type": "image"},
                   {"type": "text", "text": (HERE / "maps" / pid / "prompt.txt").read_text()}]
        raw = ask(model, processor, content, imgs, 400)
        m = ask(model, processor, [{"type": "text", "text": MARK + caption_text(raw)}], None, 5)
        m = m.strip().lower().strip(".").strip()
        rec = {"patch_id": pid, "label": p["label"], "raw": raw,
               "mark": m if m in ("land_use", "seasonal", "none", "unclear") else "unparsed",
               "seconds": round(time.time() - t0, 1)}
        with open(out, "a") as sink:
            sink.write(json.dumps(rec) + "\n")
        print(pid, rec["mark"], rec["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
