"""Image -> text for every cropped figure (run 4 SPAN-VIS): Qwen2-VL-2B-Instruct, two prompts per figure.

  ocr   exact text visible in the figure (axis labels, legends, panel letters, numbers)
  desc  literal visual description, panel by panel (chart types, axes, groups, colours, lines, trends), with no
        biological interpretation
fp32 on GPU (T4 has no bf16; fp16 can overflow in Qwen2-VL), greedy decoding, max 160 new tokens, images capped at
~600k pixels. Writes OUT.json {"paper|fig": {"ocr": ..., "desc": ...}}; figures without a crop get nothing.
usage: python figtext.py IMG_DIR OUT.json [SHARD K OF N]
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

PROMPTS = {
    "ocr": ("List the words and numbers printed in this figure (axis titles, tick labels, legend entries, panel "
            "letters), separated by commas. Answer with the list only."),
    "desc": ("Describe this chart or image literally: the type of plot or image in each panel, the axis labels, the "
             "legend entries, and which bars, lines or groups are highest and lowest. Use only words and labels "
             "that are visible in the image. Do not guess what the samples are. At most 80 words."),
}
MODEL = os.environ.get("FIGTEXT_MODEL", "Qwen/Qwen2-VL-2B-Instruct")


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    img_dir, out = sys.argv[1], sys.argv[2]
    k, n = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
    limit = int(os.environ.get("FIGTEXT_LIMIT", "0"))
    index = json.load(open(os.path.join(img_dir, "index.json")))
    tp = os.path.join(img_dir, "pdf_text.json")
    pdf_text = json.load(open(tp)) if os.path.exists(tp) else {}
    items = sorted((key, fn) for key, fn in index.items() if fn)[k::n]
    if limit:
        items = items[:limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=128 * 28 * 28, max_pixels=768 * 28 * 28)
    dt = getattr(torch, os.environ.get("FIGTEXT_DTYPE", "float32"))     # bfloat16 only for low-RAM CPU tests
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL, dtype=dt)
    except TypeError:
        model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=dt)
    model = model.to(dev).eval()
    res, t0 = {}, time.time()
    for j, (key, fn) in enumerate(items):
        img = Image.open(os.path.join(img_dir, fn)).convert("RGB")
        res[key] = {}
        for name, prompt in PROMPTS.items():
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = proc.apply_chat_template(msgs, add_generation_prompt=True)
            enc = proc(text=[text], images=[img], return_tensors="pt").to(dev)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=160, do_sample=False, repetition_penalty=1.15)
            ans = proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            ans = " ".join(ans.split())[:900]
            if name == "ocr":                      # drop repeated labels (small VLMs loop on tick values)
                seen, keep = set(), []
                for t in (x.strip() for x in ans.split(",")):
                    if t and t.lower() not in seen:
                        seen.add(t.lower())
                        keep.append(t)
                ans = ", ".join(keep[:40])
            if name == "ocr" and (pdf_text.get(key) or re.fullmatch(r"[\d(),.\s]*", ans)):
                ans = pdf_text.get(key, "")        # exact text layer of a vector figure; drop box-coordinate answers
            res[key][name] = ans
        print(f"[figtext {k}/{n}] {j + 1}/{len(items)} {key}  {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
