"""Back-translated paraphrases (en -> de -> en, MarianMT, greedy) of every sentence of the TRAINING-pool articles.

Label-agnostic: every sentence gets one (positives and negatives alike), so "translationese" cannot become a
label shortcut; the trainer swaps a candidate for its paraphrase with probability 0.3.
Figure mentions are protected: each merged mention span becomes "Figure 9k" before translation and the original
span text is put back afterwards; if any placeholder is lost or duplicated, the paraphrase is discarded.
Kept only if token-F1 with the original is in [0.35, 0.92] (changed, but still the same sentence).
usage: python backtranslate.py ARTICLES.jsonl OUT.json      (CPU, about 15-25 min for the 15 training papers)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import torch
from transformers import MarianMTModel, MarianTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textaug import mention_spans  # noqa: E402

TOK = re.compile(r"[a-z0-9]+")
PH = re.compile(r"(?:Fig(?:ure)?s?\.?|Abb(?:ildung)?(?:en)?\.?)\s*9(\d)\b")


def tf1(a, b):
    a, b = set(TOK.findall(a.lower())), set(TOK.findall(b.lower()))
    return 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0


def protect(s):
    spans = mention_spans(s)
    if len(spans) > 9:
        return None, []
    out, last, orig = [], 0, []
    for k, (st, en, _) in enumerate(spans, 1):
        out.append(s[last:st])
        out.append(f"Figure 9{k}")
        orig.append(s[st:en])
        last = en
    out.append(s[last:])
    return "".join(out), orig


def restore(t, orig):
    found = [int(m.group(1)) for m in PH.finditer(t)]
    if sorted(found) != list(range(1, len(orig) + 1)):
        return None
    return PH.sub(lambda m: orig[int(m.group(1)) - 1], t)


def translate(model, tok, texts, bs=16):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            gen = model.generate(**enc, num_beams=1, max_new_tokens=300)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def main():
    arts = [json.loads(l) for l in open(sys.argv[1])]
    torch.set_num_threads(os.cpu_count() or 2)
    items = []
    for a in arts:
        if a["split"] != "train":
            continue
        for i, s in enumerate(a["sentences"]):
            p, orig = protect(s)
            if p is not None:
                items.append((a["id"], i, s, p, orig))
    print("sentences", len(items), flush=True)
    m1, t1 = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de"), MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    m2, t2 = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en"), MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    t0 = time.time()
    de = translate(m1, t1, [x[3] for x in items])
    print(f"en->de done {time.time() - t0:.0f}s", flush=True)
    en = translate(m2, t2, de)
    print(f"de->en done {time.time() - t0:.0f}s", flush=True)
    res, stats = {}, {"kept": 0, "lost_placeholder": 0, "too_close": 0, "too_far": 0}
    for (pid, i, s, p, orig), t in zip(items, en):
        r = restore(t, orig)
        if r is None:
            stats["lost_placeholder"] += 1
            continue
        f = tf1(s, r)
        if f > 0.92:
            stats["too_close"] += 1
        elif f < 0.35:
            stats["too_far"] += 1
        else:
            res.setdefault(pid, {})[str(i)] = r
            stats["kept"] += 1
    json.dump(res, open(sys.argv[2], "w"), ensure_ascii=False, indent=0)
    print(stats)


if __name__ == "__main__":
    main()
