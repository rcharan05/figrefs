"""M2 training data from free labels (PMC OA JATS), plus the FigSum scoring pairs.

Per article (bench/pmc_free/*.json) and figure F with a caption:
  P        sentences citing only F, where every figure reference is a parenthetical "(Fig. 2A)" / "(Figure 2, B and C)";
           the parentheticals are deleted, leaving a fluent sentence with no trace of the citation   -> label 1
  N-other  same kind of sentence citing only another figure G (same deletion), paired with F's caption -> label 0
  N-pre    sentences before F's first citation, outside that paragraph, citing nothing, not Abstract; half the
           hardest by TF-IDF similarity to F's caption, half random                                    -> label 0
Sentences near citations are never used as negatives (positive-unlabeled learning: unlabeled != negative).
Input to the model = (caption without "Figure N." prefix, sentence). No context sentences (avoids label leakage).
Split by article: 90% train / 10% valid.
Also writes figsum_pairs.jsonl: every FigSum (article, figure, candidate sentence) pair, for scoring.
usage: python build_m2.py PMC_DIR OUT_DIR
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAREN = re.compile(r"\s*\((?:[^()]*?\b(?:see\s+)?(?:Fig(?:ure)?s?\.?|Supplementary\s+Fig\w*\.?)\s*S?\d[^()]*)\)")
ANYFIG = re.compile(r"\b(?:Fig(?:ure)?s?\.?)\s*S?\d", re.I)
ANYREF = re.compile(r"\b(?:Fig(?:ure)?s?|Tables?|Panels?)\b", re.I)
CAPPFX = re.compile(r"^\s*(?:Fig(?:ure)?\.?|FIG\.?)\s*\d+\s*[.:|]?\s*", re.I)
CITE = re.compile(r"\[\d+(?:[,–-]\s*\d+)*\]")


def clean_paren(s: str) -> str | None:
    """delete parenthetical figure refs; None if any figure reference survives outside parentheses"""
    t = PAREN.sub("", s)
    if ANYFIG.search(t) or ANYREF.search(t):
        return None
    t = re.sub(r"\s+([.,;:])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t if len(t.split()) >= 5 else None


def caption_text(c: str) -> str:
    return CAPPFX.sub("", c).strip()[:700]


def article_examples(a, rng):
    from sklearn.feature_extraction.text import TfidfVectorizer
    S, cue, para = a["sentences"], a["cue"], a["sent_para"]
    sec = a["top_section"]
    figs = {int(k): caption_text(v) for k, v in a["figures"].items()}
    single = {}
    for i, c in enumerate(cue):
        if len(c) == 1 and c[0] in figs:
            t = clean_paren(S[i])
            if t:
                single.setdefault(c[0], []).append(t)
    first = {}
    for i, c in enumerate(cue):
        for f in c:
            first.setdefault(f, i)
    try:
        vec = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(S + list(figs.values()))
    except ValueError:
        return []
    out = []
    for f, cap in figs.items():
        pos = single.get(f, [])
        if not pos or f not in first:
            continue
        pos = rng.sample(pos, min(len(pos), 8))
        out += [(cap, t, 1, "P") for t in pos]
        others = [t for g, ts in single.items() if g != f for t in ts]
        out += [(cap, t, 0, "N-other") for t in rng.sample(others, min(len(others), len(pos)))]
        fi = first[f]
        pool = [i for i in range(fi) if para[i] != para[fi] and not cue[i] and not ANYREF.search(S[i])
                and "abstract" not in sec[para[i]].lower() and len(S[i].split()) >= 5]
        if pool:
            sims = (vec.transform([S[i] for i in pool]) @ vec.transform([cap]).T).toarray()[:, 0]
            hard = [pool[j] for j in sims.argsort()[::-1][:len(pos)]]
            rest = [i for i in pool if i not in set(hard)]
            rand = rng.sample(rest, min(len(rest), max(1, len(pos) // 2)))
            out += [(cap, CITE.sub("", S[i]).strip(), 0, "N-pre-hard") for i in hard]
            out += [(cap, CITE.sub("", S[i]).strip(), 0, "N-pre-rand") for i in rand]
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    rng = random.Random(230819)
    files = sorted(os.listdir(src))
    rng.shuffle(files)
    nval = max(50, len(files) // 10)
    counts = {}
    with open(os.path.join(dst, "train.jsonl"), "w") as ftr, open(os.path.join(dst, "valid.jsonl"), "w") as fva:
        for k, fn in enumerate(files):
            a = json.load(open(os.path.join(src, fn)))
            for cap, sent, y, kind in article_examples(a, rng):
                rec = json.dumps({"art": fn[:-5], "caption": cap, "sentence": sent, "label": y, "kind": kind})
                (fva if k < nval else ftr).write(rec + "\n")
                counts[("valid" if k < nval else "train", kind)] = counts.get(("valid" if k < nval else "train", kind), 0) + 1
    print("articles", len(files), "valid articles", nval)
    for k in sorted(counts):
        print(k, counts[k])
    # FigSum scoring pairs (candidates in features rows), same input format
    import phase2 as P
    D = P.load()
    with open(os.path.join(dst, "figsum_pairs.jsonl"), "w") as f:
        for r in D.rows:
            a = D.A[r["art"]]
            s = PAREN.sub("", a.sentences[r["i"]])
            s = re.sub(r"\s{2,}", " ", CITE.sub("", s)).strip()
            f.write(json.dumps({"key": f"{r['art']}|{r['fig']}|{r['i']}", "caption": caption_text(a.figures[r["fig"]]),
                                "sentence": s}) + "\n")
    print("figsum pairs", len(D.rows))


if __name__ == "__main__":
    main()
