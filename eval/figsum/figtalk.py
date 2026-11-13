"""'Figure-talk' classifier by distant supervision (no FigSum labels used).

Training data: 200 bioRxiv JATS papers (bench/biorxiv). Label = the sentence cites a figure (main or supplementary).
Every figure / table reference is masked out of the text (with "shown in", "see", panel letters), so the model has to
learn how result-describing prose reads, not the citation itself. Sentences citing tables are dropped.
Model: word 1-2gram TF-IDF + logistic regression. Held-out check: 20% of the bioRxiv papers.
Then scores every FigSum sentence (masked the same way) -> figtalk_GS1.json {art: [p, ...]} and reports, on FigSum dev
only, how well it separates implicit gold from other candidates.
usage: python figtalk.py
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_article  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BIO = "/home/charan/bench/biorxiv"
REF = r"(?:Supplementary\s+|Suppl\.?\s+|Extended\s+Data\s+)?(?:Fig(?:ure)?s?\.?|Panels?)\s*S?\d+(?:\s*[A-Za-z]\b)?(?:\s*(?:,|and|&|–|-|to)\s*(?:S?\d+)?(?:\s*[A-Za-z]\b)?)*"
VERB = r"(?:(?:as\s+)?(?:is\s+|are\s+|was\s+|were\s+)?(?:shown|seen|depicted|illustrated|presented|summari[sz]ed|displayed|plotted|indicated|see)\s+(?:in\s+)?(?:the\s+)?)?"
PAREN = re.compile(r"\([^()]*\b(?:Fig|Figure|Panel|Table)[^()]*\)", re.I)
INLINE = re.compile(VERB + REF, re.I)
ANYFIG = re.compile(r"\b(?:Fig(?:ure)?s?\.?)\s*S?\d", re.I)
TABLE = re.compile(r"\bTables?\s*S?\d", re.I)


def mask(s: str) -> str:
    s = PAREN.sub(" ", s)
    s = INLINE.sub(" ", s)
    s = re.sub(r"\b(?:Fig(?:ure)?s?|panels?)\b", " ", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def biorxiv():
    out = []
    for pid in sorted(os.listdir(BIO)):
        p = os.path.join(BIO, pid, "paper.xml")
        if not os.path.exists(p):
            continue
        try:
            a = load_article(pid, p)
        except Exception:  # noqa: BLE001
            continue
        for s, c in zip(a.sentences, a.cue):
            if TABLE.search(s):
                continue
            out.append((pid, mask(s), int(bool(c) or bool(ANYFIG.search(s)))))
    return out


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rows = biorxiv()
    pids = sorted({r[0] for r in rows})
    random.Random(230819).shuffle(pids)
    held = set(pids[: len(pids) // 5])
    tr = [r for r in rows if r[0] not in held]
    te = [r for r in rows if r[0] in held]
    print(f"bioRxiv sentences {len(rows)} (positive {sum(r[2] for r in rows)}), papers {len(pids)}")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, lowercase=True)
    X = vec.fit_transform([r[1] for r in tr])
    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(X, [r[2] for r in tr])
    p = clf.predict_proba(vec.transform([r[1] for r in te]))[:, 1]
    print(f"held-out bioRxiv AUC {roc_auc_score([r[2] for r in te], p):.3f}")
    # refit on all bioRxiv, score FigSum
    X = vec.fit_transform([r[1] for r in rows])
    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(X, [r[2] for r in rows])
    w = clf.coef_[0]
    names = vec.get_feature_names_out()
    order = np.argsort(w)
    print("top positive n-grams:", ", ".join(names[order[-30:]][::-1]))
    print("top negative n-grams:", ", ".join(names[order[:20]]))
    import phase2 as P
    D = P.load()
    out = {aid: [float(x) for x in clf.predict_proba(vec.transform([mask(s) for s in a.sentences]))[:, 1]]
           for aid, a in D.A.items()}
    json.dump(out, open(os.path.join(HERE, "figtalk_GS1.json"), "w"))
    dev = D.split["dev"]
    for lab, cond in (("window d<=3", lambda r: r["d_sent"] <= 3), ("B0 picks", lambda r: r["_b0"]),
                      ("d=2-3 same para", lambda r: 2 <= r["d_sent"] <= 3 and r["same_para"] == 1),
                      ("far", lambda r: r["_ok_far"])):
        R = [r for r in D.rows if r["art"] in dev and cond(r)]
        y = [r["y"] for r in R]
        s = [out[r["art"]][r["i"]] for r in R]
        print(f"FigSum dev {lab:16s} n={len(R):5d} gold={sum(y):4d}  AUC figtalk {roc_auc_score(y, s):.3f}")
    # far: does any-figure gold (sentence is gold for some figure) separate?
    R = [r for r in D.rows if r["art"] in dev and r["_ok_far"]]
    anyg = {}
    for r in R:
        anyg[(r["art"], r["i"])] = max(anyg.get((r["art"], r["i"]), 0), r["y"])
    keys = sorted(anyg)
    print(f"FigSum dev far, sentence-level (gold for any figure) n={len(keys)} gold={sum(anyg.values())} "
          f"AUC {roc_auc_score([anyg[k] for k in keys], [out[k[0]][k[1]] for k in keys]):.3f}")


if __name__ == "__main__":
    main()
