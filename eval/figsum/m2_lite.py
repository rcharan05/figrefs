"""M2-lite: CPU content model on the free-label data (sanity check of the data before GPU training).

Features of a (caption, sentence) pair, hashed:
  s:<w>     sentence words (uni+bigrams)     -> what a figure-describing sentence reads like
  b:<w>     words shared by caption and sentence -> which overlaps matter
  cos       TF-IDF cosine(caption, sentence), number of shared content words
Logistic regression. Reports valid AUC (overall and per negative kind), FigSum dev AUCs, and writes
m2lite_scores.json {"art|fig|i": logit} for all FigSum pairs.
usage: python m2_lite.py DATA_DIR
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = re.compile(r"[a-z][a-z0-9\-]+")
STOP = set("""a an the of in on at to for from by with and or but is are was were be been being this that these those
it its as into than then there their they we our not no can may also which who whom whose such between within both
each other more most less via using used use after before during under over""".split())


def toks(s):
    return [w for w in TOK.findall(s.lower()) if w not in STOP and len(w) > 2]


def feats(cap, sent):
    ts, tc = toks(sent), set(toks(cap))
    f = [f"s:{w}" for w in ts] + [f"s:{a}_{b}" for a, b in zip(ts, ts[1:])]
    shared = [w for w in set(ts) if w in tc]
    f += [f"b:{w}" for w in shared]
    return " ".join(f), len(shared), len(shared) / (np.sqrt(len(set(ts)) * len(tc)) + 1e-6)


def load(p):
    return [json.loads(l) for l in open(p)]


def matrix(recs, hv, cap_key="caption", sent_key="sentence"):
    docs, extra = [], []
    for r in recs:
        d, n, c = feats(r[cap_key], r[sent_key])
        docs.append(d)
        extra.append([np.log1p(n), c])
    return sp.hstack([hv.transform(docs), sp.csr_matrix(np.array(extra))]).tocsr()


def main():
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    d = sys.argv[1]
    tr, va = load(os.path.join(d, "train.jsonl")), load(os.path.join(d, "valid.jsonl"))
    hv = HashingVectorizer(n_features=2 ** 20, token_pattern=r"\S+", lowercase=False, alternate_sign=False,
                           norm="l2")
    Xtr, ytr = matrix(tr, hv), np.array([r["label"] for r in tr])
    clf = LogisticRegression(C=2.0, max_iter=2000).fit(Xtr, ytr)
    pv = clf.decision_function(matrix(va, hv))
    yv = np.array([r["label"] for r in va])
    print(f"train {len(tr)}  valid {len(va)}  valid AUC {roc_auc_score(yv, pv):.3f}")
    kinds = np.array([r["kind"] for r in va])
    for k in ("N-other", "N-pre-hard", "N-pre-rand"):
        m = (kinds == "P") | (kinds == k)
        print(f"   P vs {k:11s} AUC {roc_auc_score(yv[m], pv[m]):.3f}")
    pairs = load(os.path.join(d, "figsum_pairs.jsonl"))
    ps = clf.decision_function(matrix(pairs, hv))
    out = {r["key"]: float(s) for r, s in zip(pairs, ps)}
    json.dump(out, open(os.path.join(HERE, "m2lite_scores.json"), "w"))
    sys.path.insert(0, HERE)
    import phase2 as P
    D = P.load()
    dev = D.split["dev"]
    for lab, cond in (("window d<=3", lambda r: r["d_sent"] <= 3), ("B0 picks", lambda r: r["_b0"]),
                      ("d=2-3 same para", lambda r: 2 <= r["d_sent"] <= 3 and r["same_para"]),
                      ("far", lambda r: r["_ok_far"]), ("all", lambda r: True)):
        R = [r for r in D.rows if r["art"] in dev and cond(r)]
        y = [r["y"] for r in R]
        s = [out[f"{r['art']}|{r['fig']}|{r['i']}"] for r in R]
        print(f"FigSum dev {lab:16s} n={len(R):5d} AUC m2lite {roc_auc_score(y, s):.3f}  "
              f"tf_ment {roc_auc_score(y, [r['tf_ment'] for r in R]):.3f}  tf_cap {roc_auc_score(y, [r['tf_cap'] for r in R]):.3f}")


if __name__ == "__main__":
    main()
