"""Normalised-similarity features (round 3), added to the round-1 candidate rows.

For every figure f and every sentence s, raw similarity to three queries, with two encoders:
  cap    the caption                     (TF-IDF cosine; BGE cosine)
  ment   the figure's mention sentences   (TF-IDF: their concatenation; BGE: mean of their embeddings)
  topic  title + abstract
Each raw similarity x is turned into a z-score against a reference set R:
  all    every sentence of the article:          (x - mean_all) / sd_all
  ment   the sentences that name the figure:      (x - mean_mentions) / sd_all   (sd from all: 1-2 mentions give no sd)
  local  the candidates near the mention (2 before .. 5 after): (x - mean_local) / sd_local
For the ment query, a mention sentence's own similarity is computed leave-one-out (against the other mentions).
Also: signed distance (da = sentences after the closest preceding mention, db = before the closest following one),
and the "chain" of sentences between a candidate and its mention (for the contiguity variant).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

HERE = os.path.dirname(__file__)
ENCS, QUERIES, REFS = ("tf", "bge"), ("cap", "ment", "topic"), ("all", "ment", "local")
LOCAL_B, LOCAL_A = 2, 5


def _unit(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def article_sims(a):
    """raw similarity arrays: sims[enc][q] -> [n_sent, n_fig] (topic: [n_sent, 1]); loo[enc] -> ment sims leave-one-out"""
    from figrefs.dense import embed
    S, n = a.sentences, len(a.sentences)
    figs = sorted(a.figures)
    abstract = " ".join(S[i] for i in range(n) if a.top_section[a.sent_para[i]] == "Abstract")
    topic = a.title + ". " + abstract
    ment = {f: [i for i in range(n) if f in a.cue[i]] for f in figs}
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9\-]+\b")
    vec.fit(S + [a.figures[f] for f in figs])
    TS = _unit(vec.transform(S).toarray())
    E = embed("bge", S)
    sims = {"tf": {}, "bge": {}}
    sims["tf"]["cap"] = TS @ _unit(vec.transform([a.figures[f] for f in figs]).toarray()).T
    sims["bge"]["cap"] = E @ embed("bge", [a.figures[f] for f in figs], is_query=True).T
    sims["tf"]["topic"] = TS @ _unit(vec.transform([topic]).toarray()).T
    sims["bge"]["topic"] = E @ embed("bge", [topic]).T
    for enc, M in (("tf", TS), ("bge", E)):
        out = np.zeros((n, len(figs)))
        for j, f in enumerate(figs):
            ids = ment[f]
            if not ids:
                out[:, j] = sims[enc]["cap"][:, j]
                continue
            c = M[ids].sum(0)
            out[:, j] = M @ (c / (np.linalg.norm(c) + 1e-12))
            for i in ids:  # leave-one-out for the mention sentences themselves
                rest = c - M[i]
                out[i, j] = float(M[i] @ (rest / (np.linalg.norm(rest) + 1e-12))) if len(ids) > 1 else np.nan
        sims[enc]["ment"] = out
    return figs, ment, sims


def add(D):
    """add round-3 keys to every candidate row of D (an experiments.Data)."""
    for aid, a in D.A.items():
        rows = [D.rows[k] for k in D.by_art.get(aid, [])]
        if not rows:
            continue
        figs, ment, sims = article_sims(a)
        n = len(a.sentences)
        for f in figs:
            j = figs.index(f)
            ids = ment[f]
            da = np.full(n, 999)
            db = np.full(n, 999)
            for i in range(n):
                before = [m for m in ids if m < i]
                after = [m for m in ids if m > i]
                if before:
                    da[i] = i - max(before)
                if after:
                    db[i] = min(after) - i
            local = [i for i in range(n) if f not in a.cue[i] and (da[i] <= LOCAL_A or db[i] <= LOCAL_B)]
            z = {}
            for enc in ENCS:
                for q in QUERIES:
                    x = sims[enc][q][:, 0 if q == "topic" else j]
                    mu_all, sd_all = np.nanmean(x), np.nanstd(x) + 1e-9
                    xm = x[ids] if ids else np.array([])
                    xm = xm[~np.isnan(xm)]
                    mu_m = xm.mean() if len(xm) else mu_all
                    xl = x[local] if local else x
                    mu_l, sd_l = np.nanmean(xl), np.nanstd(xl) + 1e-9
                    z[(enc, q, "all")] = (x - mu_all) / sd_all
                    z[(enc, q, "ment")] = (x - mu_m) / sd_all
                    z[(enc, q, "local")] = (x - mu_l) / sd_l
            for r in rows:
                if r["fig"] != f:
                    continue
                i = r["i"]
                r["da"], r["db"] = int(da[i]), int(db[i])
                for (enc, q, R), v in z.items():
                    r[f"z_{enc}_{q}_{R}"] = float(v[i])
                # chain: sentences strictly between i and the mention it is attached to (closer side wins)
                if da[i] <= db[i] and da[i] < 999:
                    r["chain"] = list(range(i - da[i] + 1, i))
                elif db[i] < 999:
                    r["chain"] = list(range(i + 1, i + db[i]))
                else:
                    r["chain"] = []
                for k in range(1, 6):
                    r[f"da{k}"] = int(da[i] == k)
                for k in range(1, 4):
                    r[f"db{k}"] = int(db[i] == k)
                r["far_disc"] = int(min(da[i], db[i]) > 5 and r["sec_discussion"] == 1 and r["same_section"] == 0)
    return D


def cache_path():
    return os.path.join(HERE, "features3_GS1.json")


if __name__ == "__main__":
    import experiments as E
    D = add(E.Data())
    keep = [k for k in D.rows[0] if k.startswith(("z_", "da", "db", "chain", "far_disc"))]
    json.dump([{**{k: r[k] for k in ("art", "fig", "i")}, **{k: r.get(k) for k in keep}} for r in D.rows],
              open(cache_path(), "w"))
    print("rows", len(D.rows), "keys", len(keep))
