"""Feature table for implicit-reference classification on FigSum+.

One row per (article, figure, sentence) where the sentence does NOT name the figure ("candidate").
Label = sentence is in the gold set for that figure.  Features (all computed without the gold):

distance        d_sent, log_d, d_para, same_para, adjacent, after, same_section, sandwich
ownership       own_nearest (nearest figure mention in the text is this figure), own_prev, own_next,
                other_closer (another figure is mentioned closer), names_other
section         sec_results / sec_discussion / sec_methods / sec_intro / sec_abstract
surface         anaphora (opens with a back-reference), subfig_letter, visual words, offtarget
                ("data not shown", supplementary, tables), log_len
semantic        TF-IDF, BGE (bi-encoder) and bge-reranker (cross-encoder) similarity to the caption (*_cap),
                to the figure's own mention sentences (*_ment), and z-scored across the paper's figures (*_z)
topic penalty   similarity to the article's general topic: title + abstract + section headings (*_topic) and to
                the article centroid (*_centroid); specificity = caption sim - topic sim (spec_*)
graph           biased LexRank: personalised PageRank on the sentence-similarity graph, teleporting to the
                figure's mention sentences (lexrank, lexrank_rank)
segmentation    TextTiling: sentence is in the same topical segment as a mention (same_tile)
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data import load_all  # noqa: E402

from figrefs.implicit import ANAPHORA_RE, SUBFIG_BARE_RE, VISUAL_RE  # noqa: E402

OFFTARGET = re.compile(r"data not shown|not shown|supplement|additional file|figure s\d|fig\.? s\d|\btable\b|web site", re.I)
SEC = {"results": r"result", "discussion": r"discussion|conclusion", "methods": r"method|material|procedure",
       "intro": r"introduction|background", "abstract": r"^abstract$"}
HERE = os.path.dirname(__file__)


def _cos_rows(A, B):
    """cosine between rows of sparse/dense A and rows of B -> [len(A), len(B)]"""
    A = A.toarray() if hasattr(A, "toarray") else A
    B = B.toarray() if hasattr(B, "toarray") else B
    na = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    nb = np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
    return (A / na) @ (B / nb).T


def _z(M):
    if M.shape[1] < 2:
        return np.zeros_like(M)
    return (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-6)


def biased_lexrank(sim: np.ndarray, seeds: list[int], d: float = 0.85, edge: float = 0.1, iters: int = 60):
    """Personalised PageRank (Otterbacher et al. 2005 'biased LexRank'): random walk on the sentence graph
    that jumps back to the figure's mention sentences with probability 1-d."""
    n = sim.shape[0]
    W = np.where(sim >= edge, sim, 0.0)
    np.fill_diagonal(W, 0)
    for i in range(n - 1):  # reading-order adjacency: text flows to the next / previous sentence
        W[i, i + 1] += 0.2
        W[i + 1, i] += 0.2
    rs = W.sum(1, keepdims=True)
    P = np.divide(W, rs, out=np.full_like(W, 1.0 / n), where=rs > 0)
    v = np.zeros(n)
    v[seeds] = 1.0 / len(seeds)
    r = v.copy()
    for _ in range(iters):
        r = (1 - d) * v + d * (r @ P)
    return r


def texttiling(sim: np.ndarray, sent_para: list[int], w: int = 3) -> list[int]:
    """TextTiling (Hearst 1997) on sentences: cohesion between the w sentences before and after each gap,
    depth score of each valley, boundary where depth > mean - std/2. Returns a segment id per sentence."""
    n = sim.shape[0]
    if n < 2 * w + 2:
        return [0] * n
    gaps = []
    for g in range(1, n):
        a, b = range(max(0, g - w), g), range(g, min(n, g + w))
        gaps.append(float(np.mean(sim[np.ix_(list(a), list(b))])))
    depth = []
    for k, c in enumerate(gaps):
        lpk = max(gaps[:k + 1])
        rpk = max(gaps[k:])
        depth.append((lpk - c) + (rpk - c))
    thr = np.mean(depth) - np.std(depth) / 2
    seg, out = 0, [0]
    for k in range(1, n):
        if depth[k - 1] > thr and (k < 2 or depth[k - 1] >= max(depth[max(0, k - 3):k + 2])):
            seg += 1
        out.append(seg)
    return out


def article_features(a, gs: str, dense: bool = True, cross: bool = True) -> list[dict]:
    figs = sorted(f for f in a.gold.get(gs, {}) if f in a.figures)
    allfigs = sorted(a.figures)
    S = a.sentences
    n = len(S)
    caps = [a.figures[f] for f in allfigs]
    ment_txt = {f: " ".join(S[i] for i in range(n) if f in a.cue[i]) for f in allfigs}
    heads = " ".join(dict.fromkeys(x for x in a.top_section + a.para_section if x))
    abstract = " ".join(S[i] for i in range(n) if a.top_section[a.sent_para[i]] == "Abstract")
    topic = " ".join([a.title, abstract, heads])

    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9\-]+\b")
    vec.fit(S + caps)
    TS = vec.transform(S)
    tf_cap = _cos_rows(TS, vec.transform(caps))                                   # n x F
    tf_ment = _cos_rows(TS, vec.transform([ment_txt[f] or a.figures[f] for f in allfigs]))
    tf_topic = _cos_rows(TS, vec.transform([topic]))[:, 0]
    cen = np.asarray(TS.mean(0))
    tf_cen = _cos_rows(TS, cen)[:, 0]
    tf_sim = _cos_rows(TS, TS)

    if dense:
        from figrefs.dense import embed
        E = embed("bge", S)
        Q = embed("bge", caps, is_query=True)
        Mq = embed("bge", [ment_txt[f] or a.figures[f] for f in allfigs])
        Tq = embed("bge", [a.title + ". " + abstract])
        bge_cap, bge_ment = E @ Q.T, E @ Mq.T
        bge_topic = (E @ Tq.T)[:, 0]
        c = E.mean(0)
        bge_cen = E @ (c / np.linalg.norm(c))
        sim_graph = np.clip(E @ E.T, 0, 1)
    else:
        bge_cap = bge_ment = np.zeros_like(tf_cap)
        bge_topic = bge_cen = np.zeros(n)
        sim_graph = tf_sim
    if cross:
        from figrefs.dense import cross_cached
        ce_cap = np.zeros_like(tf_cap)
        for j, f in enumerate(allfigs):
            ce_cap[:, j] = cross_cached("bge-reranker", [(a.figures[f][:1000], s) for s in S])
        ce_topic = cross_cached("bge-reranker", [((a.title + ". " + abstract)[:1000], s) for s in S])
    else:
        ce_cap = np.zeros_like(tf_cap)
        ce_topic = np.zeros(n)

    Z = {"tf_cap": _z(tf_cap), "tf_ment": _z(tf_ment), "bge_cap": _z(bge_cap), "bge_ment": _z(bge_ment), "ce_cap": _z(ce_cap)}
    tiles = texttiling(sim_graph, a.sent_para)
    mentions_of = {f: [i for i in range(n) if f in a.cue[i]] for f in allfigs}
    any_ment = [(i, f) for i in range(n) for f in a.cue[i]]
    rows = []
    for f in figs:
        j = allfigs.index(f)
        ment = mentions_of[f]
        gold = a.gold[gs][f]
        lr = biased_lexrank(sim_graph, ment) if ment else np.zeros(n)
        cand = [i for i in range(n) if f not in a.cue[i]]
        lr_c = lr[cand]
        order = {i: r for r, i in enumerate(sorted(cand, key=lambda i: -lr[i]))}
        mpar = {a.sent_para[i] for i in ment}
        msec = {a.top_section[p] for p in mpar}
        mtiles = {tiles[i] for i in ment}
        for i in cand:
            r = {"art": a.id, "fig": f, "i": i, "y": int(i in gold)}
            if ment:
                d = min(abs(i - m) for m in ment)
                near = min(ment, key=lambda m: abs(m - i))
                r["d_sent"] = d
                r["d_para"] = min(abs(a.sent_para[i] - p) for p in mpar)
                r["after"] = int(i > near)
                before = [m for m in ment if m < i]
                after = [m for m in ment if m > i]
                r["sandwich"] = int(bool(before) and bool(after) and
                                    a.top_section[a.sent_para[max(before)]] == a.top_section[a.sent_para[min(after)]]
                                    == a.top_section[a.sent_para[i]] and min(after) - max(before) <= 12)
            else:
                r["d_sent"], r["d_para"], r["after"], r["sandwich"] = 999, 999, 0, 0
            r["log_d"] = math.log1p(r["d_sent"])
            r["same_para"] = int(r["d_para"] == 0)
            r["adjacent"] = int(r["d_sent"] == 1)
            r["same_section"] = int(a.top_section[a.sent_para[i]] in msec)
            # ownership: which figure does the surrounding text "belong" to?
            others = [(abs(i - m), g) for m, g in any_ment if g != f]
            dmin_other = min((x for x, _ in others), default=999)
            r["other_closer"] = int(dmin_other < r["d_sent"])
            r["own_nearest"] = int(r["d_sent"] <= dmin_other and r["d_sent"] < 999)
            prev = [(m, g) for m, g in any_ment if m < i and a.top_section[a.sent_para[m]] == a.top_section[a.sent_para[i]]]
            nxt = [(m, g) for m, g in any_ment if m > i and a.top_section[a.sent_para[m]] == a.top_section[a.sent_para[i]]]
            r["own_prev"] = int(bool(prev) and max(prev)[1] == f) if prev else 0
            r["own_next"] = int(bool(nxt) and min(nxt)[1] == f) if nxt else 0
            r["names_other"] = int(bool(a.cue[i]))
            sec = a.top_section[a.sent_para[i]].lower()
            for k, pat in SEC.items():
                r[f"sec_{k}"] = int(bool(re.search(pat, sec)))
            s = S[i]
            r["anaphora"] = int(bool(ANAPHORA_RE.match(s)))
            r["subfig_letter"] = int(bool(SUBFIG_BARE_RE.search(s)))
            r["visual"] = int(bool(VISUAL_RE.search(s)))
            r["offtarget"] = int(bool(OFFTARGET.search(s)))
            r["log_len"] = math.log1p(len(s.split()))
            r["tf_cap"], r["tf_ment"] = float(tf_cap[i, j]), float(tf_ment[i, j])
            r["tf_cap_z"], r["tf_ment_z"] = float(Z["tf_cap"][i, j]), float(Z["tf_ment"][i, j])
            r["tf_topic"], r["tf_centroid"] = float(tf_topic[i]), float(tf_cen[i])
            r["spec_tf"] = r["tf_cap"] - r["tf_topic"]
            r["bge_cap"], r["bge_ment"] = float(bge_cap[i, j]), float(bge_ment[i, j])
            r["bge_cap_z"], r["bge_ment_z"] = float(Z["bge_cap"][i, j]), float(Z["bge_ment"][i, j])
            r["bge_topic"], r["bge_centroid"] = float(bge_topic[i]), float(bge_cen[i])
            r["spec_bge"] = r["bge_cap"] - r["bge_topic"]
            r["ce_cap"], r["ce_cap_z"] = float(ce_cap[i, j]), float(Z["ce_cap"][i, j])
            r["ce_topic"] = float(ce_topic[i])
            r["spec_ce"] = r["ce_cap"] - r["ce_topic"]
            r["lexrank"] = float(lr[i] / (lr_c.max() + 1e-12)) if ment else 0.0
            r["lexrank_rank"] = math.log1p(order[i])
            r["same_tile"] = int(tiles[i] in mtiles)
            rows.append(r)
    return rows


def build(gs: str = "GS1", out: str | None = None, dense=True, cross=True):
    arts, _ = load_all(os.path.join(HERE, "gold"), "/home/charan/bench/figsum")
    rows = []
    for a in arts:
        rows += article_features(a, gs, dense, cross)
        print(a.id, len(rows), flush=True)
    out = out or os.path.join(HERE, f"features_{gs}.json")
    json.dump(rows, open(out, "w"))
    return rows


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "GS1")
