"""Implicit-reference experiments on FigSum+ GS1 (dev = 9 articles, test = 4 held-out articles).

Every method outputs, per figure: the sentences that name it (SurfaceCue, shared by all methods) plus the
candidate sentences it selects. Scored like the paper: per-figure P/R/F1 averaged over figures ("mean F1"),
plus precision/recall/F1 on the implicit sentences alone (pooled over figures).

Honest dev estimate = nested leave-one-article-out (LOAO): for each dev article, every choice (grid parameters,
model weights, threshold) is made on the other 8 articles only, then applied to the held-out article.
usage: python eval/figsum/experiments.py dev            # all experiments, LOAO on dev
       python eval/figsum/experiments.py test NAME...   # final: fit on all dev, score the test articles once
"""
from __future__ import annotations

import itertools
import json
import math
import os
import statistics
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data import load_all  # noqa: E402
from systems import SYSTEMS, Ctx  # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
GS = "GS1"

G = {
    "dist": ["log_d", "same_para", "adjacent", "after", "same_section", "sandwich", "d_para_log"],
    "own": ["own_nearest", "own_prev", "own_next", "other_closer", "names_other"],
    "sec": ["sec_results", "sec_discussion", "sec_methods", "sec_intro", "sec_abstract"],
    "surf": ["anaphora", "subfig_letter", "visual", "offtarget", "log_len"],
    "sem_tf": ["tf_cap", "tf_ment", "tf_cap_z", "tf_ment_z"],
    "sem_dense": ["bge_cap", "bge_ment", "bge_cap_z", "bge_ment_z", "ce_cap", "ce_cap_z"],
    "topic": ["tf_topic", "tf_centroid", "spec_tf", "bge_topic", "bge_centroid", "spec_bge", "ce_topic", "spec_ce"],
    "graph": ["lexrank", "lexrank_rank", "same_tile"],
}


# ---------------------------------------------------------------- data

class Data:
    def __init__(self):
        arts, _ = load_all(os.path.join(HERE, "gold"), "/home/charan/bench/figsum")
        self.A = {a.id: a for a in arts}
        self.split = json.load(open(os.path.join(HERE, "split.json")))
        self.rows = json.load(open(os.path.join(HERE, f"features_{GS}.json")))
        for r in self.rows:
            r["d_para_log"] = math.log1p(r["d_para"])
        self.by_art = {}
        for k, r in enumerate(self.rows):
            self.by_art.setdefault(r["art"], []).append(k)
        self.figs = {aid: sorted(f for f in a.gold[GS] if f in a.figures and a.gold[GS][f]) for aid, a in self.A.items()}

    def X(self, idx, feats):
        return np.array([[self.rows[k][f] for f in feats] for k in idx], dtype=float)

    def y(self, idx):
        return np.array([self.rows[k]["y"] for k in idx])


def evaluate(D: Data, arts: list[str], chosen: set[tuple[str, int, int]]) -> dict:
    """chosen = {(art, fig, sentence)} candidate selections; cue sentences are always added."""
    per, tp_i = [], 0
    n_sel_i = n_gold_i = 0
    for aid in arts:
        a = D.A[aid]
        for f in D.figs[aid]:
            gold = a.gold[GS][f]
            cue = {i for i, c in enumerate(a.cue) if f in c}
            imp = {i for (x, g, i) in chosen if x == aid and g == f}
            sel = cue | imp
            tp = len(sel & gold)
            P = tp / len(sel) if sel else 0
            R = tp / len(gold)
            per.append(2 * P * R / (P + R) if P + R else 0)
            ig = gold - cue
            tp_i += len(imp & ig)
            n_sel_i += len(imp)
            n_gold_i += len(ig)
    Pi, Ri = tp_i / max(1, n_sel_i), tp_i / max(1, n_gold_i)
    return {"mean_f1": statistics.mean(per), "imp_p": Pi, "imp_r": Ri,
            "imp_f1": 2 * Pi * Ri / (Pi + Ri) if Pi + Ri else 0, "imp_sel": n_sel_i, "figs": len(per)}


def select_by_score(D, idx, scores, t):
    return {(D.rows[k]["art"], D.rows[k]["fig"], D.rows[k]["i"]) for k, s in zip(idx, scores) if s >= t}


T_GRID = [round(x, 3) for x in np.linspace(0.05, 0.95, 37)]


def fast_mean_f1(D, arts, idx, scores, thresholds) -> np.ndarray:
    """mean per-figure F1 for every threshold at once. Per figure, F1 = 2 TP / (|selected| + |gold|), where
    selected = cue sentences + candidates scoring >= t. Candidates not in idx are never selected."""
    scores = np.asarray(scores, float)
    th = np.asarray(thresholds, float)
    by_fig: dict[tuple[str, int], list[int]] = {}
    for pos, k in enumerate(idx):
        by_fig.setdefault((D.rows[k]["art"], D.rows[k]["fig"]), []).append(pos)
    tot = np.zeros(len(th))
    nf = 0
    for aid in arts:
        a = D.A[aid]
        for f in D.figs[aid]:
            gold = a.gold[GS][f]
            cue = {i for i, c in enumerate(a.cue) if f in c}
            tp0, n0, g = len(cue & gold), len(cue), len(gold)
            p = by_fig.get((aid, f), [])
            if p:
                s = scores[p]
                y = np.array([D.rows[idx[q]]["y"] for q in p])
                sel = (s[None, :] >= th[:, None])
                tp = tp0 + (sel & (y[None, :] == 1)).sum(1)
                n = n0 + sel.sum(1)
            else:
                tp, n = np.full(len(th), tp0), np.full(len(th), n0)
            tot += np.where(n > 0, 2 * tp / (n + g), 0.0)
            nf += 1
    return tot / max(1, nf)


def best_threshold(D, arts, idx, scores):
    m = fast_mean_f1(D, arts, idx, scores, T_GRID)
    return T_GRID[int(np.argmax(m))]


def loao(D: Data, arts: list[str], fit_predict) -> dict:
    """fit_predict(train_arts, test_arts) -> chosen set for test_arts. Returns pooled evaluation on all arts."""
    chosen = set()
    for held in arts:
        train = [x for x in arts if x != held]
        chosen |= fit_predict(D, train, [held])
    return evaluate(D, arts, chosen)


# ---------------------------------------------------------------- methods

def m_fixed(sel_fn):
    """a method with no parameters: sel_fn(D, row) -> bool"""
    def fp(D, train, test):
        return {(D.rows[k]["art"], D.rows[k]["fig"], D.rows[k]["i"]) for a in test for k in D.by_art[a] if sel_fn(D.rows[k])}
    return fp


def m_window(D, train, test, ks=(1, 2, 3, 4, 5, 6, 8)):
    """pure distance rule: select candidates within k sentences of a mention (k chosen on train)."""
    idx = [k for a in train for k in D.by_art[a]]
    sc = [-D.rows[k]["d_sent"] for k in idx]
    m = fast_mean_f1(D, train, idx, sc, [-K - 0.5 for K in ks])
    best = ks[int(np.argmax(m))]
    return {(D.rows[k]["art"], D.rows[k]["fig"], D.rows[k]["i"]) for a in test for k in D.by_art[a]
            if D.rows[k]["d_sent"] <= best}, {"k": best}


def handset_score(r, tau, dmax, lam, sem="mix"):
    """User's two penalties on top of the old figrefs semantic score.
    sem       0.5 * sigmoid(TF-IDF z + BGE z across figures) + 0.5 * reranker(caption, sentence)
    distance  multiply by exp(-(d-1)/tau), and 0 beyond dmax sentences
    topic     subtract lam * max(0, sim(sentence, article topic) - sim(sentence, caption))   [BGE cosine]"""
    s = 0.5 / (1 + math.exp(-(r["tf_cap_z"] + r["bge_cap_z"]))) + 0.5 * r["ce_cap"]
    d = r["d_sent"]
    dist = 0.0 if d > dmax else math.exp(-(d - 1) / tau)
    topic = max(0.0, r["bge_topic"] - r["bge_cap"])
    return s * dist - lam * topic


HS_GRID = list(itertools.product([0.5, 1, 2, 4, 8, 1000], [1, 2, 3, 5, 8, 20, 999], [0, 0.5, 1, 2, 4]))


def m_handset(D, train, test, grid=HS_GRID):
    idx = [k for a in train for k in D.by_art[a]]
    best = (-1, None, None)
    tg = T_GRID + [0.0, -0.05]
    for tau, dmax, lam in grid:
        sc = [handset_score(D.rows[k], tau, dmax, lam) for k in idx]
        m = fast_mean_f1(D, train, idx, sc, tg)
        j = int(np.argmax(m))
        if m[j] > best[0]:
            best = (m[j], (tau, dmax, lam), tg[j])
    (tau, dmax, lam), t = best[1], best[2]
    tidx = [k for a in test for k in D.by_art[a]]
    sc = [handset_score(D.rows[k], tau, dmax, lam) for k in tidx]
    return select_by_score(D, tidx, sc, t), {"tau": tau, "dmax": dmax, "lam": lam, "t": t}


def make_model(kind):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=30,
                                              l2_regularization=1.0, class_weight="balanced")
    raise ValueError(kind)


def m_learned(feats, kind="lr", dmax=None):
    """learned combiner; threshold picked by an inner LOAO on the training articles.
    dmax: optional hard window (candidates beyond it are never selected)."""
    def fp(D, train, test):
        def fit(arts):
            idx = [k for a in arts for k in D.by_art[a] if dmax is None or D.rows[k]["d_sent"] <= dmax]
            m = make_model(kind)
            m.fit(D.X(idx, feats), D.y(idx))
            return m

        def predict(m, arts):
            idx = [k for a in arts for k in D.by_art[a] if dmax is None or D.rows[k]["d_sent"] <= dmax]
            return idx, m.predict_proba(D.X(idx, feats))[:, 1]
        # inner LOAO predictions on train -> threshold
        inner_idx, inner_sc = [], []
        for h in train:
            m = fit([x for x in train if x != h])
            i, s = predict(m, [h])
            inner_idx += i
            inner_sc += list(s)
        t = best_threshold(D, train, inner_idx, inner_sc)
        m = fit(train)
        idx, sc = predict(m, test)
        return select_by_score(D, idx, sc, t), {"t": t}
    return fp


def wrap(fp):
    """normalise fit_predict outputs to the chosen set"""
    def g(D, train, test):
        out = fp(D, train, test)
        return out[0] if isinstance(out, tuple) else out
    return g


# ---------------------------------------------------------------- FigSum+ baselines on the same articles

def figsum_baseline(D, name, arts):
    fn = SYSTEMS[name]
    chosen = set()
    for aid in arts:
        a = D.A[aid]
        ctx = Ctx(a)
        for f in D.figs[aid]:
            cue = {i for i, c in enumerate(a.cue) if f in c}
            for i in fn(ctx, f, 8):
                if i not in cue:
                    chosen.add((aid, f, i))
    return evaluate(D, arts, chosen)


def methods():
    allf = sum(G.values(), [])
    M = {
        "SurfaceCue only (no implicit)": m_fixed(lambda r: False),
        "Paragraph rule (FigSum+)": m_fixed(lambda r: r["same_para"] == 1),
        "Adjacent sentence only (d=1)": m_fixed(lambda r: r["d_sent"] == 1),
        "Window, k tuned": m_window,
        "Hand-set: old score, no penalties": None,  # filled below
        "Hand-set: + distance + topic penalties": m_handset,
        "LR semantic only": m_learned(G["sem_tf"] + G["sem_dense"]),
        "LR semantic + distance": m_learned(G["sem_tf"] + G["sem_dense"] + G["dist"]),
        "LR semantic + distance + topic": m_learned(G["sem_tf"] + G["sem_dense"] + G["dist"] + G["topic"]),
        "LR + ownership": m_learned(G["sem_tf"] + G["sem_dense"] + G["dist"] + G["topic"] + G["own"]),
        "LR + section": m_learned(G["sem_tf"] + G["sem_dense"] + G["dist"] + G["topic"] + G["own"] + G["sec"]),
        "LR + surface cues": m_learned(G["sem_tf"] + G["sem_dense"] + G["dist"] + G["topic"] + G["own"] + G["sec"] + G["surf"]),
        "LR all (+ LexRank, TextTiling)": m_learned(allf),
        "LR all, window d<=3": m_learned(allf, dmax=3),
        "LR all, window d<=8": m_learned(allf, dmax=8),
        "LR all, TF-IDF only (no neural)": m_learned([f for f in allf if not f.startswith(("bge", "ce", "spec_bge", "spec_ce"))]),
        "GBM all": m_learned(allf, "gbm"),
        "GBM all, window d<=8": m_learned(allf, "gbm", dmax=8),
    }
    M["Hand-set: old score, no penalties"] = lambda D, tr, te: m_handset(D, tr, te, grid=[(1000, 999, 0)])
    return M


def ablation(D, arts):
    """all-features LR minus one group at a time"""
    allf = sum(G.values(), [])
    out = {}
    for g, fs in G.items():
        out[f"LR all minus {g}"] = loao(D, arts, wrap(m_learned([f for f in allf if f not in fs])))
    return out


def coefficients(D, arts, feats):
    m = make_model("lr")
    idx = [k for a in arts for k in D.by_art[a]]
    m.fit(D.X(idx, feats), D.y(idx))
    w = m[-1].coef_[0]
    return sorted(zip(feats, w), key=lambda x: -abs(x[1]))


def fmt(name, r):
    return f"{name:42s} meanF1 {r['mean_f1']:.3f} | implicit P {r['imp_p']:.3f} R {r['imp_r']:.3f} F1 {r['imp_f1']:.3f} (sel {r['imp_sel']})"


def main():
    D = Data()
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    dev, test = D.split["dev"], D.split["test"]
    res = {}
    if mode == "dev":
        print(f"== DEV, nested leave-one-article-out over {len(dev)} articles, {sum(len(D.figs[a]) for a in dev)} figures")
        for name in ("FigSum+ Paragraph", "FigSum+ Hybrid", "FigSum+ TFIDF"):
            r = figsum_baseline(D, name, dev)
            res[name] = r
            print(fmt(name + " [as published system]", r), flush=True)
        for name, fp in methods().items():
            r = loao(D, dev, wrap(fp))
            res[name] = r
            print(fmt(name, r), flush=True)
        print("\n-- ablation (all-features LR minus one group)")
        for name, r in ablation(D, dev).items():
            res[name] = r
            print(fmt(name, r), flush=True)
        print("\n-- hand-set parameters chosen on all dev:", m_handset(D, dev, dev)[1])
        print("-- window k chosen on all dev:", m_window(D, dev, dev)[1])
        print("\n-- LR (all features) standardised coefficients, fit on all dev:")
        for f, w in coefficients(D, dev, sum(G.values(), []))[:20]:
            print(f"   {f:16s} {w:+.3f}")
        json.dump(res, open(os.path.join(HERE, "exp_dev.json"), "w"), indent=1)
    else:
        names = sys.argv[2:]
        M = methods()
        print(f"== TEST ({len(test)} articles, {sum(len(D.figs[a]) for a in test)} figures); fit on all {len(dev)} dev articles")
        for name in ("FigSum+ Paragraph", "FigSum+ Hybrid", "FigSum+ SurfaceCue", "FigSum+ TFIDF"):
            r = figsum_baseline(D, name, test)
            res[name] = r
            print(fmt(name + " [published system]", r))
        for name in names:
            out = M[name](D, dev, test)
            chosen, params = out if isinstance(out, tuple) else (out, {})
            r = evaluate(D, test, chosen)
            r["params"] = params
            res[name] = r
            print(fmt(name, r), params)
        json.dump(res, open(os.path.join(HERE, "exp_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
