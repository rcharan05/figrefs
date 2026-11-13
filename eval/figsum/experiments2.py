"""Round 2 of the implicit experiments (same protocol as experiments.py: nested leave-one-article-out on dev).

New ideas
  figure-level features   how often the figure is named, caption length, number of figures in the paper
  stacking (collective    pass 1 = LR; pass 2 adds pass-1 probabilities of the previous / next sentence for the
  classification)         same figure, so a sentence next to a likely-relevant sentence is pushed up (annotators
                          mark contiguous runs)
  window shapes           LR restricted to d<=2 / d<=3 / d<=3 & same section / d<=5
  per-figure top-k        take the k best candidates of each figure (k tuned) instead of a global threshold
Also: paired bootstrap over figures (95% interval) of each method's mean F1 minus the adjacent-sentence rule.
usage: python eval/figsum/experiments2.py dev | test NAME...
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import experiments as E  # noqa: E402

ALL = sum(E.G.values(), [])
FIGF = ["log_n_ment", "log_cap_len", "log_n_figs"]


def add_fig_features(D):
    for aid, a in D.A.items():
        for k in D.by_art.get(aid, []):
            r = D.rows[k]
            f = r["fig"]
            r["log_n_ment"] = math.log1p(sum(1 for c in a.cue if f in c))
            r["log_cap_len"] = math.log1p(len(a.figures[f].split()))
            r["log_n_figs"] = math.log1p(len(a.figures))
    # neighbour index: (art, fig, i) -> row
    D.key = {(r["art"], r["fig"], r["i"]): k for k, r in enumerate(D.rows)}


def per_fig_f1(D, arts, chosen):
    out = {}
    for aid in arts:
        a = D.A[aid]
        for f in D.figs[aid]:
            gold = a.gold[E.GS][f]
            cue = {i for i, c in enumerate(a.cue) if f in c}
            sel = cue | {i for (x, g, i) in chosen if x == aid and g == f}
            tp = len(sel & gold)
            out[(aid, f)] = 2 * tp / (len(sel) + len(gold)) if sel else 0.0
    return out


def window_ok(r, w):
    if w is None:
        return True
    if callable(w):
        return w(r)
    if w == "d3sec":
        return r["d_sent"] <= 3 and r["same_section"] == 1
    return r["d_sent"] <= w


def lr_fit_predict(D, train, test, feats, window=None, kind="lr"):
    def idx_of(arts):
        return [k for a in arts for k in D.by_art[a] if window_ok(D.rows[k], window)]
    m = E.make_model(kind)
    tr = idx_of(train)
    m.fit(D.X(tr, feats), D.y(tr))
    te = idx_of(test)
    return te, m.predict_proba(D.X(te, feats))[:, 1]


def oof(D, arts, feats, window=None, kind="lr"):
    """out-of-fold probabilities for every candidate of `arts` (LOAO inside arts)"""
    idx, sc = [], []
    for h in arts:
        i, s = lr_fit_predict(D, [x for x in arts if x != h], [h], feats, window, kind)
        idx += i
        sc += list(s)
    return idx, np.array(sc)


def neighbour_feats(D, idx, probs):
    p = {k: v for k, v in zip(idx, probs)}
    out = []
    for k in idx:
        r = D.rows[k]
        prv = D.key.get((r["art"], r["fig"], r["i"] - 1))
        nxt = D.key.get((r["art"], r["fig"], r["i"] + 1))
        a = D.A[r["art"]]
        # a neighbour that NAMES the figure is not a candidate: treat it as certain (probability 1)
        pp = p.get(prv, 1.0 if r["i"] - 1 >= 0 and r["fig"] in a.cue[r["i"] - 1] else 0.0)
        pn = p.get(nxt, 1.0 if r["i"] + 1 < len(a.cue) and r["fig"] in a.cue[r["i"] + 1] else 0.0)
        out.append([p[k], pp, pn, max(pp, pn)])
    return np.array(out)


def m_stacked(feats, window=None):
    def fp(D, train, test):
        def stage(trn, tst):
            i1, p1 = oof(D, trn, feats, window)                     # pass-1 OOF on train
            j1, q1 = lr_fit_predict(D, trn, tst, feats, window)     # pass-1 on test
            Xtr = np.hstack([D.X(i1, feats), neighbour_feats(D, i1, p1)])
            Xte = np.hstack([D.X(j1, feats), neighbour_feats(D, j1, q1)])
            m = E.make_model("lr")
            m.fit(Xtr, D.y(i1))
            return j1, m.predict_proba(Xte)[:, 1]
        # threshold from an inner LOAO of the whole two-pass procedure
        inner_i, inner_s = [], []
        for h in train:
            i, s = stage([x for x in train if x != h], [h])
            inner_i += i
            inner_s += list(s)
        t = E.best_threshold(D, train, inner_i, inner_s)
        i, s = stage(train, test)
        return E.select_by_score(D, i, s, t), {"t": t}
    return fp


def m_topk(feats, window=None, ks=range(0, 9)):
    def fp(D, train, test):
        i, s = oof(D, train, feats, window)
        best = max(ks, key=lambda K: E.evaluate(D, train, topk_select(D, i, s, K))["mean_f1"])
        j, q = lr_fit_predict(D, train, test, feats, window)
        return topk_select(D, j, q, best), {"k": best}
    return fp


def topk_select(D, idx, sc, K):
    by = {}
    for k, s in zip(idx, sc):
        r = D.rows[k]
        by.setdefault((r["art"], r["fig"]), []).append((s, r["i"]))
    out = set()
    for (a, f), lst in by.items():
        for s, i in sorted(lst, reverse=True)[:K]:
            out.add((a, f, i))
    return out


def m_lr(feats, window=None, kind="lr"):
    def fp(D, train, test):
        i, s = oof(D, train, feats, window, kind)
        t = E.best_threshold(D, train, i, s)
        j, q = lr_fit_predict(D, train, test, feats, window, kind)
        return E.select_by_score(D, j, q, t), {"t": t}
    return fp


def methods():
    return {
        "Adjacent sentence only (d=1)": E.m_fixed(lambda r: r["d_sent"] == 1),
        "Paragraph rule": E.m_fixed(lambda r: r["same_para"] == 1),
        "Hand-set: old score, no penalties": lambda D, tr, te: E.m_handset(D, tr, te, grid=[(1000, 999, 0)]),
        "Hand-set: + distance + topic penalties": E.m_handset,
        "LR all (no window)": m_lr(ALL),
        "LR all, window d<=3": m_lr(ALL, 3),
        "GBM all": m_lr(ALL, None, "gbm"),
        "LR all + figure feats": m_lr(ALL + FIGF),
        "LR all + figure feats, d<=2": m_lr(ALL + FIGF, 2),
        "LR all + figure feats, d<=3": m_lr(ALL + FIGF, 3),
        "LR all + figure feats, d<=3 same section": m_lr(ALL + FIGF, "d3sec"),
        "LR all + figure feats, d<=5": m_lr(ALL + FIGF, 5),
        "GBM all + figure feats, d<=3": m_lr(ALL + FIGF, 3, "gbm"),
        "Stacked LR (neighbour context)": m_stacked(ALL + FIGF),
        "Stacked LR (neighbour context), d<=3": m_stacked(ALL + FIGF, 3),
        "Top-k per figure (LR all + figure feats)": m_topk(ALL + FIGF),
        "Top-k per figure, d<=3": m_topk(ALL + FIGF, 3),
    }


def bootstrap(base: dict, other: dict, n=5000, seed=0):
    keys = sorted(base)
    d = np.array([other[k] - base[k] for k in keys])
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    D = E.Data()
    add_fig_features(D)
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    dev, test = D.split["dev"], D.split["test"]
    M = methods()
    res, per = {}, {}
    if mode == "dev":
        print(f"== DEV round 2, nested LOAO over {len(dev)} articles")
        for name, fp in M.items():
            chosen = set()
            for h in dev:
                out = fp(D, [x for x in dev if x != h], [h])
                chosen |= out[0] if isinstance(out, tuple) else out
            r = E.evaluate(D, dev, chosen)
            per[name] = per_fig_f1(D, dev, chosen)
            res[name] = r
            print(E.fmt(name, r), flush=True)
        base = per["Adjacent sentence only (d=1)"]
        print("\n-- paired bootstrap over figures: mean F1 difference vs adjacent rule [95% interval]")
        for name in M:
            m, lo, hi = bootstrap(base, per[name])
            res[name]["vs_adjacent"] = [m, lo, hi]
            print(f"   {name:44s} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
        json.dump(res, open(os.path.join(E.HERE, "exp2_dev.json"), "w"), indent=1)
    else:
        names = sys.argv[2:] or list(M)
        print(f"== TEST ({len(test)} articles, {sum(len(D.figs[a]) for a in test)} figures); fitted on all dev")
        for name in ("FigSum+ SurfaceCue", "FigSum+ Paragraph", "FigSum+ Hybrid", "FigSum+ TFIDF"):
            r = E.figsum_baseline(D, name, test)
            res[name] = r
            print(E.fmt(name + " [published system]", r))
        for name in names:
            out = M[name](D, dev, test)
            chosen, params = out if isinstance(out, tuple) else (out, {})
            r = E.evaluate(D, test, chosen)
            r["params"] = {k: float(v) if hasattr(v, "item") else v for k, v in params.items()}
            per[name] = per_fig_f1(D, test, chosen)
            res[name] = r
            print(E.fmt(name, r), r["params"], flush=True)
        json.dump(res, open(os.path.join(E.HERE, "exp2_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
