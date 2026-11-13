"""Phase 7: score a whole-paper model with NO rules.

Scores: {"art|fig|i": logit} for every sentence of the paper (citing sentences included), out-of-fold.
Prediction for figure F = sentences whose per-article z-score >= t; nothing is added automatically.
t chosen by leave-one-article-out over the other 12 articles (objective: per-figure mean F1).
Reports per-figure mean P/R/F1 (all 67 figures) and breakdowns: citing sentences, window (<=3), far (>3),
vs B0f (which always includes citing sentences).  usage: python eval/figsum/phase7.py SCORES.json
"""
from __future__ import annotations

import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import experiments as E  # noqa: E402
import experiments2 as E2  # noqa: E402
import phase2 as P  # noqa: E402
import phase3 as P3  # noqa: E402

ZG = [float(x) for x in np.arange(-1.0, 4.01, 0.1)]


def zscores(path):
    S = json.load(open(path))
    by = {}
    for k, v in S.items():
        by.setdefault(k.split("|")[0], []).append(v)
    st = {a: (np.mean(v), np.std(v) + 1e-9) for a, v in by.items()}
    return {k: (v - st[k.split("|")[0]][0]) / st[k.split("|")[0]][1] for k, v in S.items()}


def per_fig(D, arts, sel):
    """sel: set of (art, fig, i). Returns per-figure F1 dict and aggregate P/R/F1 + breakdown counts."""
    P_, R_, F_, per = [], [], [], {}
    br = {k: [0, 0, 0] for k in ("cites", "window", "far")}   # tp, selected, gold
    for a in arts:
        A = D.A[a]
        for f in D.figs[a]:
            gold = A.gold[E.GS][f]
            ms = A._ment.get(f, [])
            s = {i for (x, g, i) in sel if x == a and g == f}
            tp = len(s & gold)
            p = tp / len(s) if s else 0.0
            r = tp / len(gold)
            F = 2 * p * r / (p + r) if p + r else 0.0
            P_.append(p), R_.append(r), F_.append(F)
            per[(a, f)] = F

            def kind(i):
                if f in A.cue[i]:
                    return "cites"
                return "far" if min(abs(i - m) for m in ms) > 3 else "window"
            for i in s:
                br[kind(i)][1] += 1
                br[kind(i)][0] += i in gold
            for i in gold:
                br[kind(i)][2] += 1
    return per, {"P": statistics.mean(P_), "R": statistics.mean(R_), "F1": statistics.mean(F_), "br": br}


def main():
    path = sys.argv[1]
    D = P3.load()
    Z = zscores(path)
    allA = D.split["dev"] + D.split["test"]

    def select(arts, t):
        return {tuple([k.split("|")[0], int(k.split("|")[1]), int(k.split("|")[2])]) for k, z in Z.items()
                if k.split("|")[0] in arts and z >= t}

    chosen, ts = set(), []
    for h in allA:
        tr = [x for x in allA if x != h]
        t = max(ZG, key=lambda t: per_fig(D, tr, select(tr, t))[1]["F1"])
        ts.append(round(t, 2))
        chosen |= select([h], t)
    per_m, agg = per_fig(D, allA, chosen)
    b0f = P3.keys(D, allA, lambda r: r["_b0f"])
    cites = {(a, f, i) for a in allA for f in D.figs[a] for i in D.A[a]._ment.get(f, [])}
    per_b, aggb = per_fig(D, allA, b0f | cites)
    m, lo, hi = E2.bootstrap(per_b, per_m)
    print(f"thresholds per held-out article (z): {ts}")
    for name, g in (("B0f (rule; citing sentences always included)", aggb), ("MODEL ALONE (whole paper, no rules)", agg)):
        b = g["br"]
        print(f"{name:46s} P {g['P']:.3f} R {g['R']:.3f} F1 {g['F1']:.3f}")
        for k in ("cites", "window", "far"):
            tp, s, gd = b[k]
            print(f"     {k:7s} correct {tp:3d} / selected {s:4d} / gold {gd:3d}   P {tp / max(1, s):.3f}  R {tp / max(1, gd):.3f}")
    print(f"MODEL vs B0f: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  (67 figures, all 13 articles, everything out-of-fold)")
    from sklearn.metrics import roc_auc_score
    for lab, cond in (("all pairs", lambda k: True), ("far pairs", None), ("window pairs", None)):
        pass
    keys = list(Z)
    y, s, kinds = [], [], []
    for k in keys:
        a, f, i = k.split("|")
        f, i = int(f), int(i)
        A = D.A[a]
        y.append(int(i in A.gold[E.GS][f]))
        s.append(Z[k])
        ms = A._ment.get(f, [])
        kinds.append("cites" if f in A.cue[i] else ("far" if min(abs(i - m) for m in ms) > 3 else "window"))
    y, s, kinds = np.array(y), np.array(s), np.array(kinds)
    for k in ("window", "far"):
        mk = kinds == k
        print(f"AUC {k:6s} pairs n={mk.sum():5d} gold={y[mk].sum():3d}  {roc_auc_score(y[mk], s[mk]):.3f}")
    print(f"AUC all pairs n={len(y)}  {roc_auc_score(y, s):.3f}")


if __name__ == "__main__":
    main()
