"""Score sentence-selection systems on FigSumGS1 / FigSumGS2 like the FigSum+ paper.

Per figure: P = |sel & gold| / |sel|, R = |sel & gold| / |gold|, F1 (0 if no overlap). The paper's tables give
mean ± SD over figures, so `mean` is the per-figure average (figures with an empty selection get P = 0);
`micro` pools counts over all figures. n = 8 (GS1) and 11 (GS2), as in the paper.
"""
from __future__ import annotations

import statistics

PUBLISHED = {  # Tables 2 and 3 of the paper: (P, R, F1)
    "GS1": {"RandomSent": (.06, .06, .06), "RandomPara": (.04, .01, .01), "MEAD": (.05, .06, .05),
            "FigSum+ Similarity": (.28, .38, .30), "FigSum+ TFIDF": (.30, .34, .30),
            "FigSum+ SurfaceCue": (.96, .41, .54), "FigSum+ Paragraph": (.64, .82, .66),
            "FigSum+ Hybrid": (.67, .64, .62)},
    "GS2": {"RandomSent": (.08, .09, .08), "RandomPara": (.04, .01, .01), "MEAD": (.08, .07, .07),
            "FigSum+ Similarity": (.31, .28, .29), "FigSum+ TFIDF": (.27, .20, .22),
            "FigSum+ SurfaceCue": (.63, .16, .24), "FigSum+ Paragraph": (.51, .42, .41),
            "FigSum+ Hybrid": (.54, .33, .39)},
}
N = {"GS1": 8, "GS2": 11}


def prf(sel: set, gold: set):
    tp = len(sel & gold)
    P = tp / len(sel) if sel else 0.0
    R = tp / len(gold) if gold else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F, tp


def evaluate(arts, gs: str, select, subset=None) -> dict:
    """select(article, fig, n) -> iterable of sentence indices.
    subset(article, fig, gold_set) -> restricted gold set (e.g. implicit sentences only); selections are
    restricted the same way so precision stays comparable."""
    per = []
    tp_s = sel_s = gold_s = 0
    for a in arts:
        for fig, gold in sorted(a.gold.get(gs, {}).items()):
            if fig not in a.figures or not gold:
                continue
            sel = set(select(a, fig, N[gs]))
            if subset:
                gold = subset(a, fig, gold)
                sel = subset(a, fig, sel)
                if not gold:
                    continue
            P, R, F, tp = prf(sel, gold)
            per.append((P, R, F))
            tp_s += tp
            sel_s += len(sel)
            gold_s += len(gold)
    mean = [statistics.mean(x[k] for x in per) for k in range(3)]
    sd = [statistics.pstdev(x[k] for x in per) for k in range(3)]
    mP, mR = tp_s / max(1, sel_s), tp_s / max(1, gold_s)
    return {"figures": len(per), "gold": gold_s, "selected": sel_s,
            "mean": [round(v, 3) for v in mean], "sd": [round(v, 3) for v in sd],
            "micro": [round(mP, 3), round(mR, 3), round(2 * mP * mR / max(1e-9, mP + mR), 3)]}
