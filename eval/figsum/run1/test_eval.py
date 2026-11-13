"""One-shot sealed test evaluation, exactly as pre-registered in TEST_PLAN.md.
usage: python test_eval.py DATA.jsonl C0L_OOF SPAN_OOF C0L_TEST SPAN_TEST
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase8 as P  # noqa: E402


def keyed(S):
    return {tuple(int(x) if j else x for j, x in enumerate(k.split("|")[:3])): v for k, v in S.items()}


def ens(a, b):
    return {k: 0.5 * (a[k] + b[k]) for k in a if k in b}


def anchored(arts, R):
    by = {}
    for (p, f, i), v in R.items():
        by.setdefault((p, f), {})[i] = v
    out = {}
    for (p, f), d in by.items():
        E = [i for i, c in enumerate(arts[p].cue) if f in c and i in d]
        ref = np.mean([d[i] for i in E]) if E else np.median(list(d.values()))
        for i, v in d.items():
            out[(p, f, i)] = v - ref
    return out


def choose(arts, ids, Z, grid):
    return max(grid, key=lambda t: (P.metrics(arts, ids, P.predict(arts, ids, Z, t, True))["F1"], -abs(t - 1)))


def main():
    data, c0l_oof, span_oof, c0l_test, span_test = sys.argv[1:6]
    arts = {d["id"]: P.A(d) for d in (json.loads(l) for l in open(data))}
    train = sorted(p for p, a in arts.items() if a.split == "train")
    test = sorted(p for p, a in arts.items() if a.split == "test")
    Lo, So = keyed(json.load(open(c0l_oof))), keyed(json.load(open(span_oof)))
    Lt, St = keyed(json.load(open(c0l_test))), keyed(json.load(open(span_test)))
    RG = [round(float(x), 2) for x in np.arange(-8.0, 8.001, 0.1)]
    AG = [round(float(x), 2) for x in np.arange(-10.0, 4.001, 0.1)]
    b0f_test = P.metrics(arts, test, {(p, f): P._b0f(arts[p], f) for p in test for f in arts[p].figs})
    rows = [("B0f (rule)", b0f_test, None)]
    for name, Zo, Zt, grid in [
        ("PRIMARY ens SPAN+C0L +E anchored", anchored(arts, ens(So, Lo)), anchored(arts, ens(St, Lt)), AG),
        ("C0L+E raw", Lo, Lt, RG),
        ("SPAN+E raw", So, St, RG),
    ]:
        t = choose(arts, train, Zo, grid)
        oof = P.metrics(arts, train, P.predict(arts, train, Zo, t, True))
        m = P.metrics(arts, test, P.predict(arts, test, Zt, t, True))
        rows.append((f"{name} (t={t}; OOF F1 {oof['F1']:.3f})", m, b0f_test))
    nf = len(b0f_test["per"])
    print(f"SEALED TEST: {len(test)} papers {test}, {nf} figures")
    for name, m, base in rows:
        print(P.line(name, m, base))


if __name__ == "__main__":
    main()
