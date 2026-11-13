"""Evaluate single methods and z-score ensembles from saved proxy cases."""
import itertools
import json
import sys

import numpy as np

cases = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "eval/implicit_proxy_cases.json"))
methods = sorted(cases[0]["scores"])


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-9)


def evaluate(combo, weights=None):
    acc, mrr = [], []
    for c in cases:
        if not all(m in c["scores"] for m in combo):
            continue
        s = sum((weights[i] if weights else 1) * z(c["scores"][m]) for i, m in enumerate(combo))
        order = np.argsort(-s)
        r = int(np.where(order == c["target"])[0][0]) + 1
        acc.append(r == 1)
        mrr.append(1 / r)
    return np.mean(acc), np.mean(mrr), len(acc)


rows = [(m,) + evaluate([m]) for m in methods]
for k in (2, 3):
    for combo in itertools.combinations(methods, k):
        rows.append(("+".join(combo),) + evaluate(list(combo)))
rows.sort(key=lambda r: -r[2])
print(f"{len(cases)} cases, {len(set(c['paper'] for c in cases))} papers")
print("singles:")
for m in methods:
    a, r, n = evaluate([m])
    print(f"  {m:28s} top1={a:.3f} mrr={r:.3f}")
print("best combos:")
for name, a, r, n in rows[:12]:
    print(f"  {name:70s} top1={a:.3f} mrr={r:.3f}")
