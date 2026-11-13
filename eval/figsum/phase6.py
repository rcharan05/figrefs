"""Phase 6: plug a content-score file (M2 cross-encoder, M2-lite, ...) into the B0f backbone.

score file: {"art|fig|i": score}; z-normalised within each article over all its candidate pairs.
Systems (every threshold chosen inside the folds, objective per-figure mean F1):
  B0f & S            drop B0f picks with z < t1
  B0f + S-ext        add d=2-3 same-paragraph candidates with z >= t2 when the sentence between is selected
  B0f & S + S-ext    both
  B0f + S-far        far candidates (> 3 from every citation, not citing another figure), per-paper selector (top-k /
                     alpha * #figures / threshold, k = 0 allowed)
  best window + far  union of the best window system and the S-far far picks
Protocols: dev nested LOAO + test (fit on dev, scored once) and all-13 LOAO. Also AUCs on dev.
usage: python eval/figsum/phase6.py SCORES.json [NAME]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import experiments2 as E2  # noqa: E402
import phase1 as P1  # noqa: E402
import phase2 as P  # noqa: E402
import phase3 as P3  # noqa: E402

HERE = os.path.dirname(__file__)
ZG = [float(x) for x in np.arange(-2.0, 3.01, 0.25)]


def set_scores(D, path):
    S = json.load(open(path))
    by = {}
    for k, v in S.items():
        by.setdefault(k.split("|")[0], []).append(v)
    st = {a: (np.mean(v), np.std(v) + 1e-9) for a, v in by.items()}
    for r in D.rows:
        k = f"{r['art']}|{r['fig']}|{r['i']}"
        if k in S:
            m, s = st[r["art"]]
            r["llm"], r["llm_has"] = (S[k] - m) / s, 1
        else:
            r["llm"], r["llm_has"] = P3.NEG, 0


def systems():
    b0f = lambda r: r["_b0f"]  # noqa: E731
    return {
        "B0f": P3.fixed(b0f),
        "B0f & S": P3.tuned(lambda D, arts, t: P3.keys(D, arts, lambda r: r["_b0f"] and r["llm"] >= t), ZG),
        "B0f + S-ext": P3.tuned(P3.b_ext, ZG),
        "B0f & S + S-ext": P3.tuned(P3.b_and, [(a, b) for a in ZG[::2] for b in ZG[::2]]),
        "B0f + S-far": P3.m_far("llm", b0f),
    }


def main():
    from sklearn.metrics import roc_auc_score
    path = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(path).split(".")[0]
    D = P3.load()
    set_scores(D, path)
    dev, test = D.split["dev"], D.split["test"]
    for lab, cond in (("window d<=3", lambda r: r["d_sent"] <= 3), ("B0f picks", lambda r: r["_b0f"]),
                      ("d=2-3 same para", lambda r: 2 <= r["d_sent"] <= 3 and r["same_para"]),
                      ("far", lambda r: r["_ok_far"])):
        R = [r for r in D.rows if r["art"] in dev and cond(r)]
        print(f"AUC dev {lab:16s} n={len(R):5d} {name} {roc_auc_score([r['y'] for r in R], [r['llm'] for r in R]):.3f}"
              f"  tf_cap {roc_auc_score([r['y'] for r in R], [r['tf_cap'] for r in R]):.3f}")
    M = systems()
    allA = dev + test
    res = {"dev": {}, "test": {}, "all13": {}}
    per = {k: {} for k in res}
    sets = {}
    for n, fp in M.items():
        cd, ct, pd, pt = P.chosen_sets(D, fp, dev, test)
        ca, pa = set(), []
        for h in allA:
            o = fp(D, [x for x in allA if x != h], [h])
            c, p = o if isinstance(o, tuple) else (o, {})
            ca |= c
            pa.append(p)
        sets[n] = {"dev": cd, "test": ct, "all13": ca}
        for split, arts, ch, pp in (("dev", dev, cd, pd), ("test", test, ct, pt), ("all13", allA, ca, pa)):
            r, p = P1.score(D, arts, ch)
            r["params"] = pp
            res[split][n], per[split][n] = r, p
        print(f"done {n}", flush=True)
    win = ["B0f & S", "B0f + S-ext", "B0f & S + S-ext"]
    bw = max(win, key=lambda n: res["dev"][n]["F1"])
    cn = f"[{bw}] + far"
    for split, arts in (("dev", dev), ("test", test), ("all13", allA)):
        far = {x for x in sets["B0f + S-far"][split] if D.sel[x]["_ok_far"]}
        r, p = P1.score(D, arts, sets[bw][split] | far)
        res[split][cn], per[split][cn] = r, p
    for split in res:
        for n in res[split]:
            res[split][n]["vs_B0"] = E2.bootstrap(per[split]["B0f"], per[split][n])
    P1.B0 = "B0f"
    for split in res:
        print(f"\n[{name}] intervals are vs B0f")
        P1.table({split: res[split]}, split)
    for n in M:
        if n != "B0f":
            print("params", n, [p.get("p", p.get("far")) for p in res["dev"][n]["params"]], "| test",
                  res["test"][n]["params"])
    json.dump(res, open(os.path.join(HERE, f"phase6_{name}.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
