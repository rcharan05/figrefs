"""Phase 3: failure-driven filters, LLM judge (Qwen2.5-1.5B zero-shot), distant-supervision 'figure-talk' score.

Every system = sentences naming the figure + its own selections. Tuned parameters are chosen with nested
leave-one-article-out on dev (objective: per-figure mean F1); test = fit on all dev, scored once.
Filters (fixed rules, no parameters), applied to B0 picks:
  -other   drop sentences that name a different figure
  -off     drop 'offtarget' sentences (tables, supplementary figures, 'data not shown', web supplement)
  -meth    drop sentences in Methods sections
  B0f = B0 -other -off -meth
LLM systems (score = logit Yes - logit No):
  L-window      window candidates (d<=3) with score >= t          (B0 not used)
  L-window-sp   same, same paragraph only
  B0f + L-ext   B0f, plus d=2-3 same-paragraph candidates with score >= t whose in-between sentence is selected
  B0f & L       B0f picks with score >= t1, plus L-ext with t2
  L-far         B0f + far shortlist (top 20 per paper by mix similarity) chosen with the E3 per-paper selector
Learned: stacked LR d<=3 with llm + figtalk features.
Figure-talk (FT): E3 far selection with score = figtalk probability x mix similarity rank.
usage: python eval/figsum/phase3.py
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

HERE = os.path.dirname(__file__)
NEG = -99.0


def load():
    D = P.load()
    L = json.load(open(os.path.join(HERE, "llm_scores.json")))
    FT = json.load(open(os.path.join(HERE, "figtalk_GS1.json")))
    for r in D.rows:
        r["llm"] = L.get(f"{r['art']}|{r['fig']}|{r['i']}", NEG)
        r["llm_has"] = int(f"{r['art']}|{r['fig']}|{r['i']}" in L)
        r["figtalk"] = FT[r["art"]][r["i"]]
        r["_meth"] = r["sec_methods"] == 1
        r["_b0f"] = r["_b0"] and not (r["names_other"] or r["offtarget"] or r["_meth"])
    D.sel = {(r["art"], r["fig"], r["i"]): r for r in D.rows}
    return D


def keys(D, arts, pred):
    return {(r["art"], r["fig"], r["i"]) for a in arts for k in D.by_art[a] for r in (D.rows[k],) if pred(r)}


def fixed(pred):
    return lambda D, train, test: keys(D, test, pred)


def tune(D, train, build, grid):
    best, bp = -1.0, None
    for p in grid:
        f1 = P1.score(D, train, build(D, train, p))[0]["F1"]
        if f1 > best + 1e-12:
            best, bp = f1, p
    return bp


def tuned(build, grid):
    def fp(D, train, test):
        p = tune(D, train, build, grid)
        return build(D, test, p), {"p": p}
    return fp


TG = [float(x) for x in np.arange(-4, 6.01, 0.5)]


def ext(D, arts, base, t):
    """add d=2..3 same-paragraph candidates scoring >= t if the sentence between them and the mention is selected"""
    out = set(base)
    for _ in range(2):
        for a in arts:
            for k in D.by_art[a]:
                r = D.rows[k]
                if 2 <= r["d_sent"] <= 3 and r["same_para"] and r["llm"] >= t and not r["names_other"]:
                    A = D.A[a]
                    ms = A._ment[r["fig"]]
                    m = min(ms, key=lambda j: abs(j - r["i"]))
                    step = 1 if r["i"] > m else -1
                    if (a, r["fig"], r["i"] - step) in out:
                        out.add((a, r["fig"], r["i"]))
    return out


def b_window(sp):
    return lambda D, arts, t: keys(D, arts, lambda r: r["d_sent"] <= 3 and (r["same_para"] or not sp)
                                   and r["llm"] >= t and not r["names_other"])


def b_ext(D, arts, t):
    return ext(D, arts, keys(D, arts, lambda r: r["_b0f"]), t)


def b_and(D, arts, p):
    t1, t2 = p
    return ext(D, arts, keys(D, arts, lambda r: r["_b0f"] and r["llm"] >= t1), t2)


def far_scores(D, arts, key):
    out = {}
    for a in arts:
        for k in D.by_art[a]:
            r = D.rows[k]
            if not r["_ok_far"]:
                continue
            if key == "llm":
                if r["llm_has"]:
                    out[(a, r["fig"], r["i"])] = r["llm"]
            elif key == "figtalk*mix":
                mix = (r["z_tf_ment_all"] + r["z_bge_ment_all"] + r["z_tf_cap_all"] + r["z_bge_cap_all"]) / 4
                out[(a, r["fig"], r["i"])] = mix + 3 * np.log(r["figtalk"] + 1e-6)
    return out


def m_far(key, window):
    def fp(D, train, test):
        cfg = P.tune_far(D, train, far_scores(D, train, key), lambda D, arts: keys(D, arts, window))
        return keys(D, test, window) | P.far_pick(D, test, far_scores(D, test, key), *cfg), {"far": cfg}
    return fp


def methods():
    base = E2.ALL + E2.FIGF
    b0f = lambda r: r["_b0f"]  # noqa: E731
    return {
        "B0": fixed(lambda r: r["_b0"]),
        "B0 -other": fixed(lambda r: r["_b0"] and not r["names_other"]),
        "B0 -off": fixed(lambda r: r["_b0"] and not r["offtarget"]),
        "B0 -meth": fixed(lambda r: r["_b0"] and not r["_meth"]),
        "B0f (= B0 -other -off -meth)": fixed(b0f),
        "L-window (LLM only, d<=3)": tuned(b_window(False), TG),
        "L-window-sp (LLM only, d<=3 same para)": tuned(b_window(True), TG),
        "B0f + L-ext (d=2-3 contiguous)": tuned(b_ext, TG),
        "B0f & L (filter) + L-ext": tuned(b_and, [(a, b) for a in TG[::2] for b in TG[::2]]),
        "Stacked LR d<=3 + llm + figtalk": E2.m_stacked(base + ["llm", "figtalk"], 3),
        "B0f + L-far (LLM on far shortlist)": m_far("llm", b0f),
        "B0f + FT-far (figtalk x similarity)": m_far("figtalk*mix", b0f),
    }


def main():
    D = load()
    dev, test = D.split["dev"], D.split["test"]
    M = methods()
    res, per, sets = {"dev": {}, "test": {}}, {"dev": {}, "test": {}}, {}
    for name, fp in M.items():
        cd, ct, pd, pt = P.chosen_sets(D, fp, dev, test)
        sets[name] = (cd, ct)
        for split, arts, ch in (("dev", dev, cd), ("test", test, ct)):
            r, p = P1.score(D, arts, ch)
            r["params"] = pd if split == "dev" else pt
            res[split][name], per[split][name] = r, p
        print(f"done {name}: dev F1 {res['dev'][name]['F1']:.3f}", flush=True)
    # combination: best window system + best far system (by dev F1)
    win = [n for n in M if "far" not in n]
    far = [n for n in M if "far" in n]
    bw, bf = max(win, key=lambda n: res["dev"][n]["F1"]), max(far, key=lambda n: res["dev"][n]["F1"])
    nm = f"COMBO [{bw}] + far part of [{bf}]"
    for split, arts, j in (("dev", dev, 0), ("test", test, 1)):
        far_only = {x for x in sets[bf][j] if D.sel[x]["_ok_far"]}
        r, p = P1.score(D, arts, sets[bw][j] | far_only)
        res[split][nm], per[split][nm] = r, p
    for split in ("dev", "test"):
        for n in res[split]:
            res[split][n]["vs_B0"] = E2.bootstrap(per[split]["B0"], per[split][n])
    P1.B0 = "B0"
    for split in ("dev", "test"):
        P1.table({split: res[split]}, split)
    print("\n-- chosen parameters (dev folds | test)")
    for n in M:
        if res["dev"][n].get("params") and res["dev"][n]["params"][0]:
            print(f"   {n:42s} {[list(p.values())[0] for p in res['dev'][n]['params']]} | {res['test'][n]['params']}")
    json.dump(res, open(os.path.join(HERE, "phase3.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
