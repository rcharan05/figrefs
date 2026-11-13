"""Phase 2: E1 visual vocabulary, E2 anaphora / coreference continuation, E3 far-sentence selection, E5 combination.

Every system = sentences naming the figure + B0 (adjacent sentence, same paragraph) + its own additions.
Window = within 3 sentences of a mention of the figure; far = further than 3 (and not naming another figure).

E1  visual vocabulary (narrow lexicon VIS: colours, line/bar/arrow/panel/axis/legend/lane words; no generic verbs)
    E1a-sp   + window candidates in the same paragraph that contain VIS
    E1a      + window candidates (any paragraph) that contain VIS
    E1a-LR   stacked LR d<=3 (phase-1 model) + VIS feature
    E1b-*    far candidates containing VIS, selected with the E3 machinery (similarity score, global per paper)
E2  continuation: grow a run from each mention sentence outwards (up to 3 sentences, same paragraph); the next
    sentence joins only if the later sentence of the pair continues the earlier one:
    anaph      later sentence opens with a demonstrative / pronoun ("These results", "This", "It", "They", "Such")
    chain-any  the two sentences share a Maverick coreference cluster
    chain-ana  they share a cluster and the later sentence's mention is a pronoun / demonstrative phrase
    anaph|chain-ana   either
E3  far candidates: score each (sentence, figure), keep each sentence's best figure, pool over the paper's figures,
    then keep the best ones: top-k per paper (k in 0,1,2,3,5,10), k = alpha * #figures, or a global threshold.
    The family and its parameter are chosen on the training articles (k = 0 allowed).
    Scores: plain TF-IDF / BGE similarity to the figure's mention sentences (ment) or caption (cap), the mean of
    their z-scores (mix), and LR with order (Bockhorst EdgesCrossed analogue) + degree features (LR-far).
E5  best window variant + best far variant (best = dev nested-LOAO mean F1).
Protocol: dev = nested leave-one-article-out (all choices made without the held-out article); test = fit on all
dev, score once.   usage: python eval/figsum/phase2.py
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import experiments as E  # noqa: E402
import experiments2 as E2  # noqa: E402
import experiments3 as E3  # noqa: E402
import phase1 as P1  # noqa: E402

HERE = os.path.dirname(__file__)
WIN = 3

VIS = re.compile(
    r"\b(red|blue|green|yellow|orange|purple|magenta|cyan|black|gr[ae]y|white|pink|brown|"
    r"curves?|dashed|dotted|solid lines?|(?<!cell )lines?\b(?! of)|bars?|histograms?|markers?|arrows?|arrowheads?|"
    r"asterisks?|insets?|panels?|lanes?|columns?|rows?|x-axis|y-axis|axis|axes|legend|plotted|plots?|depicted|"
    r"highlighted|boxed|circled|shaded|graphs?|images?|micrographs?|traces?|peaks?|"
    r"(left|right|upper|lower|top|bottom|middle) (panel|row|column|corner|side|part|graph|image|plot)s?)\b", re.I)
ANAPH = re.compile(r"^(these|this|those|such|it|they|its|their)\b", re.I)
PRON = re.compile(r"^(it|its|they|them|their|this|these|that|those|such|the latter|the former)\b", re.I)


# ---------------------------------------------------------------- data + features

def load():
    D = E3.load()
    coref = json.load(open(os.path.join(HERE, "coref_GS1.json")))
    D.links_any, D.links_ana = {}, {}
    for aid, cls in coref.items():
        la, lp = set(), set()
        for cl in cls:
            for (i, ti) in cl:
                for (j, tj) in cl:
                    if j < i:                       # i later than j
                        la.add((i, j))
                        if PRON.match(ti.strip()):
                            lp.add((i, j))
        D.links_any[aid], D.links_ana[aid] = la, lp
    for aid, a in D.A.items():
        ment = {}
        for i, c in enumerate(a.cue):
            for f in c:
                ment.setdefault(f, []).append(i)
        a._ment = ment
        a._vis = [bool(VIS.search(s)) for s in a.sentences]
        a._ana = [bool(ANAPH.match(s)) for s in a.sentences]
        # nearest preceding / following explicitly named figure for every sentence
        prev, nxt, last = [None] * len(a.sentences), [None] * len(a.sentences), None
        for i, c in enumerate(a.cue):
            prev[i] = last
            if c:
                last = max(c)
        last = None
        for i in range(len(a.cue) - 1, -1, -1):
            nxt[i] = last
            if a.cue[i]:
                last = min(a.cue[i])
        a._prev, a._next = prev, nxt
    for r in D.rows:
        a = D.A[r["art"]]
        i, f = r["i"], r["fig"]
        r["vis_n"] = int(a._vis[i])
        p, n = a._prev[i], a._next[i]
        lo = min(x for x in (p, n, f) if x is not None)
        hi = max(x for x in (p, n, f) if x is not None)
        inside = (p is None or n is None) or (min(p, n) <= f <= max(p, n))
        r["order_ok"] = int(inside)
        gaps = [abs(f - x) for x in (p, n) if x is not None]
        r["order_gap"] = math.log1p(min(gaps)) if gaps else 0.0
        sec = a.top_section[a.sent_para[i]]
        r["n_ment_sec"] = math.log1p(sum(1 for j in a._ment.get(f, []) if a.top_section[a.sent_para[j]] == sec))
        r["link_ment"] = int(any((max(i, j), min(i, j)) in D.links_any[r["art"]] for j in a._ment.get(f, [])
                                 if a.sent_para[j] == a.sent_para[i]))
        r["_ok_far"] = r["d_sent"] > WIN and r["names_other"] == 0
        r["_b0"] = r["d_sent"] == 1 and r["same_para"] == 1
    return D


def keyset(D, arts, pred):
    return {(r["art"], r["fig"], r["i"]) for a in arts for k in D.by_art[a] for r in (D.rows[k],) if pred(r)}


def b0(D, arts):
    return keyset(D, arts, lambda r: r["_b0"])


# ---------------------------------------------------------------- E1a / E2 rules (no parameters)

def m_rule(pred):
    def fp(D, train, test):
        return keyset(D, test, lambda r: r["_b0"] or pred(r))
    return fp


def m_cont(kind):
    """grow runs outwards from mention sentences inside the same paragraph, up to WIN sentences"""
    def cont(D, aid, later, earlier):
        a = D.A[aid]
        if kind == "anaph":
            return a._ana[later]
        if kind == "chain-any":
            return (later, earlier) in D.links_any[aid]
        if kind == "chain-ana":
            return (later, earlier) in D.links_ana[aid]
        if kind == "anaph|chain-ana":
            return a._ana[later] or (later, earlier) in D.links_ana[aid]
        raise ValueError(kind)

    def fp(D, train, test):
        out = b0(D, test)
        for aid in test:
            a = D.A[aid]
            n = len(a.sentences)
            for f in D.figs[aid]:
                for m in a._ment.get(f, []):
                    for step in (1, -1):
                        cur = m
                        for _ in range(WIN):
                            nx = cur + step
                            if not (0 <= nx < n) or a.sent_para[nx] != a.sent_para[m] or f in a.cue[nx]:
                                break
                            later, earlier = max(cur, nx), min(cur, nx)
                            if abs(nx - m) == 1 or cont(D, aid, later, earlier):
                                out.add((aid, f, nx))
                                cur = nx
                            else:
                                break
        return out
    return fp


# ---------------------------------------------------------------- E3 far selection

KS = [0, 1, 2, 3, 5, 10]
ALPHAS = [0.25, 0.5, 1.0, 2.0]


def far_pick(D, arts, sc, family, param):
    """sc: {(art, fig, i): score} over far candidates. Each sentence keeps its best figure; then per paper."""
    out = set()
    for aid in arts:
        best = {}
        for (x, f, i), s in sc.items():
            if x == aid and (i not in best or s > best[i][0]):
                best[i] = (s, f)
        ranked = sorted(((s, f, i) for i, (s, f) in best.items()), reverse=True)
        if family == "topk":
            take = ranked[:param]
        elif family == "alpha":
            take = ranked[:int(round(param * len(D.figs[aid])))]
        else:
            take = [t for t in ranked if t[0] >= param]
        out |= {(aid, f, i) for (s, f, i) in take}
    return out


def tune_far(D, arts, sc, base_fn, families=("topk", "alpha", "thresh")):
    base = base_fn(D, arts)
    vals = np.array(list(sc.values())) if sc else np.array([0.0])
    grid = []
    if "topk" in families:
        grid += [("topk", k) for k in KS]
    if "alpha" in families:
        grid += [("alpha", a) for a in ALPHAS]
    if "thresh" in families:
        grid += [("thresh", float(q)) for q in np.quantile(vals, [0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999])]
        grid += [("thresh", float("inf"))]
    best, bcfg = -1, None
    for fam, p in grid:
        r, _ = P1.score(D, arts, base | far_pick(D, arts, sc, fam, p))
        if r["F1"] > best + 1e-12:
            best, bcfg = r["F1"], (fam, p)
    return bcfg


def plain_scores(D, arts, key, filt=None):
    out = {}
    for a in arts:
        for k in D.by_art[a]:
            r = D.rows[k]
            if r["_ok_far"] and (filt is None or filt(r)):
                if key == "mix":
                    s = (r["z_tf_ment_all"] + r["z_bge_ment_all"] + r["z_tf_cap_all"] + r["z_bge_cap_all"]) / 4
                else:
                    s = r[key]
                out[(r["art"], r["fig"], r["i"])] = s
    return out


def m_far_plain(key, families=("topk", "alpha", "thresh"), filt=None, window=None):
    window = window or (lambda D, arts: b0(D, arts))

    def fp(D, train, test):
        cfg = tune_far(D, train, plain_scores(D, train, key, filt), window, families)
        return window(D, test) | far_pick(D, test, plain_scores(D, test, key, filt), *cfg), {"far": cfg}
    return fp


FAR_FEATS = E2.ALL + E2.FIGF + ["order_ok", "order_gap", "n_ment_sec", "vis_n"]


def lr_far_scores(D, train, test):
    idx = [k for a in train for k in D.by_art[a] if D.rows[k]["_ok_far"]]
    m = E.make_model("lr")
    m.fit(D.X(idx, FAR_FEATS), D.y(idx))
    j = [k for a in test for k in D.by_art[a] if D.rows[k]["_ok_far"]]
    p = m.predict_proba(D.X(j, FAR_FEATS))[:, 1]
    return {(D.rows[k]["art"], D.rows[k]["fig"], D.rows[k]["i"]): float(s) for k, s in zip(j, p)}


def m_far_lr(families=("topk", "alpha", "thresh"), window=None):
    window = window or (lambda D, arts: b0(D, arts))

    def fp(D, train, test):
        oof = {}
        for h in train:
            oof.update(lr_far_scores(D, [x for x in train if x != h], [h]))
        cfg = tune_far(D, train, oof, window, families)
        return window(D, test) | far_pick(D, test, lr_far_scores(D, train, test), *cfg), {"far": cfg}
    return fp


# ---------------------------------------------------------------- run

def methods():
    base = E2.ALL + E2.FIGF
    vis = lambda r: r["vis_n"] == 1  # noqa: E731
    M = {
        "B0 (adjacent, same paragraph)": m_rule(lambda r: False),
        "E1a-sp  visual, window, same para": m_rule(lambda r: r["d_sent"] <= WIN and r["same_para"] == 1 and r["vis_n"] == 1),
        "E1a     visual, window": m_rule(lambda r: r["d_sent"] <= WIN and r["vis_n"] == 1),
        "E1a-LR  stacked LR d<=3 + visual": E2.m_stacked(base + ["vis_n"], 3),
        "E2 anaph": m_cont("anaph"),
        "E2 chain-any": m_cont("chain-any"),
        "E2 chain-ana": m_cont("chain-ana"),
        "E2 anaph|chain-ana": m_cont("anaph|chain-ana"),
        "E2-LR  stacked LR d<=3 + visual + coref": E2.m_stacked(base + ["vis_n", "link_ment"], 3),
    }
    for key in ("tf_ment", "bge_ment", "tf_cap", "bge_cap", "mix"):
        M[f"E1b visual far, {key}"] = m_far_plain(key, filt=vis)
    for key in ("tf_ment", "bge_ment", "tf_cap", "bge_cap", "mix"):
        for fam in ("topk", "alpha", "thresh"):
            M[f"E3 {key}, {fam}"] = m_far_plain(key, (fam,))
        M[f"E3 {key}, any family"] = m_far_plain(key)
    for fam in ("topk", "alpha", "thresh"):
        M[f"E3 LR-far, {fam}"] = m_far_lr((fam,))
    M["E3 LR-far, any family"] = m_far_lr()
    return M


def chosen_sets(D, fp, dev, test):
    cd, params = set(), []
    for h in dev:
        out = fp(D, [x for x in dev if x != h], [h])
        c, p = out if isinstance(out, tuple) else (out, {})
        cd |= c
        params.append(p)
    out = fp(D, dev, test)
    ct, pt = out if isinstance(out, tuple) else (out, {})
    return cd, ct, params, pt


def main():
    D = load()
    dev, test = D.split["dev"], D.split["test"]
    M = methods()
    res, per, sets = {"dev": {}, "test": {}}, {"dev": {}, "test": {}}, {}
    for name, fp in M.items():
        cd, ct, pd, pt = chosen_sets(D, fp, dev, test)
        sets[name] = (cd, ct)
        for split, arts, ch in (("dev", dev, cd), ("test", test, ct)):
            r, p = P1.score(D, arts, ch)
            r["params"] = pd if split == "dev" else pt
            res[split][name], per[split][name] = r, p
        print(f"done {name}: dev F1 {res['dev'][name]['F1']:.3f}", flush=True)
    # E5: best window variant + best far variant by dev nested-LOAO F1 (union of their per-fold selections)
    win_names = [n for n in M if n.startswith(("E1a", "E2"))]
    far_names = [n for n in M if n.startswith(("E1b", "E3"))]
    bw = max(win_names, key=lambda n: res["dev"][n]["F1"])
    bf = max(far_names, key=lambda n: res["dev"][n]["F1"])
    name = f"E5 = [{bw.split('  ')[0].strip()}] + [{bf}]"
    for split, arts, j in (("dev", dev, 0), ("test", test, 1)):
        r, p = P1.score(D, arts, sets[bw][j] | sets[bf][j])
        res[split][name], per[split][name] = r, p
    for split in ("dev", "test"):
        for n in res[split]:
            res[split][n]["vs_B0"] = E2.bootstrap(per[split]["B0 (adjacent, same paragraph)"], per[split][n])
    P1.B0 = "B0 (adjacent, same paragraph)"
    for split in ("dev", "test"):
        P1.table({split: {n: r for n, r in res[split].items()}}, split)
    print("\n-- far-selection parameters chosen per dev fold / on all dev (test):")
    for n in far_names:
        print(f"   {n:32s} folds {[p.get('far') for p in res['dev'][n]['params']]} | test {res['test'][n]['params']}")
    json.dump(res, open(os.path.join(HERE, "phase2.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
