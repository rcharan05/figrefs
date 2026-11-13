"""Phase 8: score run-1 models (out-of-fold on the 15 training papers; sealed test once at the very end).

Prediction for figure F (one threshold t on the per-article z-score of the logit; no distance rule anywhere):
  C0, S1        {i : z(F, i) >= t}                       whole paper, citing sentences included (model alone)
  S2*           E(F) U {i not in E(F) : z(F, i) >= t}    E(F) = sentences citing F (detector: JATS xref / grammar)
  C0+E          E(F) U {i : z >= t}                      phase-7 "model + citing sentences" variant
t is chosen by the same 5-fold rotation: for the papers of fold k, t maximises mean F1 on the other 4 folds.
Metric = FigSum+ per-figure P/R/F1 averaged over figures (73 figures). Breakdown of selections into citing /
window (<= 3 sentences from a citation) / far is descriptive only. Interval = paired bootstrap over figures
(5,000 resamples) of the gain over B0f.
usage: python phase8.py DATA.jsonl OOF_DIR                 # all configs found in OOF_DIR, training papers
       python phase8.py DATA.jsonl OOF_DIR test CFG SEALED  # ONCE: t from all 15 papers, applied to the test papers
"""
from __future__ import annotations

import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from articles import b0f as _b0f  # noqa: E402

ZG = [round(float(x), 2) for x in np.arange(-1.0, 4.001, 0.05)]   # per-article z-score grid
RG = [round(float(x), 2) for x in np.arange(-8.0, 8.001, 0.1)]     # raw-logit grid
ZG_ACTIVE = ZG


class A:     # minimal article view for articles.b0f
    def __init__(self, d):
        self.id, self.sentences, self.sent_para, self.cue = d["id"], d["sentences"], d["sent_para"], [set(c) for c in d["cue"]]
        self.top_section = d["para_section"]
        self.gold = {int(f): set(v) for f, v in d["gold"].items()}
        self.figs, self.fold, self.split = d["figs"], d.get("fold", -1), d["split"]


def zscores(S):
    by = {}
    for k, v in S.items():
        by.setdefault(k.split("|")[0], []).append(v)
    st = {a: (float(np.mean(v)), float(np.std(v)) + 1e-9) for a, v in by.items()}
    return {tuple(int(x) if j else x for j, x in enumerate(k.split("|")[:3])): (v - st[k.split("|")[0]][0]) / st[k.split("|")[0]][1]
            for k, v in S.items()}


def predict(arts, ids, Z, t, with_E):
    sel = {}
    for p in ids:
        a = arts[p]
        for f in a.figs:
            E = {i for i, c in enumerate(a.cue) if f in c}
            s = {i for i in range(len(a.sentences)) if Z.get((p, f, i), -1e9) >= t}
            sel[(p, f)] = (s | E) if with_E else s
    return sel


def metrics(arts, ids, sel):
    P, R, F, per = [], [], [], {}
    br = {k: [0, 0, 0] for k in ("cites", "window", "far")}
    for p in ids:
        a = arts[p]
        for f in a.figs:
            g, s = a.gold[f], sel.get((p, f), set())
            tp = len(s & g)
            pp, rr = (tp / len(s) if s else 0.0), tp / len(g)
            ff = 2 * pp * rr / (pp + rr) if pp + rr else 0.0
            P.append(pp), R.append(rr), F.append(ff)
            per[(p, f)] = ff
            E = [i for i, c in enumerate(a.cue) if f in c]
            kind = lambda i: "cites" if i in E else ("far" if not E or min(abs(i - m) for m in E) > 3 else "window")
            for i in s:
                br[kind(i)][1] += 1
                br[kind(i)][0] += i in g
            for i in g:
                br[kind(i)][2] += 1
    return {"P": statistics.mean(P), "R": statistics.mean(R), "F1": statistics.mean(F), "br": br, "per": per}


def bootstrap(base, new, n=5000, seed=0):
    keys = sorted(base)
    d = np.array([new[k] - base[k] for k in keys])
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def rotate(arts, ids, Z, with_E):
    folds = sorted({arts[p].fold for p in ids})
    chosen, ts = {}, {}
    for k in folds:
        held = [p for p in ids if arts[p].fold == k]
        rest = [p for p in ids if arts[p].fold != k]
        t = max(ZG_ACTIVE, key=lambda t: (metrics(arts, rest, predict(arts, rest, Z, t, with_E))["F1"], -abs(t - 1)))
        ts[k] = t
        chosen.update(predict(arts, held, Z, t, with_E))
    return chosen, ts


def auc(arts, ids, Z):
    from sklearn.metrics import roc_auc_score
    y, s, kind = [], [], []
    for (p, f, i), z in Z.items():
        if p not in ids or f not in arts[p].figs:
            continue
        a = arts[p]
        E = [j for j, c in enumerate(a.cue) if f in c]
        y.append(int(i in a.gold[f])), s.append(z)
        kind.append("cites" if i in E else ("far" if not E or min(abs(i - m) for m in E) > 3 else "window"))
    y, s, kind = np.array(y), np.array(s), np.array(kind)
    out = {}
    for k in ("window", "far"):
        m = kind == k
        if y[m].min() != y[m].max():
            out[k] = roc_auc_score(y[m], s[m])
    out["all"] = roc_auc_score(y, s)
    return out


def line(name, m, base=None):
    b = m["br"]
    txt = f"{name:34s} P {m['P']:.3f} R {m['R']:.3f} F1 {m['F1']:.3f}"
    if base is not None:
        d, lo, hi = bootstrap(base["per"], m["per"])
        txt += f"  vs B0f {d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    txt += "   | " + "  ".join(f"{k} {b[k][0]}/{b[k][1]}/{b[k][2]}" for k in ("cites", "window", "far"))
    return txt


def main():
    data, oof = sys.argv[1], sys.argv[2]
    arts = {d["id"]: A(d) for d in (json.loads(l) for l in open(data))}
    train = sorted(p for p, a in arts.items() if a.split == "train")
    b0f = {(p, f): _b0f(arts[p], f) for p in train for f in arts[p].figs}
    base = metrics(arts, train, b0f)
    if len(sys.argv) > 3 and sys.argv[3] == "test":
        cfg, sealed = sys.argv[4], sys.argv[5]
        test = sorted(p for p, a in arts.items() if a.split == "test")
        withE = cfg.startswith("S2") or cfg.endswith("+E")
        Zo = zscores(json.load(open(os.path.join(oof, cfg.replace("+E", "") + ".json"))))
        t = max(ZG, key=lambda t: (metrics(arts, train, predict(arts, train, Zo, t, withE))["F1"], -abs(t - 1)))
        Zt = zscores(json.load(open(os.path.join(sealed, cfg.replace("+E", "") + ".json"))))
        bt = metrics(arts, test, {(p, f): _b0f(arts[p], f) for p in test for f in arts[p].figs})
        mt = metrics(arts, test, predict(arts, test, Zt, t, withE))
        print(f"TEST (3 papers, {len(bt['per'])} figures), threshold z >= {t} from all 15 training papers")
        print(line("B0f", bt)), print(line(cfg, mt, bt))
        return
    print(f"training pool: {len(train)} papers, {len(base['per'])} figures; selections shown as correct/selected/gold")
    print(line("B0f (rule)", base))
    for fn in sorted(os.listdir(oof)):
        if not fn.endswith(".json"):
            continue
        cfg = fn[:-5]
        raw = json.load(open(os.path.join(oof, fn)))
        Z = zscores(raw)
        R = {tuple(int(x) if j else x for j, x in enumerate(k.split("|")[:3])): v for k, v in raw.items()}
        s2 = cfg.startswith("S2")
        variants = [(cfg + " [z]", Z, s2, ZG)] + ([] if s2 else [(cfg + "+E [z]", Z, True, ZG)])
        variants.append((cfg + ("" if s2 else "+E") + " [raw]", R, True, RG))   # the fixed decision since run 2b
        global ZG_ACTIVE
        for name, Zv, withE, grid in variants:
            ZG_ACTIVE = grid
            sel, ts = rotate(arts, train, Zv, withE)
            print(line(name, metrics(arts, train, sel), base) + f"   t={sorted(set(ts.values()))}")
        ZG_ACTIVE = ZG
        print(f"{'':34s} AUC " + "  ".join(f"{k} {v:.3f}" for k, v in auc(arts, train, Z).items()))


if __name__ == "__main__":
    main()
