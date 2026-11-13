"""Round 3: asymmetric window + normalised similarity (user's proposal) + Discussion-revisit detector.

Two regimes per figure (sentences that name the figure are always selected):
  window      b sentences before .. a sentences after a mention (b, a tuned)
  far-Discussion  further than 5 from every mention, in Discussion/Conclusion, and not in a section that names the figure
Normalised score  S = mean_q z(q, R) - lam * z(topic, R)   q in {cap}, {ment}, {cap, ment};  R in {all, ment, local};
                  encoder TF-IDF, BGE or both (mean of the two z's).   beta = bonus for the adjacent sentence.
Variants: contiguity (a sentence is kept only if every sentence between it and the mention is kept),
          same-paragraph window, per-figure top-k inside the window, LR / stacked LR with the new features.
All tuning: nested leave-one-article-out on the 9 dev articles. usage: experiments3.py dev | test NAME...
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(__file__))
import experiments as E  # noqa: E402
import experiments2 as E2  # noqa: E402

HERE = os.path.dirname(__file__)
NEG = -1e9


def load():
    D = E.Data()
    E2.add_fig_features(D)
    F = json.load(open(os.path.join(HERE, "features3_GS1.json")))
    for r, f in zip(D.rows, F):
        assert (r["art"], r["fig"], r["i"]) == (f["art"], f["fig"], f["i"])
        r.update(f)
    return D


class FE:
    """fast evaluator over the candidate rows of a set of articles (mean per-figure F1 for many thresholds)"""

    def __init__(self, D, arts):
        self.D, self.arts = D, arts
        figs = [(a, f) for a in arts for f in D.figs[a]]
        fid = {x: j for j, x in enumerate(figs)}
        self.K = [k for a in arts for k in D.by_art[a] if (a, D.rows[k]["fig"]) in fid]
        self.fig = np.array([fid[(D.rows[k]["art"], D.rows[k]["fig"])] for k in self.K])
        self.y = np.array([D.rows[k]["y"] for k in self.K], bool)
        self.G = sparse.csr_matrix((np.ones(len(self.K)), (np.arange(len(self.K)), self.fig)), shape=(len(self.K), len(figs)))
        self.cue_tp, self.cue_n, self.gold_n = [], [], []
        for a, f in figs:
            A = D.A[a]
            gold = A.gold[E.GS][f]
            cue = {i for i, c in enumerate(A.cue) if f in c}
            self.cue_tp.append(len(cue & gold))
            self.cue_n.append(len(cue))
            self.gold_n.append(len(gold))
        self.cue_tp, self.cue_n, self.gold_n = map(np.array, (self.cue_tp, self.cue_n, self.gold_n))
        self.col = {}

    def arr(self, key):
        if key not in self.col:
            self.col[key] = np.array([self.D.rows[k][key] for k in self.K], float)
        return self.col[key]

    def mean_f1(self, score, ths):
        sel = (score[None, :] >= np.asarray(ths)[:, None]).astype(float)
        tp = self.cue_tp + (sel * self.y) @ self.G
        n = self.cue_n + sel @ self.G
        f1 = np.where(n > 0, 2 * tp / (n + self.gold_n), 0.0)
        return f1.mean(1)

    def chosen(self, score, t):
        return {(self.D.rows[k]["art"], self.D.rows[k]["fig"], self.D.rows[k]["i"]) for k, s in zip(self.K, score) if s >= t}


# ---------------------------------------------------------------- score construction

def zscore(fe, enc, q, R):
    if enc == "both":
        return 0.5 * (fe.arr(f"z_tf_{q}_{R}") + fe.arr(f"z_bge_{q}_{R}"))
    return fe.arr(f"z_{enc}_{q}_{R}")


def norm_score(fe, enc, qs, R, lam):
    s = np.mean([zscore(fe, enc, q, R) for q in qs], axis=0)
    return s - lam * zscore(fe, enc, "topic", R) if lam else s


def window_mask(fe, b, a, same_para=False):
    m = (fe.arr("da") <= a) | (fe.arr("db") <= b)
    if same_para:
        m &= fe.arr("same_para") == 1
    return m


def chain_min(fe, score):
    """contiguity: a candidate's effective score is the min over itself and the candidates between it and the mention"""
    if not hasattr(fe, "chainC"):
        pos = {(fe.D.rows[k]["art"], fe.D.rows[k]["fig"], fe.D.rows[k]["i"]): j for j, k in enumerate(fe.K)}
        C = np.tile(np.arange(len(fe.K))[:, None], (1, 6))  # pad with self
        for j, k in enumerate(fe.K):
            r = fe.D.rows[k]
            ps = [pos[(r["art"], r["fig"], i2)] for i2 in r["chain"][:5] if (r["art"], r["fig"], i2) in pos]
            C[j, 1:1 + len(ps)] = ps
        fe.chainC = C
    return score[fe.chainC].min(1)


QSETS = {"cap": ("cap",), "ment": ("ment",), "cap+ment": ("cap", "ment")}
WIN_GRID = [(b, a) for b in (0, 1, 2, 3) for a in (1, 2, 3, 4, 5)]
SCORE_GRID = list(itertools.product(("tf", "bge", "both"), QSETS, ("all", "ment", "local"), (-0.5, 0.0, 0.5, 1.0)))
BETAS = (0.0, 1.0, np.inf)


def ths_for(score):
    v = score[score > NEG / 2]
    if not len(v):
        return np.array([0.0])
    return np.unique(np.concatenate([np.quantile(v, np.linspace(0, 1, 41)), [v.max() + 1]]))


def build(fe, cfg, score_cache=None):
    """cfg: dict(kind=..., b, a, enc, qs, R, lam, beta, contig, same_para)"""
    m = window_mask(fe, cfg["b"], cfg["a"], cfg.get("same_para", False))
    if cfg["kind"] == "rule":
        return np.where(m, 1.0, NEG)
    key = (cfg["enc"], cfg["qs"], cfg["R"], cfg["lam"])
    s = norm_score(fe, cfg["enc"], QSETS[cfg["qs"]], cfg["R"], cfg["lam"])
    adj = (fe.arr("da") == 1) | (fe.arr("db") == 1)
    if np.isinf(cfg["beta"]):
        s = np.where(adj, 1e6, s)
    else:
        s = s + cfg["beta"] * adj
    s = np.where(m, s, NEG)
    if cfg.get("contig"):
        s = chain_min(fe, s)
    return s


def tune_window(fe, kind, contig=False, same_para=False, grid=None):
    best = (-1, None, None)
    if kind == "rule":
        for b, a in WIN_GRID:
            cfg = dict(kind="rule", b=b, a=a, same_para=same_para)
            v = fe.mean_f1(build(fe, cfg), [0.5])[0]
            if v > best[0]:
                best = (v, cfg, 0.5)
        return best
    for b, a in grid or WIN_GRID:
        for enc, qs, R, lam in SCORE_GRID:
            for beta in BETAS:
                cfg = dict(kind="score", b=b, a=a, enc=enc, qs=qs, R=R, lam=lam, beta=beta, contig=contig, same_para=same_para)
                s = build(fe, cfg)
                th = ths_for(s)
                v = fe.mean_f1(s, th)
                j = int(np.argmax(v))
                if v[j] > best[0]:
                    best = (v[j], cfg, th[j])
    return best


DISC_GRID = list(itertools.product(("tf", "bge", "both"), QSETS, ("all", "ment"), (-0.5, 0.0, 0.5, 1.0)))


def tune_disc(fe, win_cfg, win_t):
    """second regime on top of a fixed window method"""
    sw = build(fe, win_cfg)
    base = np.where(sw >= win_t, 1e9, NEG)
    far = fe.arr("far_disc") == 1
    best = (fe.mean_f1(base, [0.0])[0], None, np.inf)  # option: no Discussion detector
    for enc, qs, R, lam in DISC_GRID:
        s = norm_score(fe, enc, QSETS[qs], R, lam)
        sd = np.where(far, s, NEG)
        th = ths_for(sd)
        comb = np.maximum(base, sd)
        v = fe.mean_f1(comb, th)
        j = int(np.argmax(v))
        if v[j] > best[0]:
            best = (v[j], dict(enc=enc, qs=qs, R=R, lam=lam), th[j])
    return best


def disc_score(fe, dcfg):
    s = norm_score(fe, dcfg["enc"], QSETS[dcfg["qs"]], dcfg["R"], dcfg["lam"])
    return np.where(fe.arr("far_disc") == 1, s, NEG)


# ---------------------------------------------------------------- methods: (D, train, test) -> (chosen, params)

def m_window(kind, contig=False, same_para=False, disc=False):
    def fp(D, train, test):
        ftr, fte = FE(D, train), FE(D, test)
        _, cfg, t = tune_window(ftr, kind, contig, same_para)
        chosen = fte.chosen(build(fte, cfg), t)
        params = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in cfg.items()}
        params["t"] = float(t)
        if disc:
            _, dcfg, t2 = tune_disc(ftr, cfg, t)
            params["disc"] = dcfg
            params["t_disc"] = float(t2) if np.isfinite(t2) else None
            if dcfg is not None:
                chosen |= fte.chosen(disc_score(fte, dcfg), t2)
        return chosen, params
    return fp


def m_topk_window():
    """per-figure top-k inside the window by the normalised score (k and score config tuned)"""
    def fp(D, train, test):
        ftr, fte = FE(D, train), FE(D, test)

        def topk(fe, s, K):
            sel = np.full(len(s), NEG)
            for g in np.unique(fe.fig):
                idx = np.where((fe.fig == g) & (s > NEG / 2))[0]
                for j in idx[np.argsort(-s[idx])][:K]:
                    sel[j] = 1.0
            return sel
        best = (-1, None, None)
        for b, a in [(1, 2), (2, 3), (2, 5)]:
            for enc, qs, R, lam in SCORE_GRID:
                cfg = dict(kind="score", b=b, a=a, enc=enc, qs=qs, R=R, lam=lam, beta=0.0)
                s = build(ftr, cfg)
                for K in range(1, 7):
                    v = ftr.mean_f1(topk(ftr, s, K), [0.5])[0]
                    if v > best[0]:
                        best = (v, cfg, K)
        cfg, K = best[1], best[2]
        return fte.chosen(topk(fte, build(fte, cfg), K), 0.5), {**cfg, "k": K}
    return fp


NEWF = [f"z_{e}_{q}_{R}" for e in ("tf", "bge") for q in ("cap", "ment", "topic") for R in ("all", "ment", "local")] + \
       [f"da{k}" for k in range(1, 6)] + [f"db{k}" for k in range(1, 4)] + ["far_disc"]
REGIME = lambda r: r["da"] <= 5 or r["db"] <= 2 or r["far_disc"] == 1  # noqa: E731
WIN_ONLY = lambda r: r["da"] <= 5 or r["db"] <= 2  # noqa: E731


def methods():
    base = E2.ALL + E2.FIGF
    return {
        "Adjacent sentence only (d=1)": E.m_fixed(lambda r: r["d_sent"] == 1),
        "Asymmetric window rule (b,a tuned)": m_window("rule"),
        "Normalised score in window": m_window("score"),
        "Normalised score in window + Discussion detector": m_window("score", disc=True),
        "Normalised score, contiguity": m_window("score", contig=True),
        "Normalised score, contiguity + Discussion detector": m_window("score", contig=True, disc=True),
        "Normalised score, same-paragraph window": m_window("score", same_para=True),
        "Top-k per figure in window": m_topk_window(),
        "LR old+new feats, window(2,5)": E2.m_lr(base + NEWF, WIN_ONLY),
        "LR old+new feats, window(2,5) + far-Discussion": E2.m_lr(base + NEWF, REGIME),
        "Stacked LR old+new, window(2,5)": E2.m_stacked(base + NEWF, WIN_ONLY),
        "Stacked LR old+new, window(2,5) + far-Discussion": E2.m_stacked(base + NEWF, REGIME),
        "Stacked LR (round 2), d<=3": E2.m_stacked(base, 3),
    }


def disc_quality(D, arts, chosen):
    """precision / recall of selections inside the far-Discussion regime"""
    far = {(r["art"], r["fig"], r["i"]): r["y"] for r in D.rows if r["art"] in arts and r["far_disc"] == 1}
    sel = [far[x] for x in chosen if x in far]
    g = sum(far.values())
    return {"selected": len(sel), "correct": int(sum(sel)), "gold": int(g)}


def run(D, arts_train_test, M, names, label):
    res, per = {}, {}
    for name in names:
        fp = M[name]
        if label == "dev":
            chosen, params = set(), []
            for h in arts_train_test:
                out = fp(D, [x for x in arts_train_test if x != h], [h])
                c, p = out if isinstance(out, tuple) else (out, {})
                chosen |= c
                params.append(p)
            arts = arts_train_test
        else:
            dev, test = arts_train_test
            out = fp(D, dev, test)
            chosen, params = out if isinstance(out, tuple) else (out, {})
            arts = test
        r = E.evaluate(D, arts, chosen)
        r["disc"] = disc_quality(D, arts, chosen)
        r["params"] = params
        res[name] = r
        per[name] = E2.per_fig_f1(D, arts, chosen)
        dq = r["disc"]
        print(E.fmt(name, r) + f" | far-Disc {dq['correct']}/{dq['selected']} sel, gold {dq['gold']}", flush=True)
    return res, per


def main():
    D = load()
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    M = methods()
    dev, test = D.split["dev"], D.split["test"]
    if mode == "dev":
        print(f"== DEV round 3, nested LOAO over {len(dev)} articles")
        res, per = run(D, dev, M, list(M), "dev")
        base = per["Adjacent sentence only (d=1)"]
        print("\n-- paired bootstrap over figures vs adjacent rule [95% interval]")
        for n in M:
            m, lo, hi = E2.bootstrap(base, per[n])
            res[n]["vs_adjacent"] = [m, lo, hi]
            print(f"   {n:52s} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
        print("\n-- parameters chosen per fold (first 3 folds) for the normalised-score methods")
        for n in ("Normalised score in window", "Normalised score in window + Discussion detector"):
            for p in res[n]["params"][:3]:
                print("  ", n[:30], {k: v for k, v in p.items()})
        json.dump(res, open(os.path.join(HERE, "exp3_dev.json"), "w"), indent=1, default=str)
    else:
        names = sys.argv[2:] or list(M)
        print(f"== TEST (second look) {len(test)} articles; fitted on all dev")
        res, per = run(D, (dev, test), M, names, "test")
        if "Adjacent sentence only (d=1)" in per:
            for n in names:
                m, lo, hi = E2.bootstrap(per["Adjacent sentence only (d=1)"], per[n])
                print(f"   vs adjacent: {n:52s} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
        # per-article mean F1
        for n in names:
            pa = {}
            for (a, f), v in per[n].items():
                pa.setdefault(a, []).append(v)
            print("   per article:", n[:40], {a: round(float(np.mean(v)), 3) for a, v in pa.items()})
        json.dump(res, open(os.path.join(HERE, "exp3_test.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
