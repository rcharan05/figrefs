"""Phase 1 re-score: baselines + best methods so far, with P / R / F1 reported separately.

Metrics
  per-figure mean P / R / F1   FigSum+ metric; the selection = sentences naming the figure + chosen candidates
  implicit pooled P / R / F1   only candidates (sentences not naming the figure), counts pooled over figures
  far                          selections with d_sent > 3: correct / selected (gold far count)
Dev = nested leave-one-article-out over the 9 dev articles; test = fit on all dev, score the 4 test articles.
Bootstrap = paired over figures vs the same-paragraph adjacent rule (B0), 5,000 resamples.
usage: python eval/figsum/phase1.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
import experiments as E  # noqa: E402
import experiments2 as E2  # noqa: E402
import experiments3 as E3  # noqa: E402

HERE = os.path.dirname(__file__)
B0 = "Adjacent, same paragraph (B0)"


def score(D, arts, chosen):
    per_p, per_r, per_f, per = [], [], [], {}
    tp_i = n_sel = n_gold = 0
    far_sel = far_ok = far_gold = 0
    dist = {(r["art"], r["fig"], r["i"]): r["d_sent"] for r in D.rows if r["art"] in arts}
    for aid in arts:
        a = D.A[aid]
        for f in D.figs[aid]:
            gold = a.gold[E.GS][f]
            cue = {i for i, c in enumerate(a.cue) if f in c}
            imp = {i for (x, g, i) in chosen if x == aid and g == f}
            sel = cue | imp
            tp = len(sel & gold)
            P = tp / len(sel) if sel else 0.0
            R = tp / len(gold)
            F = 2 * P * R / (P + R) if P + R else 0.0
            per_p.append(P), per_r.append(R), per_f.append(F)
            per[(aid, f)] = F
            ig = gold - cue
            tp_i += len(imp & ig)
            n_sel += len(imp)
            n_gold += len(ig)
            far_sel += sum(1 for i in imp if dist.get((aid, f, i), 0) > 3)
            far_ok += sum(1 for i in imp & ig if dist.get((aid, f, i), 0) > 3)
            far_gold += sum(1 for i in ig if dist.get((aid, f, i), 99) > 3)
    Pi, Ri = tp_i / max(1, n_sel), tp_i / max(1, n_gold)
    return {"P": statistics.mean(per_p), "R": statistics.mean(per_r), "F1": statistics.mean(per_f),
            "iP": Pi, "iR": Ri, "iF1": 2 * Pi * Ri / (Pi + Ri) if Pi + Ri else 0.0, "isel": n_sel,
            "far": [far_ok, far_sel, far_gold], "figs": len(per_f)}, per


def figsum_chosen(D, name, arts):
    fn = E.SYSTEMS[name]
    out = set()
    for aid in arts:
        a = D.A[aid]
        ctx = E.Ctx(a)
        for f in D.figs[aid]:
            cue = {i for i, c in enumerate(a.cue) if f in c}
            out |= {(aid, f, i) for i in fn(ctx, f, 8) if i not in cue}
    return out


def methods():
    base = E2.ALL + E2.FIGF
    return {
        "Mention sentences only": E.m_fixed(lambda r: False),
        "FigSum+ Hybrid": None,
        "Adjacent sentence (d=1)": E.m_fixed(lambda r: r["d_sent"] == 1),
        B0: E.m_fixed(lambda r: r["d_sent"] == 1 and r["same_para"] == 1),
        "Stacked LR, d<=3": E2.m_stacked(base, 3),
        "GBM all features": E.m_learned(E2.ALL, "gbm"),
    }


def run(D, M, dev, test):
    res, per = {"dev": {}, "test": {}}, {"dev": {}, "test": {}}
    for name, fp in M.items():
        if fp is None:
            cd, ct = figsum_chosen(D, name, dev), figsum_chosen(D, name, test)
        else:
            cd = set()
            for h in dev:
                out = fp(D, [x for x in dev if x != h], [h])
                cd |= out[0] if isinstance(out, tuple) else out
            out = fp(D, dev, test)
            ct = out[0] if isinstance(out, tuple) else out
        for split, arts, ch in (("dev", dev, cd), ("test", test, ct)):
            r, p = score(D, arts, ch)
            res[split][name], per[split][name] = r, p
        print(f"done {name}", flush=True)
    for split in ("dev", "test"):
        for name in M:
            res[split][name]["vs_B0"] = E2.bootstrap(per[split][B0], per[split][name])
    return res


def table(res, split):
    print(f"\n== {split.upper()} ({next(iter(res[split].values()))['figs']} figures)")
    print(f"{'system':32s} | {'P':>5s} {'R':>5s} {'F1':>5s} | {'impP':>5s} {'impR':>5s} {'impF1':>5s} {'sel':>4s} | "
          f"{'far ok/sel/gold':>15s} | F1 vs B0 [95%]")
    for name, r in res[split].items():
        m, lo, hi = r["vs_B0"]
        far = "{}/{}/{}".format(*r["far"])
        print(f"{name:32s} | {r['P']:.3f} {r['R']:.3f} {r['F1']:.3f} | {r['iP']:.3f} {r['iR']:.3f} {r['iF1']:.3f} "
              f"{r['isel']:4d} | {far:>15s} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")


def main():
    D = E3.load()
    dev, test = D.split["dev"], D.split["test"]
    res = run(D, methods(), dev, test)
    table(res, "dev")
    table(res, "test")
    json.dump(res, open(os.path.join(HERE, "phase1.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
