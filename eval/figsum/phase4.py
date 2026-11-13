"""Phase 4: compare LLM judges (model x prompt, and their z-score ensemble) on DEV only; test only on request.

Judges: v1 = Qwen2.5-1.5B prompt v1 (llm_scores.json), q25v2 = Qwen2.5-1.5B prompt v2, q3v2 = Qwen3-1.7B prompt v2;
ens = mean of the judges' z-scores (z over all window candidates of the article).
For each judge: AUC on dev window candidates, then nested-LOAO mean F1 of the window systems from phase 3.
usage: python eval/figsum/phase4.py dev
       python eval/figsum/phase4.py test JUDGE      # final, one look
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
FILES = {"v1": "llm_scores.json", "q25v2": "llm_q25_v2.json", "q3v2": "llm_q3_v2.json", "q3v1": "llm_q3_v1.json"}


def judge_scores(name):
    if name != "ens" and "+" not in name:
        return json.load(open(os.path.join(HERE, FILES[name])))
    parts = ["v1", "q25v2", "q3v2"] if name == "ens" else name.split("+")
    L = [json.load(open(os.path.join(HERE, FILES[p]))) for p in parts]
    keys = set.intersection(*(set(x) for x in L))
    out = {k: 0.0 for k in keys}
    for x in L:
        by_art = {}
        for k in keys:
            by_art.setdefault(k.split("|")[0], []).append(x[k])
        st = {a: (np.mean(v), np.std(v) + 1e-9) for a, v in by_art.items()}
        for k in keys:
            m, s = st[k.split("|")[0]]
            out[k] += (x[k] - m) / s / len(L)
    return out


def set_judge(D, name):
    L = judge_scores(name)
    for r in D.rows:
        k = f"{r['art']}|{r['fig']}|{r['i']}"
        r["llm"] = L.get(k, P3.NEG)
        r["llm_has"] = int(k in L)


def systems(name):
    TG = P3.TG if name not in ("ens",) and "+" not in name else [float(x) for x in np.arange(-2, 3.01, 0.25)]
    return {
        f"L-window-sp [{name}]": P3.tuned(P3.b_window(True), TG),
        f"B0f + L-ext [{name}]": P3.tuned(P3.b_ext, TG),
        f"B0f & L + L-ext [{name}]": P3.tuned(P3.b_and, [(a, b) for a in TG[::2] for b in TG[::2]]),
    }


def main():
    from sklearn.metrics import roc_auc_score
    D = P3.load()
    dev, test = D.split["dev"], D.split["test"]
    mode = sys.argv[1]
    if mode == "dev":
        avail = [n for n in FILES if os.path.exists(os.path.join(HERE, FILES[n]))]
        judges = ["q3v1", "v1+q3v1"]
        res = {}
        for j in judges:
            set_judge(D, j)
            for lab, cond in (("window", lambda r: r["d_sent"] <= 3), ("B0 picks", lambda r: r["_b0"]),
                              ("d=2-3 same para", lambda r: 2 <= r["d_sent"] <= 3 and r["same_para"])):
                R = [r for r in D.rows if r["art"] in dev and cond(r)]
                print(f"{j:8s} AUC {lab:16s} {roc_auc_score([r['y'] for r in R], [r['llm'] for r in R]):.3f}")
            for n, fp in systems(j).items():
                ch, params = set(), []
                for h in dev:
                    c, p = fp(D, [x for x in dev if x != h], [h])
                    ch |= c
                    params.append(p["p"])
                r, per = P1.score(D, dev, ch)
                b0 = P1.score(D, dev, P3.keys(D, dev, lambda r: r["_b0"]))[1]
                r["vs_B0"] = E2.bootstrap(b0, per)
                res[n] = r
                m, lo, hi = r["vs_B0"]
                print(f"   {n:30s} P {r['P']:.3f} R {r['R']:.3f} F1 {r['F1']:.3f} | imp P {r['iP']:.3f} R {r['iR']:.3f} "
                      f"F1 {r['iF1']:.3f} sel {r['isel']} | vs B0 {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | params {params[:3]}",
                      flush=True)
        json.dump(res, open(os.path.join(HERE, "phase4_dev.json"), "w"), indent=1, default=str)
    else:
        j = sys.argv[2]
        set_judge(D, j)
        out = {}
        b0d = P1.score(D, test, P3.keys(D, test, lambda r: r["_b0"]))
        for n, fp in systems(j).items():
            c, p = fp(D, dev, test)
            r, per = P1.score(D, test, c)
            r["vs_B0"] = E2.bootstrap(b0d[1], per)
            r["params"] = p
            out[n] = r
            m, lo, hi = r["vs_B0"]
            print(f"TEST {n:30s} P {r['P']:.3f} R {r['R']:.3f} F1 {r['F1']:.3f} | imp P {r['iP']:.3f} R {r['iR']:.3f} "
                  f"F1 {r['iF1']:.3f} sel {r['isel']} | vs B0 {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {p}")
        json.dump(out, open(os.path.join(HERE, f"phase4_test_{j}.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
