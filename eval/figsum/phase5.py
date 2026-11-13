"""Phase 5: sequence models for the window around a figure's citations (M1 CRF, M3 text-tiling segmentation).

Sequence = for figure F, a block from 3 sentences before a group of F's citing sentences to 3 after it (groups merge
when citations are <= 6 apart), cut at top-level section boundaries. Labels: M (cites F; given), Y (implicit gold),
N. The CRF sees per-sentence features (distance, paragraph, section, cue words, coherence with the neighbour toward
the citation, similarity to the figure, optional extra scores) and learns transitions (run continues / stops).
Prediction: marginal P(Y) >= t. c1, c2, t chosen by inner leave-one-article-out (objective: per-figure mean F1).
M3: from each citation, extend through same-paragraph sentences while cosine(sentence, neighbour toward citation) >= θ
(max 3 steps); θ chosen by inner LOAO.
Protocols: (a) dev nested LOAO + test (fit on dev, scored once); (b) all-13 LOAO (every choice inside the fold).
usage: python eval/figsum/phase5.py [extra_score_json ...]
"""
from __future__ import annotations

import itertools
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import experiments2 as E2  # noqa: E402
import phase1 as P1  # noqa: E402
import phase2 as P  # noqa: E402

HERE = os.path.dirname(__file__)
OFF = re.compile(r"data not shown|not shown|supplement|additional file|\bfig(?:ure)?s?\.?\s*S\d|\btables?\b", re.I)
SETUP = re.compile(r"^(to (test|examine|determine|investigate|assess|evaluate|address|explore|confirm|further|identify|"
                   r"characteri[sz]e|analy[sz]e|define|verify|study)|in order to|we (next|then|also|further|first)|next,|"
                   r"first,|finally,)", re.I)
CONN = re.compile(r"^(however|moreover|furthermore|in addition|additionally|similarly|likewise|in contrast|consistent|"
                  r"interestingly|notably|importantly|thus|therefore|together|taken together|collectively|these|this|"
                  r"such|it|they|in both|both)\b", re.I)
RESULT = re.compile(r"\b(increase[ds]?|decrease[ds]?|reduc\w+|induc\w+|inhibit\w+|significant\w*|higher|lower|"
                    r"enhanc\w+|abolish\w+|block\w+|similar|no effect|did not|failed to|resulted in)\b", re.I)
GROUP_GAP = 6
PAD = 3


def build(D, extra=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    gold_key = {(r["art"], r["fig"], r["i"]): r for r in D.rows}
    seqs = {}
    for aid, a in D.A.items():
        n = len(a.sentences)
        vec = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(a.sentences)
        X = vec.transform(a.sentences)
        sec = [a.top_section[a.sent_para[i]] for i in range(n)]
        for f in D.figs[aid]:
            ms = sorted(a._ment.get(f, []))
            if not ms:
                continue
            groups, cur = [], [ms[0]]
            for m in ms[1:]:
                if m - cur[-1] <= GROUP_GAP and sec[m] == sec[cur[-1]]:
                    cur.append(m)
                else:
                    groups.append(cur)
                    cur = [m]
            groups.append(cur)
            out = []
            for g in groups:
                lo, hi = max(0, g[0] - PAD), min(n - 1, g[-1] + PAD)
                while sec[lo] != sec[g[0]]:
                    lo += 1
                while sec[hi] != sec[g[-1]]:
                    hi -= 1
                items = []
                for i in range(lo, hi + 1):
                    near = min(ms, key=lambda m: abs(m - i))
                    d = i - near
                    toward = i - 1 if d > 0 else (i + 1 if d < 0 else i)
                    toward = min(max(toward, 0), n - 1)
                    s = a.sentences[i]
                    r = gold_key.get((aid, f, i))
                    feat = {
                        "bias": 1.0, f"d={max(-4, min(4, d))}": 1.0,
                        "same_para": float(a.sent_para[i] == a.sent_para[near]),
                        "para_start": float(i == 0 or a.sent_para[i - 1] != a.sent_para[i]),
                        "para_end": float(i == n - 1 or a.sent_para[i + 1] != a.sent_para[i]),
                        "names_other": float(bool(a.cue[i]) and f not in a.cue[i]),
                        "offtarget": float(bool(OFF.search(s))),
                        "setup": float(bool(SETUP.match(s))), "conn": float(bool(CONN.match(s))),
                        "result": float(bool(RESULT.search(s))), "we": float(bool(re.search(r"\bwe\b", s, re.I))),
                        "cite": float(bool(re.search(r"\[\d+|\(\w+ et al", s))),
                        "coh": float((X[i] @ X[toward].T).toarray()[0, 0]) if toward != i else 0.0,
                        "len": min(len(s), 400) / 400,
                    }
                    low = sec[i].lower()
                    for k in ("method", "result", "discussion", "introduction", "background", "abstract"):
                        if k in low:
                            feat["sec_" + k] = 1.0
                    if r is not None:
                        feat["tf_ment_z"] = float(r["tf_ment_z"])
                        feat["tf_cap"] = float(r["tf_cap"])
                        if extra:
                            for name, sc in extra.items():
                                feat[name] = float(sc.get(f"{aid}|{f}|{i}", 0.0))
                    is_m = f in a.cue[i]
                    if is_m:
                        feat["is_mention"] = 1.0
                    lab = "M" if is_m else ("Y" if (r is not None and r["y"]) else "N")
                    items.append((i, feat, lab))
                out.append(items)
            seqs[(aid, f)] = out
    return seqs


def train_crf(seqs, arts, c1, c2):
    import sklearn_crfsuite
    X, Y = [], []
    for (aid, f), blocks in seqs.items():
        if aid in arts:
            for b in blocks:
                X.append([it[1] for it in b])
                Y.append([it[2] for it in b])
    m = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=200, all_possible_transitions=True)
    m.fit(X, Y)
    return m


def predict(m, seqs, arts):
    out = {}
    for (aid, f), blocks in seqs.items():
        if aid in arts:
            for b in blocks:
                marg = m.predict_marginals_single([it[1] for it in b])
                for (i, _, lab), p in zip(b, marg):
                    if lab != "M":
                        k = (aid, f, i)
                        out[k] = max(out.get(k, 0.0), p.get("Y", 0.0))
    return out


TS = [0.2, 0.3, 0.4, 0.5, 0.6]
CS = [(0.0, 0.1), (0.1, 0.1), (0.1, 1.0), (1.0, 0.1), (0.0, 1.0)]


def pick(probs, t):
    return {k for k, p in probs.items() if p >= t}


def m_crf(seqs, fixed=None):
    """inner LOAO over the training articles chooses (c1, c2, t)"""
    def fp(D, train, test):
        if fixed:
            (c1, c2), t = fixed
        else:
            oof = {cfg: {} for cfg in CS}
            for h in train:
                rest = [x for x in train if x != h]
                for cfg in CS:
                    oof[cfg].update(predict(train_crf(seqs, rest, *cfg), seqs, [h]))
            best, bp = -1, None
            for cfg, t in itertools.product(CS, TS):
                f1 = P1.score(D, train, pick(oof[cfg], t))[0]["F1"]
                if f1 > best + 1e-12:
                    best, bp = f1, (cfg, t)
            (c1, c2), t = bp
        m = train_crf(seqs, train, c1, c2)
        return pick(predict(m, seqs, test), t), {"p": ((c1, c2), t)}
    return fp


def tiling(D, arts, th):
    out = set()
    for aid in arts:
        a = D.A[aid]
        X = a._tfidf
        n = len(a.sentences)
        for f in D.figs[aid]:
            for m in a._ment.get(f, []):
                for step in (1, -1):
                    cur = m
                    for _ in range(PAD):
                        nx = cur + step
                        if not (0 <= nx < n) or a.sent_para[nx] != a.sent_para[m] or f in a.cue[nx]:
                            break
                        if a.cue[nx]:  # names another figure
                            break
                        if abs(nx - m) == 1 or float((X[nx] @ X[cur].T).toarray()[0, 0]) >= th:
                            out.add((aid, f, nx))
                            cur = nx
                        else:
                            break
    return out


def m_tiling():
    grid = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 1.1]

    def fp(D, train, test):
        best, bp = -1, None
        for th in grid:
            f1 = P1.score(D, train, tiling(D, train, th))[0]["F1"]
            if f1 > best + 1e-12:
                best, bp = f1, th
        return tiling(D, test, bp), {"p": bp}
    return fp


def fixed_rule(pred):
    return lambda D, train, test: {(r["art"], r["fig"], r["i"]) for a in test for k in D.by_art[a]
                                   for r in (D.rows[k],) if pred(r)}


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    D = P.load()
    for a in D.A.values():
        a._tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit_transform(a.sentences)
    for r in D.rows:
        r["_b0f"] = r["_b0"] and not (r["names_other"] or r["offtarget"] or r["sec_methods"] == 1)
    extra = {}
    for p in sys.argv[1:]:
        extra[os.path.basename(p).split(".")[0]] = json.load(open(p))
    seqs = build(D, extra)
    M = {
        "B0": fixed_rule(lambda r: r["_b0"]),
        "B0f": fixed_rule(lambda r: r["_b0f"]),
        "M3 tiling (coherence-gated runs)": m_tiling(),
        "M1 CRF" + (" + " + "+".join(extra) if extra else ""): m_crf(seqs),
    }
    dev, test = D.split["dev"], D.split["test"]
    allA = dev + test
    res = {"dev": {}, "test": {}, "all13": {}}
    per = {k: {} for k in res}
    for name, fp in M.items():
        cd, ct, pd, pt = P.chosen_sets(D, fp, dev, test)
        ca, pa = set(), []
        for h in allA:
            o = fp(D, [x for x in allA if x != h], [h])
            c, p = o if isinstance(o, tuple) else (o, {})
            ca |= c
            pa.append(p)
        for split, arts, ch, pp in (("dev", dev, cd, pd), ("test", test, ct, pt), ("all13", allA, ca, pa)):
            r, p = P1.score(D, arts, ch)
            r["params"] = pp
            res[split][name], per[split][name] = r, p
        print(f"done {name}: dev {res['dev'][name]['F1']:.3f} test {res['test'][name]['F1']:.3f} "
              f"all13 {res['all13'][name]['F1']:.3f}", flush=True)
    for split in res:
        for n in res[split]:
            res[split][n]["vs_B0"] = E2.bootstrap(per[split]["B0"], per[split][n])
            res[split][n]["vs_B0f"] = E2.bootstrap(per[split]["B0f"], per[split][n])
    P1.B0 = "B0"
    for split in res:
        P1.table({split: res[split]}, split)
        for n in res[split]:
            m, lo, hi = res[split][n]["vs_B0f"]
            print(f"   vs B0f  {n:40s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for n in M:
        print("params", n, "| dev folds", [p.get("p") for p in res["dev"][n]["params"]][:9], "| test",
              res["test"][n]["params"])
    tag = "_".join(extra) if extra else "base"
    json.dump(res, open(os.path.join(HERE, f"phase5_{tag}.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
