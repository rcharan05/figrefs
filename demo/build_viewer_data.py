"""Precomputed per-figure outputs of every method on the 18 FigSum+ papers, for the demo viewer.

Methods: Gold, FigSum+ Paragraph (published best, our re-implementation), B0f (rule), C0L (DeBERTa-v3-large
pairwise, +E, raw cut-off), SPAN (ModernBERT-large span tagger, +E, raw), SPAN_ES (span tagger with early stopping,
+E, raw), ENS (C0L+SPAN, anchored: the sealed-test winner), ENS3 (C0L+SPAN_ES+SPAN_VIS, anchored: best out-of-fold).
Honest provenance:
  training papers (15)  learned methods = out-of-fold scores (the model never trained on the paper), cut-off chosen
                        by the same 5-fold rotation as the evaluation (phase8.rotate)
  test papers (3)       final models (trained on the 15), cut-off chosen on all 15 papers' out-of-fold scores;
                        SPAN_ES and ENS3 used these papers for early stopping -> marked "validation"
Also writes thresholds.json (cut-offs chosen on all 15 papers) for the any-paper inference script.
usage: python demo/build_viewer_data.py OUT_DIR
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval", "figsum", "run1"))
sys.path.insert(0, os.path.join(ROOT, "eval", "figsum"))
import phase8 as P  # noqa: E402
import systems as SY  # noqa: E402

B = "/home/charan/bench"
DATA = f"{B}/run1/run1_data.jsonl"
OOF = {"C0L": f"{B}/run2b/results/oof/C0L.json", "SPAN": f"{B}/run3/results/SPAN/oof/SPAN.json",
       "SPAN_ES": f"{B}/run4/results/SPAN_ES/oof/SPAN_ES.json", "SPAN_VIS": f"{B}/run4/results/SPAN_VIS/oof/SPAN_VIS.json"}
FINAL = {"C0L": f"{B}/run2b/results/test_SEALED/C0L.json", "SPAN": f"{B}/run3/results/SPAN/test_SEALED/SPAN.json",
         "SPAN_ES": f"{B}/run4/results/SPAN_ES/val/SPAN_ES.json", "SPAN_VIS": f"{B}/run4/results/SPAN_VIS/val/SPAN_VIS.json"}
RG = [round(float(x), 2) for x in np.arange(-8.0, 8.001, 0.1)]
AG = [round(float(x), 2) for x in np.arange(-10.0, 4.001, 0.1)]
METHODS = [  # key, label, parts, decision
    ("C0L", "C0L · DeBERTa-large pairwise", ["C0L"], "raw"),
    ("SPAN", "SPAN · ModernBERT span tagger", ["SPAN"], "raw"),
    ("SPAN_ES", "SPAN_ES · span tagger + early stopping", ["SPAN_ES"], "raw"),
    ("ENS", "Ensemble C0L+SPAN (test winner)", ["C0L", "SPAN"], "anchored"),
    ("ENS_ES", "Ensemble C0L+SPAN_ES", ["C0L", "SPAN_ES"], "anchored"),
    ("ENS3", "Ensemble C0L+SPAN_ES+SPAN_VIS (best out-of-fold)", ["C0L", "SPAN_ES", "SPAN_VIS"], "anchored"),
]
VAL_USED = {"SPAN_ES", "ENS_ES", "ENS3"}


def keyed(S):
    return {tuple(int(x) if j else x for j, x in enumerate(k.split("|")[:3])): v for k, v in S.items()}


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


def combine(parts, src):
    keys = set.intersection(*[set(src[c]) for c in parts])
    return {k: float(np.mean([src[c][k] for c in parts])) for k in keys}


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    raw = [json.loads(l) for l in open(DATA)]
    arts = {d["id"]: P.A(d) for d in raw}
    train = sorted(p for p, a in arts.items() if a.split == "train")
    test = sorted(p for p, a in arts.items() if a.split == "test")
    oof = {c: keyed(json.load(open(f))) for c, f in OOF.items()}
    fin = {c: keyed(json.load(open(f))) for c, f in FINAL.items()}
    preds = {m[0]: {} for m in METHODS}
    thresholds = {}
    for key, label, parts, dec in METHODS:
        Ro, Rt = combine(parts, oof), combine(parts, fin)
        Zo, Zt = (anchored(arts, Ro), anchored(arts, Rt)) if dec == "anchored" else (Ro, Rt)
        P.ZG_ACTIVE = AG if dec == "anchored" else RG
        sel, ts = P.rotate(arts, train, Zo, True)                 # out-of-fold picks on training papers
        preds[key].update(sel)
        t = max(P.ZG_ACTIVE, key=lambda t: (P.metrics(arts, train, P.predict(arts, train, Zo, t, True))["F1"], -abs(t - 1)))
        thresholds[key] = {"parts": parts, "decision": dec, "t": t, "label": label}
        preds[key].update(P.predict(arts, test, Zt, t, True))      # final models on the test papers
        m_tr = P.metrics(arts, train, sel)["F1"]
        m_te = P.metrics(arts, test, P.predict(arts, test, Zt, t, True))["F1"]
        print(f"{key:8s} t={t:5}  train(OOF) F1 {m_tr:.3f}   test-papers F1 {m_te:.3f}")
    papers = []
    for d in raw:
        a, pid = arts[d["id"]], d["id"]
        ctx = SY.Ctx(a)
        meth = {"GOLD": {}, "PARA": {}, "B0F": {}}
        for f in a.figs:
            meth["GOLD"][str(f)] = sorted(a.gold[f])
            meth["PARA"][str(f)] = sorted(SY.paragraph(ctx, f, 0))
            meth["B0F"][str(f)] = sorted(P._b0f(a, f))
            for key, *_ in METHODS:
                meth.setdefault(key, {})[str(f)] = sorted(preds[key].get((pid, f), set()))
        prov = {k: ("out-of-fold" if a.split == "train" else ("validation" if k in VAL_USED else "held-out test"))
                for k, *_ in METHODS}
        papers.append({"id": pid, "split": a.split, "source": d["source"], "sentences": d["sentences"],
                       "sent_para": d["sent_para"], "para_section": d["para_section"],
                       "captions": {str(f): d["captions"].get(str(f), "") for f in a.figs},
                       "cue": d["cue"], "figs": a.figs, "methods": meth, "provenance": prov, "has_gold": True})
    labels = {"GOLD": "Gold (human labels)", "PARA": "FigSum+ Paragraph (published best)", "B0F": "B0f rule (ours)"}
    labels.update({k: l for k, l, *_ in METHODS})
    json.dump({"papers": papers, "labels": labels, "order": list(labels)}, open(os.path.join(out, "viewer_data.json"), "w"),
              ensure_ascii=False)
    json.dump(thresholds, open(os.path.join(out, "thresholds.json"), "w"), indent=1)
    print("papers", len(papers), "->", out)


if __name__ == "__main__":
    main()
