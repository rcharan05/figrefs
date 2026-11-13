"""Any-paper inference for the Figure Reference Explorer.

  python demo/demo_infer.py score  ARTICLES.jsonl OUT_DIR      (GPU, Kaggle) score every paper with every trained
        model found under /kaggle/input (or $DEMO_MODELS): each model folder has input_format.json, whose "config"
        says what it is (C0L, SPAN, SPAN_ES). Writes OUT_DIR/scores_<cfg>.json. Missing models are skipped.
  python demo/demo_infer.py report ARTICLES.jsonl OUT_DIR [--labels DIR] [--gold both|union|claude|gemini]
        (CPU, anywhere) reads every OUT_DIR/scores_*.json, adds FigSum+ Paragraph and B0f (no model needed), builds the
        ensembles whose parts are present and writes OUT_DIR/results.json for the viewer's "Load results file" button.
        --labels DIR: AI annotations <paper>_<labeler>.json ({"figures": {"1": [ids], ...}}); each labeler becomes a
        method and GOLD = sentences both agree on (--gold both, default) / either (union) / one labeler; prints the
        benchmark table (per-figure mean P/R/F1 of every method vs that gold, and labeler-vs-labeler agreement).
Decision rules and cut-offs are exactly the ones evaluated (thresholds.json, chosen on the 15 training papers'
out-of-fold scores): singles = raw logit >= t, ensembles = mean raw logit minus the mean over the figure's citing
sentences >= t; citing sentences are always included (+E).
ARTICLES.jsonl comes from demo/article_from_source.py (LaTeX source dir, JATS XML or PDF).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "eval"), os.path.join(ROOT, "eval", "figsum"), os.path.join(ROOT, "eval", "figsum", "run1")):
    sys.path.insert(0, p)
CAPPFX = re.compile(r"^\s*(?:Fig(?:ure)?\.?|FIG\.?)\s*\d+\s*[.:|]?\s*", re.I)
SCORABLE = ("C0L", "SPAN", "SPAN_ES")


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def model_arts(raw):
    """Articles in the trainers' in-memory format (as run4_kaggle.load_articles)."""
    out = {}
    for d in raw:
        a = dict(d)
        a["cue"] = [set(c) for c in d["cue"]]
        a["gold"] = {}
        a["captions"] = {int(k): CAPPFX.sub("", v).strip()[:700] for k, v in d["captions"].items()}
        a.setdefault("masked", a["sentences"])
        for k in ("bt", "bt_masked", "cdel"):
            a.setdefault(k, {})
        out[a["id"]] = a
    return out


def find_models():
    roots = [os.environ["DEMO_MODELS"]] if os.environ.get("DEMO_MODELS") else ["/kaggle/input", "/kaggle/working"]
    found = {}
    for r in roots:
        for fp in glob.glob(os.path.join(r, "**", "input_format.json"), recursive=True):
            try:
                cfg = json.load(open(fp)).get("config")
            except Exception:
                continue
            if cfg in SCORABLE and cfg not in found:
                found[cfg] = os.path.dirname(fp)
    return found


def pick_device(torch):
    want = os.environ.get("DEMO_DEVICE", "auto")
    if want != "auto":
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")                    # Apple-silicon GPU
    return torch.device("cpu")


def cmd_score(arts_path, out, only=None):
    os.environ.setdefault("R1_OUT", os.path.join(out, "_r4"))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import time
    import torch
    from transformers import AutoTokenizer
    import run4_kaggle as R
    os.makedirs(out, exist_ok=True)
    arts = model_arts(load(arts_path))
    ids = sorted(arts)
    models = find_models()
    if only:
        models = {c: p for c, p in models.items() if c in only}
    print("models found:", models or "NONE (attach the notebook outputs / datasets holding models/<cfg>/)")
    dev = pick_device(torch)
    print("device:", dev)
    for cfg, path in models.items():
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(path)
        model = (R.load_model(path) if cfg == "C0L" else R.make_span_model(path)).to(dev)
        S = {}
        for k, pid in enumerate(ids):
            t1 = time.time()
            S.update(R.score(model, tok, arts, [pid], "C0L") if cfg == "C0L" else R.score_span(model, tok, arts, [pid], cfg))
            print(f"  {cfg} paper {k + 1}/{len(ids)} {pid}: {len(arts[pid]['figs'])} figures, {time.time() - t1:.0f}s", flush=True)
        json.dump(S, open(os.path.join(out, f"scores_{cfg}.json"), "w"))
        print(f"{cfg}: {len(S)} sentence scores -> scores_{cfg}.json  ({time.time() - t0:.0f}s)")
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        elif dev.type == "mps":
            torch.mps.empty_cache()


# ------------------------------------------------------------------ report (CPU)
def anchored(cue, R):
    by = {}
    for (p, f, i), v in R.items():
        by.setdefault((p, f), {})[i] = v
    out = {}
    for (p, f), d in by.items():
        E = [i for i, c in enumerate(cue[p]) if f in c and i in d]
        ref = np.mean([d[i] for i in E]) if E else np.median(list(d.values()))
        for i, v in d.items():
            out[(p, f, i)] = v - ref
    return out


def labeler_files(d):
    out = {}
    for fp in sorted(glob.glob(os.path.join(d, "*.json"))):
        m = re.match(r"(.+)_([A-Za-z0-9-]+)\.json$", os.path.basename(fp))
        if not m:
            continue
        try:
            j = json.load(open(fp))
        except Exception as ex:
            print("SKIP (not valid JSON):", fp, ex)
            continue
        figs = j.get("figures", j)
        out.setdefault(m.group(1), {})[m.group(2).upper()] = {str(k): sorted({int(x) for x in v}) for k, v in figs.items()}
    return out


def prf(pred, gold):
    pred, gold = set(pred), set(gold)
    if not gold and not pred:
        return None
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def cmd_report(arts_path, out, labels_dir=None, gold_mode="both"):
    import phase8 as P
    import systems as SY
    os.makedirs(out, exist_ok=True)
    raw = load(arts_path)
    arts = {d["id"]: P.A(d) for d in raw}
    cue = {p: a.cue for p, a in arts.items()}
    th = json.load(open(os.path.join(HERE, "thresholds.json")))
    S = {}
    for fp in glob.glob(os.path.join(out, "scores_*.json")):
        cfg = os.path.basename(fp)[7:-5]
        S[cfg] = {(k.split("|")[0], int(k.split("|")[1]), int(k.split("|")[2])): v for k, v in json.load(open(fp)).items()}
    print("score files:", sorted(S) or "none (rule methods only)")
    learned = {}
    for key, spec in th.items():
        if not all(c in S for c in spec["parts"]):
            continue
        keys = set.intersection(*[set(S[c]) for c in spec["parts"]])
        R = {k: float(np.mean([S[c][k] for c in spec["parts"]])) for k in keys}
        Z = anchored(cue, R) if spec["decision"] == "anchored" else R
        learned[key] = (spec["label"], P.predict(arts, sorted(arts), Z, spec["t"], True))
    ann = labeler_files(labels_dir) if labels_dir else {}
    labels = {"GOLD": {"both": "Gold = AI labelers agree", "union": "Gold = either AI labeler"}.get(gold_mode, f"Gold = {gold_mode}"),
              "PARA": "FigSum+ Paragraph (published best)", "B0F": "B0f rule (ours)"}
    labels.update({k: v[0] for k, v in learned.items()})
    papers, table = [], {}
    for d in raw:
        a, pid = arts[d["id"]], d["id"]
        ctx = SY.Ctx(a)
        meth = {"PARA": {}, "B0F": {}}
        for f in a.figs:
            meth["PARA"][str(f)] = sorted(SY.paragraph(ctx, f, 0))
            meth["B0F"][str(f)] = sorted(P._b0f(a, f))
            for k, (_, sel) in learned.items():
                meth.setdefault(k, {})[str(f)] = sorted(sel.get((pid, f), set()))
        A = ann.get(pid, {})
        for lab, figs in A.items():
            labels.setdefault(lab, f"{lab.title()} (AI labeler)")
            meth[lab] = {str(f): figs.get(str(f), []) for f in a.figs}
        if A:
            names = sorted(A)
            g = {}
            for f in map(str, a.figs):
                sets = [set(meth[n][f]) for n in names]
                if gold_mode == "both":
                    g[f] = sorted(set.intersection(*sets))
                elif gold_mode == "union":
                    g[f] = sorted(set.union(*sets))
                else:
                    g[f] = sorted(set(meth.get(gold_mode.upper(), {}).get(f, [])))
            meth = {"GOLD": g, **meth}
            for k in meth:
                if k == "GOLD":
                    continue
                for f in map(str, a.figs):
                    r = prf(meth[k][f], g[f])
                    if r and g[f]:
                        table.setdefault(k, {}).setdefault(pid, []).append(r)
            if len(names) == 2:
                for f in map(str, a.figs):
                    r = prf(meth[names[0]][f], meth[names[1]][f])
                    if r:
                        table.setdefault(f"{names[0]} vs {names[1]}", {}).setdefault(pid, []).append(r)
        prov = {k: "new paper (final model)" for k in learned}
        prov.update({k: "AI labels" for k in A})
        papers.append({"id": pid, "title": d.get("title", ""), "split": "new", "source": d.get("source", ""),
                       "sentences": d["sentences"], "sent_para": d["sent_para"], "para_section": d["para_section"],
                       "captions": d["captions"], "cue": d["cue"], "figs": d["figs"], "methods": meth,
                       "provenance": prov, "has_gold": "GOLD" in meth})
    order = [k for k in ["GOLD", "PARA", "B0F", "C0L", "SPAN", "SPAN_ES", "ENS", "ENS_ES"] if k in labels] + \
            [k for k in labels if k not in ("GOLD", "PARA", "B0F", "C0L", "SPAN", "SPAN_ES", "ENS", "ENS_ES", "ENS3")]
    if not ann:
        labels.pop("GOLD")
        order.remove("GOLD") if "GOLD" in order else None
    json.dump({"papers": papers, "labels": labels, "order": order}, open(os.path.join(out, "results.json"), "w"),
              ensure_ascii=False)
    print(f"{len(papers)} papers, methods {order} -> {os.path.join(out, 'results.json')}")
    if table:
        pids = [d["id"] for d in raw if d["id"] in ann]
        print(f"\nBenchmark vs GOLD ({labels['GOLD']}); per-figure mean F1 (figures with a non-empty gold); "
              f"last column = all figures pooled  P / R / F1")
        print(f"{'method':22s}" + "".join(f"{p[:12]:>13s}" for p in pids) + f"{'ALL P/R/F1':>22s}")
        for k, per in table.items():
            row = f"{k:22s}"
            allv = []
            for p in pids:
                v = per.get(p, [])
                allv += v
                row += f"{np.mean([x[2] for x in v]):13.3f}" if v else f"{'-':>13s}"
            m = np.mean(allv, axis=0) if allv else (0, 0, 0)
            print(row + f"   {m[0]:.3f} / {m[1]:.3f} / {m[2]:.3f}")
        with open(os.path.join(out, "benchmark.json"), "w") as fh:
            json.dump({k: {p: np.mean(v, axis=0).tolist() for p, v in per.items()} for k, per in table.items()}, fh, indent=1)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3 or args[0] not in ("score", "report"):
        raise SystemExit(__doc__)
    if args[0] == "score":
        cmd_score(args[1], args[2])
    else:
        lab = args[args.index("--labels") + 1] if "--labels" in args else None
        gm = args[args.index("--gold") + 1] if "--gold" in args else "both"
        cmd_report(args[1], args[2], lab, gm)
