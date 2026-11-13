"""Reproduce Tables 2-3 of the FigSum+ paper with our re-implementation.
usage: python eval/figsum/reproduce.py GOLD_DIR ARTICLE_ROOT [--out eval/figsum/repro.json]"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
from data import load_all  # noqa: E402
from evaluate import PUBLISHED, evaluate  # noqa: E402
from systems import RANDOM, SYSTEMS, Ctx  # noqa: E402


def main():
    gold_dir, art_root = sys.argv[1], sys.argv[2]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    arts, stats = load_all(gold_dir, art_root)
    ctx = {a.id: Ctx(a) for a in arts}
    res = {"articles": [a.id for a in arts], "missing_fulltext": stats["missing_fulltext"], "results": {}}
    for gs in ("GS1", "GS2"):
        print(f"\n== {gs}  ({len(arts)} articles with full text)")
        print(f"{'system':22s} {'ours: P    R    F1 (mean over figs)':38s} {'micro F1':9s} {'published P   R   F1':22s}")
        res["results"][gs] = {}
        for name, fn in SYSTEMS.items():
            if name in RANDOM:
                runs = [evaluate(arts, gs, lambda a, f, n, s=s: fn(ctx[a.id], f, n, seed=s)) for s in range(20)]
                r = dict(runs[0])
                r["mean"] = [round(statistics.mean(x["mean"][k] for x in runs), 3) for k in range(3)]
                r["micro"] = [round(statistics.mean(x["micro"][k] for x in runs), 3) for k in range(3)]
            else:
                r = evaluate(arts, gs, lambda a, f, n: fn(ctx[a.id], f, n))
            res["results"][gs][name] = r
            pub = PUBLISHED[gs][name]
            m, sd = r["mean"], r.get("sd", [0, 0, 0])
            print(f"{name:22s} {m[0]:.2f}±{sd[0]:.2f} {m[1]:.2f}±{sd[1]:.2f} {m[2]:.2f}±{sd[2]:.2f}       "
                  f"{r['micro'][2]:.2f}      {pub[0]:.2f} {pub[1]:.2f} {pub[2]:.2f}   (figs={r['figures']})")
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
