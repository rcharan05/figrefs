"""How far are implicit references from the figure?  (dev articles only unless --all)

For every GS1 figure and every sentence that does NOT name the figure ("candidate"), measure
  d_sent   sentences to the nearest sentence that names the figure (0 impossible; 1 = adjacent)
  d_para   paragraphs to the nearest paragraph that names the figure (0 = same paragraph)
  dir      before / after the nearest mention
  d_fig    paragraphs to where the <fig> element is placed in the XML (only where the XML places it inline)
and compare the distribution for implicit gold sentences (positives) with all candidates.
usage: python eval/figsum/distance_analysis.py [--all] [--gs GS1]
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from data import load_all  # noqa: E402


def rows_for(arts, gs):
    rows = []
    for a in arts:
        for fig, gold in a.gold.get(gs, {}).items():
            if fig not in a.figures:
                continue
            ment = [i for i, c in enumerate(a.cue) if fig in c]
            mpar = sorted({a.sent_para[i] for i in ment})
            for i, s in enumerate(a.sentences):
                if fig in a.cue[i]:
                    continue
                r = {"art": a.id, "fig": fig, "i": i, "pos": i in gold, "has_mention": bool(ment),
                     "names_other": bool(a.cue[i] - {fig}), "section": a.top_section[a.sent_para[i]]}
                if ment:
                    j = min(ment, key=lambda m: abs(m - i))
                    r["d_sent"] = abs(j - i)
                    r["dir"] = "after" if i > j else "before"
                    r["d_para"] = min(abs(a.sent_para[i] - p) for p in mpar)
                    r["same_sec_as_mention"] = a.top_section[a.sent_para[i]] in {a.top_section[p] for p in mpar}
                if fig in a.fig_pos:
                    r["d_fig"] = abs(a.sent_para[i] - a.fig_pos[fig])
                rows.append(r)
    return rows


def table(rows, key, bins):
    pos = [r for r in rows if r["pos"] and key in r]
    allc = [r for r in rows if key in r]
    print(f"\n{key}: {len(pos)} implicit gold sentences, {len(allc)} candidates")
    print(f"{'bin':>8s} {'gold':>5s} {'cands':>6s} {'P(gold|bin)':>11s} {'cum. gold recall':>16s} {'cum. cands':>10s}")
    cg = cc = 0
    for lo, hi in bins:
        g = sum(1 for r in pos if lo <= r[key] <= hi)
        c = sum(1 for r in allc if lo <= r[key] <= hi)
        cg += g
        cc += c
        lab = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**6 else f">={lo}")
        print(f"{lab:>8s} {g:5d} {c:6d} {g / max(1, c):11.3f} {cg / max(1, len(pos)):16.3f} {cc:10d}")
    vals = sorted(r[key] for r in pos)
    if vals:
        q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]  # noqa: E731
        print(f"  gold percentiles: 50%={q(.5)}  75%={q(.75)}  90%={q(.9)}  95%={q(.95)}  99%={q(.99)}  max={vals[-1]}")
    return vals


def main():
    gs = sys.argv[sys.argv.index("--gs") + 1] if "--gs" in sys.argv else "GS1"
    arts, _ = load_all(os.path.join(os.path.dirname(__file__), "gold"), "/home/charan/bench/figsum")
    split = json.load(open(os.path.join(os.path.dirname(__file__), "split.json")))
    if "--all" not in sys.argv:
        arts = [a for a in arts if a.id in split["dev"]]
    rows = rows_for(arts, gs)
    pos = [r for r in rows if r["pos"]]
    print(f"{gs}, articles: {[a.id for a in arts]}")
    print(f"candidates (sentence x figure, not naming the figure): {len(rows)}; implicit gold: {len(pos)} "
          f"(base rate {len(pos) / len(rows):.4f})")
    nm = [r for r in pos if not r["has_mention"]]
    print(f"implicit gold for figures never named in the text: {len(nm)}")
    print(f"implicit gold that names ANOTHER figure: {sum(r['names_other'] for r in pos)} "
          f"(candidates naming another figure: {sum(r['names_other'] for r in rows)})")
    sb = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 20), (21, 40), (41, 10**7)]
    pb = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 10**7)]
    table(rows, "d_sent", sb)
    table(rows, "d_para", pb)
    table(rows, "d_fig", pb)
    d = collections.Counter(r["dir"] for r in pos if "dir" in r)
    dc = collections.Counter(r["dir"] for r in rows if "dir" in r)
    print(f"\ndirection of nearest mention (gold): {dict(d)}   (all candidates: {dict(dc)})")
    s = collections.Counter(r["same_sec_as_mention"] for r in pos if "same_sec_as_mention" in r)
    print(f"gold in the same top-level section as a mention: {s[True]}/{s[True] + s[False]}")
    sec = collections.Counter(r["section"] for r in pos)
    secall = collections.Counter(r["section"] for r in rows)
    print("\nsection of implicit gold (gold / candidates / rate):")
    for k, v in sec.most_common(12):
        print(f"  {k[:30]:30s} {v:4d} {secall[k]:6d} {v / secall[k]:.3f}")
    # sentences that are gold for several figures: are they generic?
    multi = collections.Counter((r["art"], r["i"]) for r in pos)
    print(f"\nimplicit gold sentences shared by >1 figure: {sum(1 for v in multi.values() if v > 1)} of {len(multi)}")


if __name__ == "__main__":
    main()
