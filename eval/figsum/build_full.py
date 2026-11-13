"""Whole-paper data: every (figure F, sentence) pair in all 13 FigSum GS1 articles, labelled from gold.

Includes sentences that cite F (the model must learn those too). Text B:
  "[cites this figure: yes|no | <k> sentences after|before a citation of this figure | same|other paragraph |
    section: <top section> | cites: <other figures / table / supplement / none>] Citing: <nearest sentence citing F>
    Candidate: <sentence>"
far = more than 3 sentences from every citation of F (the trainer subsamples far negatives).
usage: python build_full.py OUT
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiments as E  # noqa: E402
import phase2 as P  # noqa: E402
from build_m2 import caption_text  # noqa: E402

TAB = re.compile(r"\btables?\b", re.I)
SUP = re.compile(r"supplement|additional file|figure s\d|fig\.? ?s\d|data not shown|not shown", re.I)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    D = P.load()
    n = pos = 0
    stats = {}
    with open(os.path.join(out, "full_pairs.jsonl"), "w") as f:
        for aid in sorted(D.A):
            a = D.A[aid]
            for fig in D.figs[aid]:
                gold = a.gold[E.GS][fig]
                ms = a._ment.get(fig, [])
                if not ms:
                    continue
                for i, s in enumerate(a.sentences):
                    cites = fig in a.cue[i]
                    m = min(ms, key=lambda j: abs(j - i))
                    d = abs(i - m)
                    others = sorted(g for g in a.cue[i] if g != fig)
                    c = [f"Figure {g}" for g in others]
                    if TAB.search(s):
                        c.append("table")
                    if SUP.search(s):
                        c.append("supplementary/not shown")
                    pos_txt = ("this is a citation" if cites else
                               f"{d} sentence{'s' if d > 1 else ''} {'after' if i > m else 'before'} a citation of this figure")
                    tag = (f"[cites this figure: {'yes' if cites else 'no'} | {pos_txt} | "
                           f"{'same' if a.sent_para[i] == a.sent_para[m] else 'other'} paragraph | "
                           f"section: {a.top_section[a.sent_para[i]][:40]} | cites: {', '.join(c) if c else 'none'}]")
                    ctx = "(this sentence)" if cites else a.sentences[m]
                    far = (not cites) and d > 3
                    y = int(i in gold)
                    f.write(json.dumps({"key": f"{aid}|{fig}|{i}", "art": aid, "far": far, "cites": cites,
                                        "caption": caption_text(a.figures[fig]),
                                        "text": f"{tag} Citing: {ctx} Candidate: {s}", "label": y}) + "\n")
                    n += 1
                    pos += y
                    k = "cites F" if cites else ("far" if far else "window")
                    stats[k] = [stats.get(k, [0, 0])[0] + 1, stats.get(k, [0, 0])[1] + y]
    print("pairs", n, "gold", pos, {k: f"{v[0]} pairs / {v[1]} gold" for k, v in stats.items()})


if __name__ == "__main__":
    main()
