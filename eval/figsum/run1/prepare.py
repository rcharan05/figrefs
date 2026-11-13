"""Final run-1 data file for Kaggle: articles.jsonl + precomputed text variants + fold assignment.

Adds per article:
  masked     every sentence with all figure mentions -> "Fig. [X]"                   (stage-1 input)
  bt         {i: back-translated paraphrase}  (training papers only; mentions restored)
  bt_masked  {i: masked paraphrase}
  cdel       {"i|f": sentence i with its parenthetical citation of f deleted}         (A3)
  fold       0..4 for training papers (5-fold rotation, balanced by number of scored figures), -1 for test
usage: python prepare.py ARTICLES.jsonl BACKTRANS.json OUT.jsonl
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textaug import cite_delete, mask  # noqa: E402

NFOLD = 5


def folds(arts):
    """Greedy: papers by #figures (desc, then id), each to the fold with the fewest figures so far."""
    tr = sorted((a for a in arts if a["split"] == "train"), key=lambda a: (-len(a["figs"]), a["id"]))
    load, out = [0] * NFOLD, {}
    for a in tr:
        k = min(range(NFOLD), key=lambda j: (load[j], j))
        out[a["id"]] = k
        load[k] += len(a["figs"])
    return out, load


def main():
    arts = [json.loads(l) for l in open(sys.argv[1])]
    bt = json.load(open(sys.argv[2]))
    fmap, load = folds(arts)
    n_cdel = 0
    with open(sys.argv[3], "w") as fh:
        for a in arts:
            a["masked"] = [mask(s) for s in a["sentences"]]
            a["bt"] = bt.get(a["id"], {}) if a["split"] == "train" else {}
            a["bt_masked"] = {i: mask(t) for i, t in a["bt"].items()}
            a["cdel"] = {}
            for i, c in enumerate(a["cue"]):
                for f in c:
                    t = cite_delete(a["sentences"][i], f)
                    if t:
                        a["cdel"][f"{i}|{f}"] = t
            n_cdel += len(a["cdel"])
            a["fold"] = fmap.get(a["id"], -1)
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
    print("folds (figures per fold):", load)
    for k in range(NFOLD):
        print(f"  fold {k}: {sorted(p for p, j in fmap.items() if j == k)}")
    print("paraphrases", sum(len(a['bt']) for a in arts), "of", sum(len(a['sentences']) for a in arts if a['split'] == 'train'),
          "training sentences; citation-deleted variants", n_cdel)


if __name__ == "__main__":
    main()
