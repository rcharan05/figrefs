"""Fine-tune data for idea 2: FigSum GS1 window candidates with real labels.

One example per (article, figure F, candidate sentence within 3 sentences of a citation of F):
  text_a = F's caption (without the "Figure N." prefix)
  text_b = "[<after|before> <d>, <same|other> paragraph] Citing: <nearest sentence citing F> Candidate: <candidate>"
  label  = GS1 implicit gold (1/0)
Writes OUT/ft_window.jsonl (all 13 articles; the Kaggle script does leave-one-article-out).
usage: python build_ft.py OUT
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase2 as P  # noqa: E402
from build_m2 import caption_text  # noqa: E402


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    D = P.load()
    n = pos = 0
    with open(os.path.join(out, "ft_window.jsonl"), "w") as f:
        for r in D.rows:
            if r["d_sent"] > P.WIN:
                continue
            a = D.A[r["art"]]
            i = r["i"]
            m = min(a._ment[r["fig"]], key=lambda j: abs(j - i))
            tag = (f"[{'after' if i > m else 'before'} {abs(i - m)}, "
                   f"{'same' if a.sent_para[i] == a.sent_para[m] else 'other'} paragraph]")
            f.write(json.dumps({"key": f"{r['art']}|{r['fig']}|{i}", "art": r["art"],
                                "caption": caption_text(a.figures[r["fig"]]),
                                "text": f"{tag} Citing: {a.sentences[m]} Candidate: {a.sentences[i]}",
                                "label": int(r["y"])}) + "\n")
            n += 1
            pos += r["y"]
    print("examples", n, "positive", pos, "articles", len({r['art'] for r in D.rows}))


if __name__ == "__main__":
    main()
