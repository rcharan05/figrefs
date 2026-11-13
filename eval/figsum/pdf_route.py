"""End-to-end PDF route: GROBID 0.9.1 TEI -> article model -> same features -> models trained on the dev XML.

Gold sentences are re-aligned to GROBID's sentences (token F1 >= 0.6). Only test articles that have a PDF.
usage: python eval/figsum/pdf_route.py features        # build features for the test articles from TEI
       python eval/figsum/pdf_route.py eval NAME...    # score methods on the PDF route
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import experiments as E  # noqa: E402
import experiments2 as E2  # noqa: E402
from data import load_all_tei  # noqa: E402
from features import article_features  # noqa: E402

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "features_GS1_tei091.json")


def test_tei_articles():
    split = json.load(open(os.path.join(HERE, "split.json")))
    arts, _ = load_all_tei(os.path.join(HERE, "gold"), "/home/charan/bench/figsum", "091")
    return [a for a in arts if a.id in split["test"]]


def main():
    if sys.argv[1] == "features":
        rows = []
        for a in test_tei_articles():
            rows += article_features(a, "GS1")
            print(a.id, len(rows), flush=True)
        json.dump(rows, open(OUT, "w"))
        return
    D = E.Data()
    tei = {a.id: a for a in test_tei_articles()}
    test = sorted(tei)
    rows = json.load(open(OUT))
    # swap the test articles' XML rows for the TEI rows
    D.rows = [r for r in D.rows if r["art"] not in tei] + rows
    for r in D.rows:
        r["d_para_log"] = __import__("math").log1p(r["d_para"])
    D.by_art = {}
    for k, r in enumerate(D.rows):
        D.by_art.setdefault(r["art"], []).append(k)
    for aid, a in tei.items():
        D.A[aid] = a
        D.figs[aid] = sorted(f for f in a.gold["GS1"] if f in a.figures and a.gold["GS1"][f])
    E2.add_fig_features(D)
    dev = D.split["dev"]
    lost = {aid: sorted(set(f for f in json.load(open(os.path.join(HERE, "split.json")))["test"] and []))
            for aid in tei}
    print(f"== PDF route (GROBID 0.9.1), test articles with PDFs: {test}, figures {sum(len(D.figs[a]) for a in test)}")
    for name in ("FigSum+ SurfaceCue", "FigSum+ Paragraph", "FigSum+ Hybrid"):
        print(E.fmt(name + " [published system]", E.figsum_baseline(D, name, test)))
    M = E2.methods()
    res = {}
    for name in sys.argv[2:]:
        out = M[name](D, dev, test)
        chosen, params = out if isinstance(out, tuple) else (out, {})
        r = E.evaluate(D, test, chosen)
        res[name] = r
        print(E.fmt(name, r), flush=True)
    json.dump(res, open(os.path.join(HERE, "exp_pdf_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
