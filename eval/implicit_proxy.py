"""Proxy benchmark for implicit-reference scorers (no human labels needed).

Take every sentence that explicitly mentions exactly one figure, delete the mention ("Figure 3(b)" -> "the figure"),
and ask each scorer which figure of the paper the sentence is about. The query for each figure is its caption
(+ sub-captions) plus its mention sentences from OTHER paragraphs (leave-one-paragraph-out), i.e. exactly the
information the implicit stage has at run time. Metrics: top-1 accuracy and MRR over figures.

This is an upper-bound-ish proxy: real implicit sentences are vaguer than masked explicit ones.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from figrefs.dense import cross_cached, embed  # noqa: E402
from figrefs.doc import build_doc  # noqa: E402
from figrefs.pdfparse import parse_pdf  # noqa: E402

CAP_PREFIX = re.compile(r"^(Fig\.?|Figure|FIG\.?)\s*\S+\s*[.:|]?\s*")


def mask(text: str, mentions) -> str:
    for m in sorted(mentions, key=lambda m: -m.start):
        text = text[:m.start] + "the figure" + text[m.end:]
    return re.sub(r"(?<![\w.])\d+\([a-z]\)", "the figure", text)


def build_cases(pdf: str):
    doc = build_doc(parse_pdf(pdf))
    figs = [f for f in doc.figures]
    if len(figs) < 3:
        return None
    fid = [f.id for f in figs]
    cases = []
    for s in doc.sentences:
        if len(s.mentions) != 1 or s.mentions[0].figure not in fid:
            continue
        target = s.mentions[0].figure
        queries, pieces = [], []
        for f in figs:
            q = [CAP_PREFIX.sub("", f.caption)] + list(f.subcaptions.values())
            q += [mask(t.text, t.mentions) for t in doc.sentences
                  if t.pid != s.pid and any(m.figure == f.id for m in t.mentions)]
            queries.append(" ".join(q))
            pieces.append([x for x in q if len(x) > 10][:12])
        cases.append({"sentence": mask(s.text, s.mentions), "target": fid.index(target), "queries": queries, "pieces": pieces,
                      "caption_only": [CAP_PREFIX.sub("", f.caption) for f in figs]})
    return cases


def rank_metrics(scores: np.ndarray, target: int) -> tuple[float, float]:
    order = np.argsort(-scores)
    r = int(np.where(order == target)[0][0]) + 1
    return float(r == 1), 1.0 / r


def tfidf_scores(sentence: str, queries: list[str], corpus: list[str]) -> np.ndarray:
    v = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
    v.fit(corpus + queries)
    S = v.transform([sentence])
    Q = v.transform(queries)
    return (S @ Q.T).toarray()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval/arxiv")
    ap.add_argument("--extra", nargs="*", default=["data/tvcg2019_sampling.pdf"])
    ap.add_argument("--methods", default="tfidf,specter2,bge,mpnet,bge-reranker")
    ap.add_argument("--max_papers", type=int, default=40)
    ap.add_argument("--out", default="eval/implicit_proxy.json")
    a = ap.parse_args()
    pdfs = [os.path.join(a.data, d, "paper.pdf") for d in sorted(os.listdir(a.data))
            if os.path.exists(os.path.join(a.data, d, "paper.pdf"))] + a.extra
    methods = a.methods.split(",")
    res: dict = {}
    dump: list = []
    n_papers = 0
    t0 = time.time()
    for pdf in pdfs[: a.max_papers]:
        try:
            import pymupdf
            if pymupdf.open(pdf).page_count > 60:
                continue
            cases = build_cases(pdf)
        except Exception as e:
            print("skip", pdf, e)
            continue
        if not cases:
            continue
        n_papers += 1
        sents = [c["sentence"] for c in cases]
        queries = cases[0]["queries"]  # NB: leave-one-out queries differ per case; handled below for bi-encoders
        for m in methods:
            for c in cases:
                if m == "tfidf":
                    variants = {"": tfidf_scores(c["sentence"], c["queries"], sents)}
                elif m in ("specter2", "bge", "mpnet"):
                    sv = embed(m, [c["sentence"]])[0]
                    concat = embed(m, c["queries"], is_query=True) @ sv
                    cap = embed(m, c["caption_only"], is_query=True) @ sv
                    mx = np.array([max(embed(m, p, is_query=True) @ sv) if p else 0 for p in c["pieces"]])
                    variants = {"+concat": concat, "+caption": cap, "+maxpiece": mx}
                else:
                    cap = cross_cached(m, [(q[:1000], c["sentence"]) for q in c["caption_only"]])
                    mx = np.array([max(cross_cached(m, [(x[:1000], c["sentence"]) for x in p])) if p else 0
                                   for p in c["pieces"]])
                    variants = {"+caption": cap, "+maxpiece": mx}
                c.setdefault("scores", {}).update({m + k: [float(x) for x in v] for k, v in variants.items()})
                for k, sc in variants.items():
                    acc, mrr = rank_metrics(np.asarray(sc), c["target"])
                    res.setdefault(m + k, {"acc": [], "mrr": []})
                    res[m + k]["acc"].append(acc)
                    res[m + k]["mrr"].append(mrr)
        dump.extend({"paper": pdf, "target": c["target"], "scores": c["scores"]} for c in cases)
        json.dump(dump, open(a.out.replace(".json", "_cases.json"), "w"))
        print(f"{os.path.basename(os.path.dirname(pdf)) or pdf}: {len(cases)} cases, "
              f"{len(queries)} figures, {time.time() - t0:.0f}s", flush=True)
    summary = {m: {"n": len(v["acc"]), "top1": round(float(np.mean(v["acc"])), 3) if v["acc"] else None,
                   "mrr": round(float(np.mean(v["mrr"])), 3) if v["mrr"] else None} for m, v in res.items()}
    summary["_papers"] = n_papers
    json.dump(summary, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
