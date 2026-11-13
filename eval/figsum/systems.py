"""Re-implementation of the FigSum+ systems and baselines (Ramesh, Sethi & Yu 2015), from the paper's descriptions.

Each system: (article, figure number, n) -> ordered list of sentence indices.
  SurfaceCue   all sentences that explicitly refer to the figure
  Paragraph    all sentences of every paragraph containing such a sentence
  Similarity   top-n sentences by cosine of raw term-count vectors with the caption
  TFIDF        top-n sentences by cosine of TF*IDF vectors with the caption (IDF over the article's sentences)
  Hybrid       MEAD (centroid + position) over the sentences of the figure-referring paragraphs, top n
  RandomSent   n random sentences;  RandomPara: first sentence of n random paragraphs
  MEAD         MEAD over the full text, top n (same summary for every figure)
Unspecified details (stop words, MEAD weights, ties) are listed in DEVIATIONS and are our choices.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

DEVIATIONS = [
    "stop words: scikit-learn English list; tokens = lowercase alphanumerics, length >= 2",
    "Similarity uses raw term counts, TFIDF uses tf * log(N/df) over the article's sentences",
    "MEAD: centroid = words with tf*idf above 1/3 of the max in the cluster (top 30 kept), score = centroid + position "
    "(MEAD defaults, weights 1/1), sentences shorter than 9 words skipped (MEAD -length 9)",
    "ties broken by document order; random baselines averaged over 20 seeds",
]

W = re.compile(r"[a-z0-9]{2,}")


def words(s: str) -> list[str]:
    return [w for w in W.findall(s.lower()) if w not in ENGLISH_STOP_WORDS]


def _cos(a: Counter, b: Counter) -> float:
    num = sum(v * b.get(k, 0) for k, v in a.items())
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


class Ctx:
    """Per-article caches."""

    def __init__(self, art):
        self.art = art
        self.bags = [Counter(words(s)) for s in art.sentences]
        N = len(self.bags)
        df = Counter(w for b in self.bags for w in b)
        self.idf = {w: math.log(N / d) for w, d in df.items()}

    def tfidf(self, bag: Counter) -> Counter:
        return Counter({w: c * self.idf.get(w, math.log(len(self.bags))) for w, c in bag.items()})


def surface_cue(ctx: Ctx, fig: int, n: int) -> list[int]:
    return [i for i, c in enumerate(ctx.art.cue) if fig in c]


def paragraph(ctx: Ctx, fig: int, n: int) -> list[int]:
    paras = {ctx.art.sent_para[i] for i in surface_cue(ctx, fig, n)}
    return [i for i, p in enumerate(ctx.art.sent_para) if p in paras]


def _top(scores: list[tuple[float, int]], n: int) -> list[int]:
    return [i for s, i in sorted(scores, key=lambda x: (-x[0], x[1]))[:n]]


def similarity(ctx: Ctx, fig: int, n: int) -> list[int]:
    q = Counter(words(ctx.art.figures[fig]))
    return _top([(_cos(b, q), i) for i, b in enumerate(ctx.bags)], n)


def tfidf(ctx: Ctx, fig: int, n: int) -> list[int]:
    q = ctx.tfidf(Counter(words(ctx.art.figures[fig])))
    return _top([(_cos(ctx.tfidf(b), q), i) for i, b in enumerate(ctx.bags)], n)


def _mead(ctx: Ctx, idx: list[int], n: int) -> list[int]:
    idx = [i for i in idx if len(ctx.art.sentences[i].split()) >= 9] or idx
    if not idx:
        return []
    cl = Counter()
    for i in idx:
        cl.update(ctx.bags[i])
    w = {k: v / len(idx) * ctx.idf.get(k, 0) for k, v in cl.items()}
    if not w:
        return idx[:n]
    mx = max(w.values())
    cent = dict(sorted(((k, v) for k, v in w.items() if v >= mx / 3), key=lambda x: -x[1])[:30])
    C = {i: sum(cent.get(k, 0) for k in ctx.bags[i]) for i in idx}
    cmax = max(C.values()) or 1.0
    m = len(idx)
    score = [(C[i] + (m - r) / m * cmax, i) for r, i in enumerate(idx)]
    return _top(score, n)


def hybrid(ctx: Ctx, fig: int, n: int) -> list[int]:
    return _mead(ctx, paragraph(ctx, fig, n), n)


def mead_full(ctx: Ctx, fig: int, n: int) -> list[int]:
    return _mead(ctx, list(range(len(ctx.bags))), n)


def random_sent(ctx: Ctx, fig: int, n: int, seed: int = 0) -> list[int]:
    r = random.Random(f"{ctx.art.id}-{fig}-{seed}")
    k = len(ctx.bags)
    return r.sample(range(k), min(n, k))


def random_para(ctx: Ctx, fig: int, n: int, seed: int = 0) -> list[int]:
    r = random.Random(f"p{ctx.art.id}-{fig}-{seed}")
    first = {}
    for i, p in enumerate(ctx.art.sent_para):
        first.setdefault(p, i)
    ps = sorted(first)
    return sorted(first[p] for p in r.sample(ps, min(n, len(ps))))


SYSTEMS = {
    "RandomSent": random_sent, "RandomPara": random_para, "MEAD": mead_full,
    "FigSum+ Similarity": similarity, "FigSum+ TFIDF": tfidf, "FigSum+ SurfaceCue": surface_cue,
    "FigSum+ Paragraph": paragraph, "FigSum+ Hybrid": hybrid,
}
RANDOM = {"RandomSent", "RandomPara"}
