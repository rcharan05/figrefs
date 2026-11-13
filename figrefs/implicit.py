"""Implicit references: sentences outside the explicit paragraphs that still discuss a figure.

v1 combines interpretable signals (no neural model needed):
  continuation  paragraph right after an explicit paragraph that opens with a back-reference ("This shows...")
  subfig        a bare "(b)" / "in (a)" that resolves to the most recently mentioned figure
  lexical       TF-IDF similarity to the figure's caption + mention sentences, normalised across figures
  proximity     closeness (in paragraphs) to the figure's explicit mentions
An optional dense scorer (SPECTER2 / cross-encoder) plugs in through `dense_scores`.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .doc import Doc, Sentence

ANAPHORA_RE = re.compile(
    r"^(this|these|those|such|here|thus|hence|therefore|similarly|likewise|comparing|compared|note that|"
    r"the (above|previous|preceding|resulting|left|right|top|bottom|first|second|third|last|former|latter)\b|"
    r"in (this|these|the (above|left|right|top|bottom|first|second|middle))\b|"
    r"as (can be |is |we can )?(seen|shown|observed|illustrated|depicted|evident)|"
    r"we (can )?(also )?(see|observe|notice|note)|it (can be|is) (seen|observed|clear|evident|noticeable)|"
    r"from (this|these|the (above|figure|results|plots?|images?|visuali[sz]ations?))|"
    r"(the|our) (results?|images?|plots?|renderings?|visuali[sz]ations?) (show|demonstrate|indicate|reveal|suggest)|"
    r"(on|to) the (left|right)|(at|in) the (top|bottom)|(clearly|visually))\b",
    re.I,
)
VISUAL_RE = re.compile(
    r"\b(shown|show[s]?|depict\w*|illustrat\w*|visuali[sz]\w*|render\w*|plot\w*|image\w*|colou?r\w*|"
    r"left|right|top|bottom|curve\w*|observ\w*|see|seen|visible|highlight\w*|appear\w*|panel|inset|"
    r"isocontour\w*|isosurface\w*|histogram\w*|scatter|bar|line|axis|legend|red|blue|green|orange|black|white)\b",
    re.I,
)
SUBFIG_BARE_RE = re.compile(r"(?<![\w)])\(([a-h])\)(?!\s*[A-Z][a-z]+\s*[:.])")


@dataclass
class ImplicitHit:
    sentence: Sentence
    score: float
    signals: dict[str, float] = field(default_factory=dict)


def _explicit_paras(doc: Doc) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {f.id: set() for f in doc.figures}
    for s in doc.sentences:
        for m in s.mentions:
            out.setdefault(m.figure, set()).add(s.pid)
    return out


def _fig_query(doc: Doc, fid: str) -> str:
    f = next(f for f in doc.figures if f.id == fid)
    parts = [re.sub(r"^(Fig\.?|Figure|FIG\.?)\s*\d+\s*[.:|]?", "", f.caption)] + list(f.subcaptions.values())
    parts += [s.text for s in doc.sentences if any(m.figure == fid for m in s.mentions)]
    return " ".join(parts)


def _row_z(m: np.ndarray) -> np.ndarray:
    """z-score each sentence's similarities across figures (which figure is it MOST about?)"""
    return (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-6)


def score_implicit(doc: Doc, dense: str | None = "bge", rerank: str | None = "bge-reranker",
                   threshold: float = 0.5, keep: float = 0.2, top_k: int = 25,
                   **_ignored) -> dict[str, list[ImplicitHit]]:
    """Score every (sentence, figure) pair outside the figure's explicit paragraphs.

    Features (all reported in `signals`):
      attr        P(figure | sentence): softmax over the paper's figures of z-scored TF-IDF (+ BGE) similarity
                  to the figure's caption (+ sub-captions, + mention sentences for TF-IDF)
      rerank      cross-encoder relevance of (caption, sentence) in 0..1, computed for each figure's top-k
      proximity   exp(-(d-1)/2), d = paragraph distance to the figure's nearest explicit paragraph
      continuation / subfig   structural back-references (see module docstring)
      visual      sentence uses visual-description language
    """
    figs = [f.id for f in doc.figures]
    if not figs or not doc.sentences:
        return {}
    exp = _explicit_paras(doc)
    para_index = {p.id: i for i, p in enumerate(doc.paragraphs)}
    sections = {p.id: p.section for p in doc.paragraphs}
    corpus = [s.text for s in doc.sentences]
    captions = [_caption_query(doc, fid) for fid in figs]

    # ---------------- TF-IDF (caption + subcaptions + mention sentences)
    queries = [_fig_query(doc, fid) for fid in figs]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=1,
                          token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9\-]+\b")
    vec.fit(corpus + queries)
    tf = (vec.transform(corpus) @ vec.transform(queries).T).toarray()
    zsum = _row_z(tf) if len(figs) > 1 else tf * 0

    # ---------------- dense bi-encoder (caption + subcaptions only: best in the proxy benchmark)
    if dense:
        try:
            from .dense import embed
            E = embed(dense, corpus)
            Q = embed(dense, captions, is_query=True)
            if len(figs) > 1:
                zsum = zsum + _row_z(E @ Q.T)
        except Exception as e:  # models unavailable -> lexical only
            print(f"[figrefs] dense scorer unavailable ({e}); using TF-IDF only")
            dense = None
    attr = np.exp(zsum) / np.exp(zsum).sum(1, keepdims=True) if len(figs) > 1 else np.ones_like(tf)

    # ---------------- most recent figure context for bare "(b)"
    last_fig_at: dict[int, tuple[str, int] | None] = {}
    last: tuple[str, int] | None = None
    for s in doc.sentences:
        if s.mentions:
            last = (s.mentions[-1].figure, s.pid)
        last_fig_at[s.gid] = last

    out: dict[str, list[ImplicitHit]] = {}
    for j, fid in enumerate(figs):
        fig = next(f for f in doc.figures if f.id == fid)
        ep = exp.get(fid, set())
        hits = []
        for s in doc.sentences:
            if s.pid in ep:
                continue  # already covered by an explicit paragraph
            sig: dict[str, float] = {}
            if ep:
                d = min(abs(para_index[s.pid] - para_index[e]) for e in ep)
                prox = math.exp(-(d - 1) / 2.0)
            else:
                prox = 0.15
            prev_pid = s.pid - 1
            if prev_pid in ep and sections.get(prev_pid) == sections.get(s.pid) and not doc.by_para[s.pid][0].mentions:
                if ANAPHORA_RE.match(doc.by_para[s.pid][0].text):
                    sig["continuation"] = 0.7 if s.sid == 0 else 0.45
            if not s.mentions:
                letters = SUBFIG_BARE_RE.findall(s.text)
                lf = last_fig_at.get(s.gid)
                if letters and lf and lf[0] == fid and s.pid - lf[1] <= 1:
                    if not fig.subcaptions or any(x in fig.subcaptions for x in letters):
                        sig["subfig"] = 0.6
            sig["attr"] = round(float(attr[s.gid, j]), 3)
            sig["proximity"] = round(prox, 3)
            visual = bool(VISUAL_RE.search(s.text))
            other_fig = any(m.figure != fid for m in s.mentions) or \
                any(s.pid in ps for g, ps in exp.items() if g != fid)
            if visual:
                sig["visual"] = 1
            if other_fig:
                sig["other_figure"] = 1
            hits.append(ImplicitHit(s, 0.0, sig))
        # pre-rank by cheap score, re-rank top-k with the cross-encoder
        for h in hits:
            h.score = _combine(h.signals, None)
        hits.sort(key=lambda h: -h.score)
        hits = hits[:max(top_k, 1)]
        if rerank and hits:
            try:
                from .dense import cross_cached
                rs = cross_cached(rerank, [(captions[j][:1000], h.sentence.text) for h in hits])
                for h, r in zip(hits, rs):
                    h.signals["rerank"] = round(float(r), 3)
            except Exception as e:
                print(f"[figrefs] reranker unavailable ({e})")
                rerank = None
        for h in hits:
            h.score = round(_combine(h.signals, rerank), 3)
        hits = [h for h in hits if h.score >= keep]
        hits.sort(key=lambda h: -h.score)
        out[fid] = hits
    return out


def _combine(sig: dict, rerank) -> float:
    """Hand-set combination (v2). To be replaced by a logistic regression fitted on the labelled set."""
    struct = max(sig.get("continuation", 0), sig.get("subfig", 0))
    sem = sig["attr"] if not rerank or "rerank" not in sig else 0.5 * sig["attr"] + 0.5 * sig["rerank"]
    score = struct + (1 - struct) * sem * (0.35 + 0.65 * sig["proximity"]) * (1.15 if sig.get("visual") else 1.0)
    if sig.get("other_figure"):
        score *= 0.35
    return float(min(score, 1.0))


def _caption_query(doc: Doc, fid: str) -> str:
    f = next(f for f in doc.figures if f.id == fid)
    cap = re.sub(r"^(Fig\.?|Figure|FIG\.?)\s*\S+\s*[.:|]?", "", f.caption)
    return " ".join([cap] + list(f.subcaptions.values()))
