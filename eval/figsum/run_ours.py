"""Run figrefs on the FigSum+ articles and score it next to the FigSum+ systems.

Input modes
  xml  the article's JATS paragraphs and captions are fed straight into figrefs' document model (tests the
       explicit + implicit stages alone, same sentence universe as the FigSum+ systems)
  pdf  the full tool on the PDF; its sentences are mapped to article sentences by token F1 >= 0.5
       (unmapped selections count as wrong)
Output variants (all from one run)
  figrefs (paragraphs)  explicit paragraphs, whole + implicit sentences scoring >= threshold   [the tool's output]
  figrefs (sentences)   explicit mention sentences + implicit sentences scoring >= threshold
  figrefs (top-n)       mention sentences, then implicit candidates by score, then the rest of the explicit
                        paragraphs, cut at n (the FigSum+ protocol)
usage: python eval/figsum/run_ours.py GOLD_DIR ARTICLE_ROOT [--modes xml,pdf] [--configs full,tfidf] [--out F]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data import load_all, tf1, toks  # noqa: E402
from evaluate import PUBLISHED, evaluate  # noqa: E402
from systems import SYSTEMS, Ctx  # noqa: E402

from figrefs.doc import build_doc  # noqa: E402
from figrefs.implicit import score_implicit  # noqa: E402
from figrefs.pdfparse import Figure, Paragraph, ParsedDoc, parse_pdf  # noqa: E402

CONFIGS = {"full": dict(dense="bge", rerank="bge-reranker"), "tfidf": dict(dense=None, rerank=None),
           "bge": dict(dense="bge", rerank=None)}


def parsed_from_article(a) -> ParsedDoc:
    paras = [Paragraph(i, t, a.para_section[i], [0]) for i, t in enumerate(a.paragraphs)]
    figs = [Figure(str(n), "figure", 0, f"Figure {n}. {c}", (0, 0, 0, 0)) for n, c in sorted(a.figures.items())]
    return ParsedDoc(a.id, a.title, paras, figs, 10.0, 1)


def map_sentences(doc, a, same_para: bool) -> dict[int, int]:
    """doc sentence gid -> article sentence index (or a unique negative id when unmapped)."""
    T = [toks(s) for s in a.sentences]
    exact = {}
    for i, s in enumerate(a.sentences):
        exact.setdefault(s, i)
    out = {}
    for s in doc.sentences:
        if s.text in exact:
            out[s.gid] = exact[s.text]
            continue
        cands = [i for i, p in enumerate(a.sent_para) if p == s.pid] if same_para else range(len(T))
        q = toks(s.text)
        best, bi = 0.0, None
        for i in cands:
            v = tf1(q, T[i])
            if v > best:
                best, bi = v, i
        out[s.gid] = bi if best >= 0.5 else -1 - s.gid
    return out


def run_article(a, mode: str, cfg: dict, threshold: float = 0.5):
    parsed = parsed_from_article(a) if mode == "xml" else parse_pdf(os.path.join(ART_ROOT, a.id, "paper.pdf"))
    doc = build_doc(parsed)
    imp = score_implicit(doc, threshold=threshold, **cfg)
    m = map_sentences(doc, a, same_para=(mode == "xml"))
    out = {}
    for f in doc.figures:
        if not f.id.isdigit():
            continue
        fid = int(f.id)
        ment = [s for s in doc.sentences if any(x.figure == f.id for x in s.mentions)]
        pids = {s.pid for s in ment}
        para = [s for s in doc.sentences if s.pid in pids]
        hits = imp.get(f.id, [])
        sel_imp = [h.sentence for h in hits if h.score >= threshold]
        ranked = [s.gid for s in ment] + [h.sentence.gid for h in sorted(hits, key=lambda h: -h.score)] + \
                 [s.gid for s in para]
        seen, order = set(), []
        for g in ranked:
            if m[g] not in seen:
                seen.add(m[g])
                order.append(m[g])
        out[fid] = {"paragraphs": {m[s.gid] for s in para} | {m[s.gid] for s in sel_imp},
                    "sentences": {m[s.gid] for s in ment} | {m[s.gid] for s in sel_imp},
                    "ranked": order,
                    "scores": {m[h.sentence.gid]: h.score for h in hits}}
    return out


def main():
    global ART_ROOT
    gold_dir, ART_ROOT = sys.argv[1], sys.argv[2]
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d  # noqa: E731
    modes = arg("--modes", "xml,pdf").split(",")
    configs = arg("--configs", "full,tfidf").split(",")
    out_path = arg("--out", "eval/figsum/results_ours.json")
    arts, _ = load_all(gold_dir, ART_ROOT)
    ctx = {a.id: Ctx(a) for a in arts}
    cue_free = lambda a, f, S: {i for i in S if i >= 0 and f not in a.cue[i]} | {i for i in S if i < 0}  # noqa: E731
    res = {}
    for mode in modes:
        A = [a for a in arts if mode == "xml" or os.path.exists(os.path.join(ART_ROOT, a.id, "paper.pdf"))]
        ids = {a.id for a in A}
        runs = {c: {a.id: run_article(a, mode, CONFIGS[c]) for a in A} for c in configs}
        for gs in ("GS1", "GS2"):
            print(f"\n== {gs} | input={mode} | {len(A)} articles")
            print(f"{'system':34s} {'all gold: P    R    F1':24s} {'micro F1':9s}  {'implicit only: P  R  F1':26s}")
            table = {}
            sel = {name: (lambda a, f, n, fn=fn: fn(ctx[a.id], f, n)) for name, fn in SYSTEMS.items()
                   if name.startswith("FigSum+")}
            for c in configs:
                R = runs[c]
                sel[f"figrefs[{c}] (paragraphs)"] = lambda a, f, n, R=R: R[a.id].get(f, {}).get("paragraphs", set())
                sel[f"figrefs[{c}] (sentences)"] = lambda a, f, n, R=R: R[a.id].get(f, {}).get("sentences", set())
                sel[f"figrefs[{c}] (top-n)"] = lambda a, f, n, R=R: R[a.id].get(f, {}).get("ranked", [])[:n]
            for name, fn in sel.items():
                r_all = evaluate(A, gs, fn)
                r_imp = evaluate(A, gs, fn, subset=cue_free)
                table[name] = {"all": r_all, "implicit": r_imp}
                mm, mi = r_all["mean"], r_imp["mean"]
                pub = PUBLISHED[gs].get(name)
                pubs = f"   [paper F1 {pub[2]:.2f}]" if pub and mode == "xml" and len(ids) >= 13 else ""
                print(f"{name:34s} {mm[0]:.2f} {mm[1]:.2f} {mm[2]:.2f}            {r_all['micro'][2]:.2f}      "
                      f"{mi[0]:.2f} {mi[1]:.2f} {mi[2]:.2f}{pubs}")
            res[f"{gs}/{mode}"] = {"articles": sorted(ids), "systems": table}
    # threshold sweep (diagnostic only — not used to pick the reported threshold)
    json.dump(res, open(out_path, "w"), indent=1)
    raw = {c: {aid: {f: {"scores": {str(k): v for k, v in d["scores"].items()}} for f, d in r.items()}
               for aid, r in runs[c].items()} for c in configs}
    json.dump(raw, open(out_path.replace(".json", "_scores.json"), "w"))


if __name__ == "__main__":
    main()
