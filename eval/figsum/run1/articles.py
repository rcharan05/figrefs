"""All 18 FigSum+ GS1 articles as one article model, the fixed run-1 split, and the export for Kaggle.

Sources:
  JATS XML (13 papers, PMC open access)    data.load_article; explicit cues from <xref ref-type="fig"> + literal regex
  GROBID TEI (19673075, Toxicon PDF)       data.load_article_tei; top sections rebuilt from ALL TEI heads (the
                                           loader only saw heads of divs that hold paragraphs)
  PyMuPDF (17483019, 21183645, 22473769,   pdf_articles.load_pdf_article (browser-printed pages / 3-column note)
           23041342)
PDF-derived articles drop junk "sentences" (axis ticks, table cells: fewer than 3 words of >= 3 letters).
Gold GS1 is aligned by token F1 >= 0.6 (data.align). 21183645: 16 of its 39 gold rows describe the companion
paper 21183648 (E. coli strains SW01-SW08), so they do not align and are dropped.
Split (fixed 10 Nov, before any run-1 training): test = the 3 papers never seen by anyone in this project
with the most figures; everything else is the LOAO training pool.
usage: python articles.py OUT_DIR      -> OUT_DIR/articles.jsonl (+ prints stats)
"""
from __future__ import annotations

import json
import os
import re
import sys

from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from data import GOLD_FILES, align, load_article, load_article_tei, read_gold  # noqa: E402
from pdf_articles import CUSTOM, load_pdf_article  # noqa: E402

ROOT = "/home/charan/bench/figsum"
GS = "GS1"
TEST = ["17483019", "19673075", "23041342"]
TEI_ONLY = ["19673075"]
OFFTARGET = re.compile(r"data not shown|not shown|supplement|additional file|figure s\d|fig\.? s\d|\btable\b|web site", re.I)
METHODS = re.compile(r"method|material|procedure", re.I)
MAJOR = re.compile(r"(?i)^(abstract|introduction|background|results?|discussion|conclusions?|methods?|"
                   r"materials and methods|results and discussion)\b")


def _tei_top_sections(tei_path):
    """Top section of every <p>, in load_article_tei's paragraph order, from all heads incl. empty parent divs."""
    T = "{http://www.tei-c.org/ns/1.0}"
    root = etree.parse(tei_path).getroot()
    out = []
    for cont, is_abs in [(c, True) for c in root.iter(T + "abstract")] + [(c, False) for c in root.iter(T + "body")]:
        cur = "Abstract" if is_abs else ""
        for div in cont.iter(T + "div"):
            h = div.find(T + "head")
            head = " ".join("".join(h.itertext()).split()) if h is not None else ""
            n = (h.get("n") or "") if h is not None else ""
            if not is_abs and (re.fullmatch(r"\d+\.?", n) or (not n and MAJOR.match(head))):
                cur = head
            for _ in div.findall(T + "p"):
                out.append("Abstract" if is_abs else (cur or head))
    return out


def _junk(s):
    return len(re.findall(r"[A-Za-z]{3,}", s)) < 3


def _drop_sentences(a, keep):
    idx = [i for i in range(len(a.sentences)) if keep(a.sentences[i])]
    a.sentences = [a.sentences[i] for i in idx]
    a.sent_para = [a.sent_para[i] for i in idx]
    a.cue = [a.cue[i] for i in idx]
    return a


def load(pid):
    d = os.path.join(ROOT, pid)
    if pid in CUSTOM:
        a, src = _drop_sentences(load_pdf_article(pid), lambda s: not _junk(s)), "pdf-pymupdf"
    elif pid in TEI_ONLY:
        a = load_article_tei(pid, os.path.join(d, "grobid_091.tei.xml"), os.path.join(d, "paper.pdf"))
        top = _tei_top_sections(os.path.join(d, "grobid_091.tei.xml"))
        assert len(top) == len(a.paragraphs), (len(top), len(a.paragraphs))
        a.top_section = top
        a, src = _drop_sentences(a, lambda s: not _junk(s)), "pdf-grobid"
    else:
        a, src = load_article(pid, os.path.join(d, "paper.xml")), "jats"
    a.source = src
    return a


def all_articles():
    rows = read_gold(os.path.join(HERE, "..", "gold", GOLD_FILES[GS]))
    arts = {}
    for pid in sorted({r[0] for r in rows}):
        a = load(pid)
        align(a, GS, rows)
        a.n_gold_rows = sum(1 for r in rows if r[0] == pid)
        arts[pid] = a
    return arts


def figs_of(a):
    """Scored figures: gold figures that exist in the article with a caption (as experiments.Data.figs)."""
    return sorted(f for f, g in a.gold[GS].items() if f in a.figures and g)


def b0f(a, f):
    """Rule baseline (phase 3): citing sentences of f + adjacent (d = 1) sentences in a paragraph that cites f,
    minus sentences that cite another figure, are off-target, or lie in a Methods section."""
    cit = [i for i, c in enumerate(a.cue) if f in c]
    paras = {a.sent_para[i] for i in cit}
    sel = set(cit)
    for i in range(len(a.sentences)):
        if i in sel or not cit:
            continue
        d = min(abs(i - m) for m in cit)
        if (d == 1 and a.sent_para[i] in paras and not a.cue[i] and not OFFTARGET.search(a.sentences[i])
                and not METHODS.search(a.top_section[a.sent_para[i]] or "")):
            sel.add(i)
    return sel


def export(arts, path):
    with open(path, "w") as fh:
        for pid, a in sorted(arts.items()):
            fh.write(json.dumps({
                "id": pid, "split": "test" if pid in TEST else "train", "source": a.source,
                "sentences": a.sentences, "sent_para": a.sent_para,
                "para_section": [a.top_section[p] for p in range(len(a.paragraphs))],
                "captions": {str(k): v for k, v in a.figures.items()},
                "cue": [sorted(c) for c in a.cue],
                "figs": figs_of(a),
                "gold": {str(f): sorted(a.gold[GS][f]) for f in figs_of(a)},
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    A = all_articles()
    for pid, a in sorted(A.items()):
        F = figs_of(a)
        print(f"{pid:9s} {'TEST ' if pid in TEST else 'train'} {a.source:11s} sents {len(a.sentences):4d} figs {len(F)} "
              f"gold rows {a.n_gold_rows:3d} aligned {sum(len(a.gold[GS][f]) for f in F):3d} unaligned {a.unaligned[GS]:2d} "
              f"uncited figs {[f for f in F if not any(f in c for c in a.cue)]}")
    export(A, os.path.join(out, "articles.jsonl"))
    print("wrote", os.path.join(out, "articles.jsonl"))
