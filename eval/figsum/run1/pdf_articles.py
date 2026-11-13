"""Article model for FigSum+ papers that exist only as PDFs (3 browser-printed web pages + 1 3-column AEM note),
where GROBID loses citations and sections. Plain PyMuPDF text, rebuilt into paragraphs:
  - text lines in reading order (column, then y); a paragraph ends at a vertical gap or a heading;
  - paragraphs are re-joined across columns/pages when the first one does not end a sentence;
  - captions = paragraphs that start "Fig. N." / "Figure N:" (number followed by . or :), removed from the body;
  - headings = short lines matching common section names; top section = last major heading (as load_article_tei);
  - citations = figrefs.explicit.find_mentions (lists, ranges, sub-figure letters), main figures only;
  - text before the abstract and from References / Acknowledgements on is dropped, plus web-page clutter.
GROBID is kept for the two papers it parses well (19673075, 23041342). usage: python pdf_articles.py  (prints stats)
"""
from __future__ import annotations

import os
import re
import sys

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from data import Article  # noqa: E402
from figrefs.explicit import find_mentions  # noqa: E402
from figrefs.textutil import normalize, split_sentences  # noqa: E402

ROOT = "/home/charan/bench/figsum"
CUSTOM = {   # id -> (paragraph mode, first kept line regex, stop regex)
    "17483019": ("lines", r"^Abstract$", r"^(References|Acknowledgments?|Acknowledgements?)$"),
    "22473769": ("lines", r"^Background\b", r"^(References|Acknowledg|Back to top|Article Information)$"),
    "23041342": ("lines", r"^Abstract$", r"^(References|Acknowledgments?)$"),
    "21183645": ("blocks", r"^The biosynthetic gene cluster of FK228", r"^(REFERENCES|Acknowledg|We thank)"),
}
JUNK = re.compile(r"(?i)^(view pdf|download|show more|get rights|add to mendeley|full text access|research article|"
                  r"original (article|contribution)|open in (figure )?viewer|powerpoint|pubmed|google scholar|crossref|"
                  r"article information|author affiliations|metrics|home\b|share|cite|permissions|"
                  r"downloaded from|vol\. \d+, \d{4}|\d{1,4}$|jama neurol|urologic oncology|https?://|"
                  r"©|copyright|figure in viewer|high-res|table \d|tabled? \d|article$|recommended articles|cited by|outline$|"
                  r"keywords$|next article|previous article|\d+/\d+$|\W*related articles|sections$|figures$|"
                  r"\* corresponding author|† supplemental material|published ahead of print|vol\. \d+, no\. \d+|"
                  r"you have access|view all author information|download ppt|doi\.org)")
PARA_JUNK = re.compile(r"(?i)phone: \(|e-mail:|published ahead of print|^\.asm\./org|^sciences, university of|"
                       r"^figures$|^[^.]{20,140}\.\.\.$")    # print header = truncated page title ending "..."
TABLE_FN = re.compile(r" a Apr, ampicillin resistance;.*?associated large operon\.", re.S)   # AEM Table 1 footnotes
TITLE_HDR = re.compile(r"^[^.]{20,140} \.\.\. (?=[A-Z])")   # same header glued to the first line of a page
# JAMA prints no Introduction heading, and "Discussion" is not in the print's text layer (rendered as a graphic).
MANUAL_HEAD = {"22473769": [("Introduction", "Alzheimer disease (AD) is a progressive"),
                            ("Discussion", "The pharmacokinetic characteristics of bapineuzumab")],
               "23041342": [("Introduction", "Intravesical administration of BCG is the standard")]}
CAPTION = re.compile(r"^(?:FIG|Fig|Figure|FIGURE)\.?\s*(\d+)\s*[.:|]\s*(.*)$", re.S)
HEAD = re.compile(r"(?i)^(\d+(\.\d+)*\.?\s+)?(abstract|purpose|background|objectives?|design|setting|patients|"
                  r"interventions|main outcome measures|introduction|materials and methods|methods?|results?|"
                  r"discussion|conclusions?|results and discussion|statistical analysis)$")
MAJOR = re.compile(r"(?i)^(\d+\.?\s*)?(abstract|introduction|background|results?|discussion|conclusions?|"
                   r"methods?|materials and methods|results and discussion)\b")


def _lines(doc, mode):
    """Units in reading order: (page, col, y0, y1, x0, text, block_id)."""
    out = []
    for pn, page in enumerate(doc):
        w, h = page.rect.width, page.rect.height
        tables = [pymupdf.Rect(t.bbox) for t in page.find_tables().tables] if mode == "blocks" else []
        for bi, b in enumerate(page.get_text("dict")["blocks"]):
            if b.get("type") != 0:
                continue
            for li, ln in enumerate(b["lines"]):
                t = re.sub(r"[\ue000-\uf8ff]", "", "".join(s["text"] for s in ln["spans"])).strip()  # icon glyphs
                if not t:
                    continue
                x0, y0, x1, y1 = ln["bbox"]
                if mode == "lines" and (x0 > w * 0.62 or y0 < 30 or y0 > h - 55 or t in ("Reading", "Assistant")):
                    continue    # web print: side panel, browser header/footer, floating "Reading Assistant" button
                if mode == "blocks" and (y0 < 60 or any(r.contains(pymupdf.Point(x0 + 1, y0 + 1)) for r in tables)):
                    continue    # journal running head; table cells
                col = 0 if (x0 < w * 0.45 or mode == "lines") else 1
                out.append((pn, col, y0, y1, x0, t, (pn, bi)))
    out.sort(key=lambda u: (u[0], u[1], u[2]))
    return out


def _join(a, b):
    if re.search(r"[A-Za-z][-‐‑]$", a) and b[:1].islower():
        return a[:-1] + b
    return a + " " + b


def paragraphs(pid):
    mode, start, stop = CUSTOM[pid]
    doc = pymupdf.open(os.path.join(ROOT, pid, "paper.pdf"))
    units = _lines(doc, mode)
    gaps = [round(units[i][2] - units[i - 1][3]) for i in range(1, len(units))
            if units[i][0] == units[i - 1][0] and units[i][2] > units[i - 1][3] - 1]
    base = max(set(gaps), key=gaps.count) if gaps else 3      # modal line-to-line gap of this paper
    paras, cur, prev, on = [], None, None, False
    for u in units:
        pn, col, y0, y1, x0, t, blk = u
        if not on:
            if re.search(start, t):
                on = True
            else:
                continue
        if re.search(stop, t):
            break
        if JUNK.search(t) and not CAPTION.match(t):
            continue
        head = bool(HEAD.match(t.rstrip(".:"))) and len(t) < 60
        new = (cur is None or head or (prev is not None and prev[6][0] == pn and prev[1] == col and
               (y0 - prev[3] > base + 4 if mode == "lines" else blk != prev[6])))
        if prev is not None and (prev[0] != pn or prev[1] != col):   # column/page change: continue if mid-sentence
            new = head or (cur is not None and (bool(re.search(r"[.?!:]$", cur["text"])) or
                                                bool(CAPTION.match(cur["text"]))))   # captions never continue
        if CAPTION.match(t) and (cur is None or new or len(cur["text"]) > 0):
            new = True
        if head:
            paras.append({"text": t.rstrip(".:"), "head": True})
            cur, prev = None, u
            continue
        if new or cur is None:
            cur = {"text": t, "head": False}
            paras.append(cur)
        else:
            cur["text"] = _join(cur["text"], t)
        prev = u
    return paras


def load_pdf_article(pid: str) -> Article:
    figures, body, sec, top, cur_sec, cur_top = {}, [], [], [], "", ""
    in_abs = False      # abstract sub-headings (Purpose / Results / Conclusions) stay under "Abstract"
    for p in paragraphs(pid):
        t = TABLE_FN.sub("", TITLE_HDR.sub("", normalize(p["text"])))
        if PARA_JUNK.search(t):
            continue
        for name, first in MANUAL_HEAD.get(pid, []):
            if t.startswith(first):
                cur_sec = cur_top = name
                in_abs = False
        if pid == "21183645" and body:      # AEM note: only the first paragraph is the abstract; no headings
            cur_sec = cur_top = "Main text"
        if (not p["head"] and body and t[:1].islower() and not re.search(r"[.?!]$", body[-1])
                and not CAPTION.match(t)):  # continuation split off by a caption / column / page break
            body[-1] = _join(body[-1], t)
            continue
        if p["head"]:
            cur_sec = t
            if re.match(r"(?i)abstract$", t):
                in_abs, cur_top = True, "Abstract"
            elif in_abs and not re.match(r"(?i)^(\d+\.?\s+|introduction|background)", t):
                pass
            elif MAJOR.match(t) or not cur_top:
                in_abs, cur_top = False, t
            continue
        m = CAPTION.match(t)
        if m and len(t) > 25:
            n = int(m.group(1))
            if n not in figures:
                figures[n] = normalize(m.group(2))[:1500]
            continue
        body.append(t)
        sec.append(cur_sec or "Abstract")
        top.append(cur_top or "Abstract")
    sents, sp, cue = [], [], []
    for i, ptext in enumerate(body):
        for s in split_sentences(ptext):
            figs = {int(m.figure) for m in find_mentions(s) if m.figure.isdigit()}
            sents.append(s)
            sp.append(i)
            cue.append(figs & set(figures))
    a = Article(pid, "", sents, sp, body, sec, figures, cue)
    a.top_section = top
    a.fig_pos = {}
    return a


if __name__ == "__main__":
    for pid in CUSTOM:
        a = load_pdf_article(pid)
        print(pid, "paras", len(a.paragraphs), "sents", len(a.sentences), "figs", sorted(a.figures),
              "cited", sorted(set().union(*a.cue)), "sections", sorted(set(a.top_section)))
