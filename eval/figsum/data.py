"""FigSum+ gold standards (Ramesh, Sethi & Yu, PLoS ONE 2015; figshare 858903) aligned to PMC JATS full text.

Article model (from the JATS XML):
  sentences  abstract + body paragraphs (figure/table captions excluded), split with our sentence splitter
  figures    {number: caption}
  cue        sentence -> set of figures it explicitly names (<xref ref-type="fig"> or a literal "Fig. N")
Gold sentences are aligned to article sentences by token F1 (best match, >= 0.6); the alignment rate is reported.
"""
from __future__ import annotations

import csv
import io
import os
import re
import sys
from dataclasses import dataclass, field

from lxml import etree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from figrefs.textutil import normalize, split_sentences  # noqa: E402

GOLD_FILES = {"GS1": "FigureAssociatedTextCombinedGoldStandard.csv", "GS2": "FigureSumCombinedGoldStandard.csv"}
TOK = re.compile(r"[a-z0-9]+")
LITERAL = re.compile(r"\b(?:Fig(?:ure)?s?\s?\.?)\s*(\d+)", re.I)


def toks(s):
    return frozenset(TOK.findall(s.lower()))


def tf1(a, b):
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


@dataclass
class Article:
    id: str
    title: str
    sentences: list[str]
    sent_para: list[int]          # paragraph index of each sentence
    paragraphs: list[str]
    para_section: list[str]
    figures: dict[int, str]       # number -> caption
    cue: list[set[int]]           # figures explicitly named in each sentence
    gold: dict[str, dict[int, set[int]]] = field(default_factory=dict)   # gs -> fig -> sentence indices
    unaligned: dict[str, int] = field(default_factory=dict)


def _local(t):
    return t.split("}")[-1] if isinstance(t, str) else ""


def load_article(pid: str, xml_path: str) -> Article:
    root = etree.parse(xml_path).getroot()
    rid2num, figures = {}, {}
    for f in root.iter("{*}fig"):
        lab = f.find("{*}label")
        m = re.match(r"\s*(?:Fig(?:ure)?\.?)\s*(\d+)", "".join(lab.itertext()) if lab is not None else "", re.I)
        if not m:
            continue
        n = int(m.group(1))
        cap = f.find("{*}caption")
        figures[n] = normalize("".join(cap.itertext())) if cap is not None else ""
        if f.get("id"):
            rid2num[f.get("id")] = n

    def ser(el):
        parts = [el.text or ""]
        for ch in el:
            nm = _local(ch.tag)
            if nm in ("fig", "table-wrap", "disp-formula", "fn"):
                pass
            elif nm == "xref" and ch.get("ref-type") == "fig":
                nums = [rid2num[r] for r in (ch.get("rid") or "").split() if r in rid2num]
                parts.append("".join(ch.itertext()) + "".join(f"ZZFR{n}ZZ" for n in nums))
            else:
                parts.append(ser(ch))
            parts.append(ch.tail or "")
        return "".join(parts)

    title_el = root.find(".//{*}article-meta//{*}article-title")
    title = normalize("".join(title_el.itertext())) if title_el is not None else ""
    paragraphs, sections, topsec = [], [], []
    fig_pos: dict[int, int] = {}   # figure -> index of the paragraph the <fig> element follows
    for cont in list(root.iter("{*}abstract")) + list(root.iter("{*}body")):
        for p in cont.iter("{*}p", "{*}fig"):
            if _local(p.tag) == "fig":
                lab = p.find("{*}label")
                m = re.match(r"\s*(?:Fig(?:ure)?\.?)\s*(\d+)", "".join(lab.itertext()) if lab is not None else "", re.I)
                if m and int(m.group(1)) not in fig_pos:
                    fig_pos[int(m.group(1))] = max(0, len(paragraphs) - 1)
                continue
            if any(_local(a.tag) in ("fig", "table-wrap", "caption", "fn", "boxed-text") for a in p.iterancestors()):
                continue
            if any(_local(d.tag) == "p" for d in p.iterdescendants()):
                continue
            sec = p.getparent().findtext("{*}title") if p.getparent() is not None else ""
            top = ""
            for a in p.iterancestors():
                if _local(a.tag) == "sec" and _local(a.getparent().tag) == "body":
                    top = a.findtext("{*}title") or ""
            paragraphs.append(normalize(ser(p)))
            is_abs = _local(cont.tag) == "abstract"
            sections.append("Abstract" if is_abs else normalize(sec or ""))
            topsec.append("Abstract" if is_abs else normalize(top or sec or ""))
    sents, sp, cue = [], [], []
    for i, ptext in enumerate(paragraphs):
        for s in split_sentences(ptext):
            figs = {int(n) for n in re.findall(r"ZZFR(\d+)ZZ", s)}
            clean = normalize(re.sub(r"ZZFR\d+ZZ", "", s))
            figs |= {int(n) for n in LITERAL.findall(clean) if int(n) in figures}
            sents.append(clean)
            sp.append(i)
            cue.append(figs & set(figures))
    paragraphs = [normalize(re.sub(r"ZZFR\d+ZZ", "", p)) for p in paragraphs]
    a = Article(pid, title, sents, sp, paragraphs, sections, figures, cue)
    a.top_section = topsec
    a.fig_pos = fig_pos
    return a


def load_article_tei(pid: str, tei_path: str, pdf_path: str | None = None) -> Article:
    """Same article model built from GROBID TEI (PDF route): abstract + body paragraphs, figure captions from
    <figure>/<figDesc>, explicit cues from <ref type="figure"> plus literal "Fig. N", sections from <head>."""
    T = "{http://www.tei-c.org/ns/1.0}"
    root = etree.parse(tei_path).getroot()
    id2num, figures = {}, {}
    for f in root.iter(T + "figure"):
        if f.get("type") == "table":
            continue
        lab, head = f.find(T + "label"), f.find(T + "head")
        txt = " ".join(x for x in [lab.text if lab is not None else "", "".join(head.itertext()) if head is not None else ""] if x)
        if re.search(r"(?i)\b(S\d|supp)", txt):
            continue
        m = re.search(r"(\d+)", txt)
        if not m:
            continue
        n = int(m.group(1))
        desc = f.find(T + "figDesc")
        if n not in figures:
            figures[n] = normalize("".join(desc.itertext())) if desc is not None else ""
        fid = f.get("{http://www.w3.org/XML/1998/namespace}id")
        if fid:
            id2num[fid] = n

    def ser(el):
        parts = [el.text or ""]
        for ch in el:
            if ch.tag == T + "ref" and ch.get("type") == "figure":
                tgt = (ch.get("target") or "").lstrip("#")
                txt = "".join(ch.itertext())
                if tgt in id2num:
                    nums = [id2num[tgt]]
                elif not tgt and not re.search(r"(?i)supp", txt):  # unresolved: number from the ref text
                    nums = [int(n) for n in re.findall(r"(?<![\d.SsA-Za-z])(\d{1,2})(?![\d.]\d)", txt)]
                else:
                    nums = []
                parts.append(txt + "".join(f"ZZFR{n}ZZ" for n in nums))
            else:
                parts.append(ser(ch))
            parts.append(ch.tail or "")
        return "".join(parts)

    title_el = root.find(f".//{T}titleStmt/{T}title")
    title = normalize("".join(title_el.itertext())) if title_el is not None else ""
    paragraphs, sections, topsec = [], [], []
    for cont, is_abs in [(c, True) for c in root.iter(T + "abstract")] + [(c, False) for c in root.iter(T + "body")]:
        for div in cont.iter(T + "div"):
            h = div.find(T + "head")
            head = normalize("".join(h.itertext())) if h is not None else ""
            for p in div.findall(T + "p"):
                paragraphs.append(normalize(ser(p)))
                sections.append("Abstract" if is_abs else head)
                topsec.append("Abstract" if is_abs else head)
    # GROBID does not nest sections: carry the last major heading forward as the top-level section
    major = re.compile(r"(?i)^(\d+\.?\s*)?(abstract|introduction|background|results?|discussion|conclusions?|"
                       r"methods?|materials and methods|results and discussion)\b")
    cur = ""
    for k, s in enumerate(topsec):
        if s == "Abstract" or major.match(s or ""):
            cur = s
        topsec[k] = cur or s
    # figures referenced in the text but not detected by GROBID: recover the caption from the raw PDF text
    mentioned = {int(n) for p in paragraphs for n in re.findall(r"ZZFR(\d+)ZZ", p)}
    mentioned |= {int(n) for p in paragraphs for n in LITERAL.findall(re.sub(r"ZZFR\d+ZZ", "", p)) if int(n) <= 30}
    missing = sorted((mentioned - set(figures)) | {n for n, c in figures.items() if len(c) < 40 and n in mentioned})
    recovered = []
    if missing and pdf_path and os.path.exists(pdf_path):
        import pymupdf
        raw = "\n".join(pg.get_text() for pg in pymupdf.open(pdf_path))
        for n in missing:
            m = re.search(r"(?m)^[ \t]*(?:Figure|Fig\.?)[ \t]*" + str(n) + r"[ \t]*(?:[.:|][ \t]*)?\n?([^\n][\s\S]{19,700})", raw)
            if m and not (n in figures and len(figures[n]) >= 40):
                cap = re.split(r"\n\s*\n|\n[ \t]*(?:Figure|Fig\.?|Table)[ \t]*\d", m.group(1))[0]
                figures[n] = normalize(cap.replace("-\n", "").replace("\n", " "))
                recovered.append(n)
    sents, sp, cue = [], [], []
    for i, ptext in enumerate(paragraphs):
        for s in split_sentences(ptext):
            figs = {int(n) for n in re.findall(r"ZZFR(\d+)ZZ", s)}
            clean = normalize(re.sub(r"ZZFR\d+ZZ", "", s))
            figs |= {int(n) for n in LITERAL.findall(clean) if int(n) in figures}
            sents.append(clean)
            sp.append(i)
            cue.append(figs & set(figures))
    paragraphs = [normalize(re.sub(r"ZZFR\d+ZZ", "", p)) for p in paragraphs]
    a = Article(pid, title, sents, sp, paragraphs, sections, figures, cue)
    a.top_section = topsec
    a.fig_pos = {}
    a.recovered_captions = recovered
    return a


def load_all_tei(gold_dir: str, art_root: str, tag: str = "091") -> tuple[list[Article], dict]:
    gold = {gs: read_gold(os.path.join(gold_dir, fn)) for gs, fn in GOLD_FILES.items()}
    arts = []
    for pid in sorted({r[0] for rows in gold.values() for r in rows}):
        tei = os.path.join(art_root, pid, f"grobid_{tag}.tei.xml")
        if not os.path.exists(tei):
            continue
        a = load_article_tei(pid, tei, os.path.join(art_root, pid, "paper.pdf"))
        for gs, rows in gold.items():
            align(a, gs, rows)
        arts.append(a)
    return arts, {}


def read_gold(path: str) -> list[tuple[str, int, int, str]]:
    raw = open(path, "rb").read().decode("utf-8", "replace").replace("\r", "\n")
    rows = [r for r in list(csv.reader(io.StringIO(raw)))[1:] if len(r) >= 4 and r[0].strip()]
    return [(r[0].strip(), int(r[1]), int(r[2]), r[3]) for r in rows]


def align(art: Article, gs: str, rows: list[tuple[str, int, int, str]], min_f1: float = 0.6):
    T = [toks(s) for s in art.sentences]
    out: dict[int, set[int]] = {}
    bad = 0
    for pid, fig, _num, text in rows:
        if pid != art.id:
            continue
        g = toks(text)
        best, bi = 0.0, None
        for i, t in enumerate(T):
            s = tf1(g, t)
            if s > best:
                best, bi = s, i
        if best >= min_f1:
            out.setdefault(fig, set()).add(bi)
        else:
            bad += 1
    art.gold[gs] = out
    art.unaligned[gs] = bad


def load_all(gold_dir: str, art_root: str) -> tuple[list[Article], dict]:
    gold = {gs: read_gold(os.path.join(gold_dir, fn)) for gs, fn in GOLD_FILES.items()}
    arts, stats = [], {"missing_fulltext": [], "rows": {gs: len(r) for gs, r in gold.items()}}
    ids = sorted({r[0] for rows in gold.values() for r in rows})
    for pid in ids:
        xml = os.path.join(art_root, pid, "paper.xml")
        if not os.path.exists(xml):
            stats["missing_fulltext"].append(pid)
            continue
        a = load_article(pid, xml)
        for gs, rows in gold.items():
            align(a, gs, rows)
        arts.append(a)
    return arts, stats


if __name__ == "__main__":
    arts, st = load_all(sys.argv[1], sys.argv[2])
    print(st)
    for a in arts:
        for gs in ("GS1", "GS2"):
            n = sum(len(v) for v in a.gold.get(gs, {}).values())
            print(a.id, gs, "figs", len(a.gold.get(gs, {})), "aligned", n, "unaligned", a.unaligned.get(gs),
                  "| article sents", len(a.sentences), "figs in xml", sorted(a.figures))
