"""Layout-aware PDF parsing with PyMuPDF.

Produces: body paragraphs in reading order (with section titles), figure captions,
sub-captions, and a crop rectangle per figure.

Pipeline
  1. collect every text line with font statistics
  2. infer body font (size + family) from character-weighted mode
  3. classify lines: header/footer, caption, heading, body, other (figure text, tables, footnotes, math)
  4. locate figure regions above/below captions from images + vector drawings + small text
  5. order body lines (column-aware) and segment into paragraphs
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

from .textutil import normalize

# ------------------------------------------------------------------ data model


@dataclass
class Line:
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    size: float
    font: str
    fonts: list[str]
    block: int
    kind: str = "other"
    maxsize: float = 0.0  # body | heading | caption | header | math | other

    @property
    def x0(self): return self.bbox[0]
    @property
    def y0(self): return self.bbox[1]
    @property
    def x1(self): return self.bbox[2]
    @property
    def y1(self): return self.bbox[3]


@dataclass
class Paragraph:
    id: int
    text: str
    section: str
    pages: list[int]
    bboxes: list[tuple[int, tuple[float, float, float, float]]] = field(default_factory=list)


@dataclass
class Figure:
    id: str            # "1", "2", ... (label number as printed)
    kind: str          # "figure" | "table"
    page: int
    caption: str
    caption_bbox: tuple[float, float, float, float]
    subcaptions: dict[str, str] = field(default_factory=dict)
    region: tuple[float, float, float, float] | None = None
    image_path: str | None = None


@dataclass
class ParsedDoc:
    path: str
    title: str
    paragraphs: list[Paragraph]
    figures: list[Figure]
    body_size: float
    n_pages: int


# ------------------------------------------------------------------ regexes

CAPTION_RE = re.compile(
    r"^(?P<kw>Fig\.?|Figure|FIG\.?|FIGURE|Table|TABLE)\s*(?P<num>[A-Z]?\d+(?:\.\d+)?)\s*(?P<punct>[.:|—–-])?\s*(?P<rest>.*)$"
)
SUBCAP_RE = re.compile(r"^\(([a-z])\)\s*(.*)$")
HEADING_NUM_RE = re.compile(r"^((\d+|[A-Z]|[IVX]+)(\.\d+)*\.?)\s+\S")
STOP_SECTIONS = re.compile(r"^(\d+\.?\s*)?(references|bibliography|acknowledge?ments?|biograph(y|ies))\b", re.I)
MATH_FONT_RE = re.compile(r"^(CM(MI|SY|EX)\d|LMMath|MSBM|MSAM|Symbol|STIX|rsfs|eufm|.*Math)", re.I)
BOLD_RE = re.compile(r"(Bold|Bd|Black|Heavy|Semibold|Medi)", re.I)


def _family(font: str) -> str:
    f = font.split("+")[-1]
    return re.split(r"[-,]", f)[0]


# ------------------------------------------------------------------ main entry


def parse_pdf(path: str) -> ParsedDoc:
    doc = pymupdf.open(path)
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_LIGATURES & ~pymupdf.TEXT_PRESERVE_IMAGES
    lines: list[Line] = []
    page_graphics: dict[int, list[tuple[float, float, float, float]]] = {}
    page_rects: dict[int, pymupdf.Rect] = {}
    page_hrules: dict[int, list[tuple]] = {}

    for page in doc:
        pno = page.number
        page_rects[pno] = page.rect
        d = page.get_text("dict", flags=flags)
        for bi, b in enumerate(d["blocks"]):
            if b.get("type") != 0:
                continue
            for ln in b["lines"]:
                spans = [s for s in ln["spans"] if s["text"].strip()]
                if not spans:
                    continue
                if abs(ln.get("dir", (1, 0))[1]) > 0.2:  # rotated text (axis labels, arXiv stamp)
                    lines.append(Line(pno, tuple(ln["bbox"]), normalize(" ".join(s["text"] for s in spans)), 1.0, "rot", [], bi, "other"))
                    continue
                text = normalize(" ".join(s["text"] for s in spans))
                chars = Counter()
                sizes = Counter()
                for s in spans:
                    n = len(s["text"].strip())
                    chars[s["font"]] += n
                    sizes[round(s["size"], 1)] += n
                lines.append(Line(pno, tuple(ln["bbox"]), text, sizes.most_common(1)[0][0],
                                  chars.most_common(1)[0][0], [s["font"] for s in spans], bi,
                                  maxsize=max(s["size"] for s in spans)))
        page_graphics[pno] = _graphics(page)
        page_hrules[pno] = [tuple(pymupdf.Rect(dr["rect"])) for dr in page.get_drawings()
                            if dr["rect"].height < 2 and dr["rect"].width > 60]

    lines = _merge_visual_lines(lines, page_rects)
    body_size, body_family = _body_font(lines)
    _classify(lines, body_size, body_family, page_rects)
    _mark_algorithms(lines, page_hrules, body_size)
    figures = _figures(doc, lines, page_graphics, page_rects, body_size)
    _mark_region_lines(lines, figures)
    paragraphs = _paragraphs(lines, page_rects, body_size)
    title = _title(lines)
    return ParsedDoc(path, title, paragraphs, figures, body_size, doc.page_count)


# ------------------------------------------------------------------ steps


def _graphics(page) -> list[tuple[float, float, float, float]]:
    """Bounding boxes of raster images and vector drawings on a page."""
    out = []
    pr = page.rect
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & pr
        if r.width > 8 and r.height > 8:
            out.append(tuple(r))
    for dr in page.get_drawings():
        r = pymupdf.Rect(dr["rect"]) & pr
        if r.is_empty and not (r.width > 3 or r.height > 3):
            continue
        # skip full-width hairlines (rules under headers, footnote separators)
        if r.height < 1.5 and r.width > 0.6 * pr.width:
            continue
        if r.width * r.height > 0.85 * pr.width * pr.height:  # page frames
            continue
        out.append(tuple(r))
    return out


def _merge_visual_lines(lines: list[Line], page_rects) -> list[Line]:
    """PyMuPDF often splits one visual line into several (heading number + title, math pieces, bullets)."""
    out: list[Line] = []
    by_page: dict[int, list[Line]] = {}
    for l in lines:
        by_page.setdefault(l.page, []).append(l)
    for pno, pl in by_page.items():
        mid = page_rects[pno].width / 2
        pl = sorted(pl, key=lambda l: (l.x0))
        groups: list[list[Line]] = []
        for l in sorted(pl, key=lambda l: (round(l.y1), l.x0)):
            if l.font == "rot":
                groups.append([l]); continue
            placed = False
            for g in groups:
                last = g[-1]
                if last.font == "rot":
                    continue
                h = min(l.y1 - l.y0, last.y1 - last.y0)
                vo = _overlap(l.y0, l.y1, last.y0, last.y1)
                gap = l.x0 - last.x1
                crosses_gutter = last.x1 <= mid + 2 and l.x0 >= mid - 2
                if vo > 0.5 * h and -2 < gap < 1.6 * max(l.size, last.size) and not crosses_gutter:
                    g.append(l); placed = True; break
            if not placed:
                groups.append([l])
        for g in groups:
            if len(g) == 1:
                out.append(g[0]); continue
            g.sort(key=lambda l: l.x0)
            chars, sizes, fonts = Counter(), Counter(), []
            for l in g:
                chars[l.font] += len(l.text); sizes[l.size] += len(l.text); fonts += l.fonts
            bb = (min(l.x0 for l in g), min(l.y0 for l in g), max(l.x1 for l in g), max(l.y1 for l in g))
            out.append(Line(pno, bb, normalize(" ".join(l.text for l in g)), sizes.most_common(1)[0][0],
                            chars.most_common(1)[0][0], fonts, g[0].block, maxsize=max(l.maxsize for l in g)))
    out.sort(key=lambda l: (l.page, l.y0, l.x0))
    _attach_dropcaps(out)
    return out


def _attach_dropcaps(lines: list[Line]) -> None:
    sizes = Counter(round(l.size) for l in lines)
    body = sizes.most_common(1)[0][0]
    for l in lines:
        if len(l.text) == 1 and l.text.isalpha() and l.text.isupper() and l.size > 1.8 * body:
            cands = [k for k in lines if k.page == l.page and k is not l and l.x1 - 2 <= k.x0 <= l.x1 + 20
                     and abs(k.y0 - l.y0) < body * 1.0 and len(k.text) > 3]
            if cands:
                k = min(cands, key=lambda k: (abs(k.y0 - l.y0), k.x0 - l.x1))
                k.text = l.text + k.text
                k.bbox = (l.x0, k.y0, k.x1, k.y1)
                l.font = "rot"  # neutralise
                l.text = ""


def _body_font(lines: list[Line]) -> tuple[float, str]:
    # long bibliographies (surveys) can outweigh the body text: only count text before "References"
    ref_at = None
    for ln in lines:
        if re.fullmatch(r"(\d+\.?\s*)?(references|bibliography|literature cited)", ln.text.strip(), re.I):
            ref_at = (ln.page, ln.y0)
            break
    sizes, fams = Counter(), Counter()
    for ln in lines:
        if ln.font == "rot":
            continue
        if ref_at and (ln.page, ln.y0) >= ref_at and ln.page > 0:
            continue
        n = len(ln.text)
        sizes[ln.size] += n
        fams[_family(ln.font)] += n
    return sizes.most_common(1)[0][0], fams.most_common(1)[0][0]


def _looks_like_formula(t: str) -> bool:
    toks = t.split()
    if len(toks) < 3 or not re.search(r"[=+\u2212\u00d7\u00b7\u2211\u222b\u2264\u2265<>\u2208\u2192]", t):
        return False
    words = [x for x in toks if re.fullmatch(r"[A-Za-z]{3,}[.,;:]?", x)]
    return len(words) / len(toks) < 0.3


def _is_math_line(ln: Line) -> bool:
    if not ln.fonts:
        return False
    return sum(bool(MATH_FONT_RE.match(f.split("+")[-1])) for f in ln.fonts) / len(ln.fonts) >= 0.5


def _classify(lines: list[Line], body_size: float, body_family: str, page_rects) -> None:
    # repeated header/footer text (digits stripped) across pages
    marg = Counter()
    for ln in lines:
        h = page_rects[ln.page].height
        if ln.y1 < 0.08 * h or ln.y0 > 0.92 * h:
            marg[re.sub(r"\d+", "#", ln.text)] += 1
    n_pages = len(page_rects)

    for ln in lines:
        if ln.font == "rot":
            ln.kind = "other"
            continue
        h = page_rects[ln.page].height
        in_margin = ln.y1 < 0.08 * h or ln.y0 > 0.92 * h
        key = re.sub(r"\d+", "#", ln.text)
        if CAPTION_RE.match(ln.text) and _looks_like_caption_start(ln, body_size):
            ln.kind = "caption"
            continue
        if in_margin and (marg[key] >= max(2, n_pages // 3) or re.fullmatch(r"#", key) or ln.size < body_size - 1):
            ln.kind = "header"
            continue
        if not ln.text:
            ln.kind = "other"
            continue
        if _is_math_line(ln) or re.fullmatch(r"\(\d+[a-z]?\)", ln.text):
            ln.kind = "math"
            continue
        # small-caps headings ("8 C ONCLUSION"): mostly smaller glyphs but a body-or-larger initial
        if all(_family(f) != body_family or BOLD_RE.search(f) for f in ln.fonts) and (
                ln.maxsize >= body_size + 0.4 or (ln.size >= body_size - 1.2 and HEADING_NUM_RE.match(ln.text)
                                                  and ln.text.replace(" ", "").replace(".", "").isupper())) \
                and len(ln.text) < 80 and (HEADING_NUM_RE.match(ln.text) or ln.text.replace(" ", "").isupper()):
            ln.kind = "heading"
            ln.text = _fix_smallcaps(ln.text)
            continue
        same_size = abs(ln.size - body_size) <= 0.6
        fam = _family(ln.font)
        all_non_body = all(_family(f) != body_family or BOLD_RE.search(f) for f in ln.fonts)
        if same_size or (body_size < ln.size <= body_size + 4.5):
            if all_non_body and len(ln.text) < 120 and (HEADING_NUM_RE.match(ln.text) or ln.text.isupper()
                                                        or BOLD_RE.search(ln.font) or ln.size > body_size + 0.6):
                ln.kind = "heading"
            elif same_size:
                ln.kind = "body"
            else:
                ln.kind = "other"
        else:
            ln.kind = "other"
        _ = fam


ALGO_RE = re.compile(r"^(Algorithm|ALGORITHM|Listing|LISTING)\s*\d+\s*[:.]")


def _mark_algorithms(lines: list[Line], page_hrules, body_size: float) -> None:
    """Pseudo-code floats (algorithm2e/algorithmic) are delimited by horizontal rules; drop their lines."""
    for a in [l for l in lines if ALGO_RE.match(l.text)]:
        rules = sorted([r for r in page_hrules.get(a.page, []) if r[1] > a.y1 - 1 and
                        _overlap(r[0], r[2], a.x0, a.x1) > 0.5 * (a.x1 - a.x0)], key=lambda r: r[1])
        if not rules:
            end_y = a.y1 + 2 * body_size  # unknown extent: drop only the caption region
        elif rules[0][1] - a.y1 < 2.5 * body_size and len(rules) > 1:
            end_y = rules[1][1]
        else:
            end_y = rules[0][1]
        x0 = min(r[0] for r in rules) if rules else a.x0
        x1 = max(r[2] for r in rules) if rules else a.x1
        for l in lines:
            if l.page == a.page and a.y0 - 1 <= l.y0 and l.y1 <= end_y + 1 and _overlap(l.x0, l.x1, x0, x1) > 0.5 * (l.x1 - l.x0):
                l.kind = "algo"


def _fix_smallcaps(t: str) -> str:
    # "8 C ONCLUSION" -> "8 CONCLUSION"; "R ELATED W ORKS" -> "RELATED WORKS"
    return re.sub(r"\b([A-Z]) (?=[A-Z]{2,})", r"\1", t)


def _looks_like_caption_start(ln: Line, body_size: float) -> bool:
    m = CAPTION_RE.match(ln.text)
    if not m:
        return False
    kw = m.group("kw")
    # "TABLE 1" in IEEE style has nothing after it on the line
    if kw.isupper() and not m.group("rest"):
        return True
    if m.group("punct"):
        return True
    # Caption in a distinct (usually smaller or different) font with no punctuation, e.g. "Figure 3 Overview"
    return ln.size < body_size - 0.4 and bool(m.group("rest"))


def _caption_text(lines: list[Line], start: Line) -> tuple[str, tuple]:
    """Caption = the starting line plus following lines of the same block with the same size."""
    same_block = [l for l in lines if l.page == start.page and l.block == start.block and l.y0 >= start.y0 - 1]
    same_block.sort(key=lambda l: (round(l.y0), l.x0))
    parts, bb = [], list(start.bbox)
    for l in same_block:
        if abs(l.size - start.size) > 0.6:
            break
        parts.append(l.text)
        bb = [min(bb[0], l.x0), min(bb[1], l.y0), max(bb[2], l.x1), max(bb[3], l.y1)]
        if l is not start:
            l.kind = "caption"
    # following blocks directly under, same font size, no gap (caption wrapped into new block)
    return _dehyphen_join(parts), tuple(bb)


def _figures(doc, lines, page_graphics, page_rects, body_size) -> list[Figure]:
    figs: list[Figure] = []
    seen = set()
    for ln in [l for l in lines if l.kind == "caption"]:
        m = CAPTION_RE.match(ln.text)
        if not m:
            continue  # continuation line already absorbed
        kind = "table" if m.group("kw").lower().startswith("tab") else "figure"
        num = m.group("num")
        if (kind, num) in seen:
            continue  # e.g. a "Figure 3:" continued across pages; keep first
        seen.add((kind, num))
        text, bb = _caption_text(lines, ln)
        figs.append(Figure(num, kind, ln.page, text, bb))

    by_page: dict[int, list[Figure]] = {}
    for f in figs:
        by_page.setdefault(f.page, []).append(f)
    for pno, pf in by_page.items():
        regions = _assign_regions(pf, [l for l in lines if l.page == pno], page_graphics[pno], page_rects[pno])
        for f in pf:
            f.region = regions.get(id(f))
            if f.region:
                f.subcaptions = _subcaptions(lines, f)
    return figs


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _assign_regions(caps: list[Figure], page_lines: list[Line], graphics, pr) -> dict[int, tuple]:
    """Partition graphic items on a page among captions, then take each caption's cluster bbox.

    Figures: content is above the caption (fallback below). Tables: below (fallback above).
    An item may only be assigned across a gap that contains no running body text.
    """
    blockers = [l for l in page_lines if l.kind in ("body", "heading") and len(l.text) > 25]
    algo = [l.bbox for l in page_lines if l.kind == "algo"]
    if algo:
        ab = (min(b[0] for b in algo) - 8, min(b[1] for b in algo) - 30, max(b[2] for b in algo) + 8, max(b[3] for b in algo) + 8)
        graphics = [g for g in graphics if not (_overlap(g[0], g[2], ab[0], ab[2]) > 0 and _overlap(g[1] - 1, g[3] + 1, ab[1], ab[3]) > 0)]
    items = list(graphics) + [l.bbox for l in page_lines if l.kind == "other" and l.text]

    def blocked(item, cap, direction) -> bool:
        ix0, iy0, ix1, iy1 = item
        cx0, cy0, cx1, cy1 = cap.caption_bbox
        lo, hi = (iy1, cy0) if direction < 0 else (cy1, iy0)
        xs0, xs1 = min(ix0, cx0), max(ix1, cx1)
        for b in blockers:
            if b.y0 >= lo - 1 and b.y1 <= hi + 1 and _overlap(b.x0, b.x1, ix0, ix1) > 0.3 * (ix1 - ix0 + 1e-6):
                return True
        for o in caps:  # never assign across another caption
            if o is cap:
                continue
            ox0, oy0, ox1, oy1 = o.caption_bbox
            if oy0 >= lo - 1 and oy1 <= hi + 1 and _overlap(ox0, ox1, xs0, xs1) > 0:
                return True
        return False

    def gap(item, cap, direction):
        cx0, cy0, cx1, cy1 = cap.caption_bbox
        if direction < 0:
            return cy0 - item[3] if item[3] <= cy0 + 3 else None
        return item[1] - cy1 if item[1] >= cy1 - 3 else None

    assigned: dict[int, list[tuple]] = {id(c): [] for c in caps}
    for it in items:
        best = None
        for pass_overlap in (True, False):
            for c in caps:
                prefs = (-1, 1) if c.kind == "figure" else (1, -1)
                for rank, d in enumerate(prefs):
                    g = gap(it, c, d)
                    if g is None:
                        continue
                    ov = _overlap(it[0], it[2], c.caption_bbox[0] - 5, c.caption_bbox[2] + 5) > 0
                    if pass_overlap and not ov:
                        continue
                    if blocked(it, c, d):
                        continue
                    hdist = 0 if ov else max(c.caption_bbox[0] - it[2], it[0] - c.caption_bbox[2], 0)
                    score = g + hdist + rank * 400
                    if best is None or score < best[0]:
                        best = (score, c)
            if best:
                break
        if best:
            assigned[id(best[1])].append(it)

    out = {}
    for c in caps:
        its = assigned[id(c)]
        if not its:
            continue
        # keep the connected cluster nearest to the caption (drop stray far items)
        cx0, cy0, cx1, cy1 = c.caption_bbox
        its.sort(key=lambda g: (_overlap(g[0], g[2], cx0 - 5, cx1 + 5) <= 0 or (g[2] - g[0]) * (g[3] - g[1]) < 50,
                                min(abs(cy0 - g[3]), abs(g[1] - cy1))))
        reg = list(its[0])
        changed = True
        rest = its[1:]
        while changed:
            changed = False
            keep = []
            for g in rest:
                if _overlap(g[1] - 12, g[3] + 12, reg[1], reg[3]) > 0 and _overlap(g[0] - 12, g[2] + 12, reg[0], reg[2]) > 0:
                    reg = [min(reg[0], g[0]), min(reg[1], g[1]), max(reg[2], g[2]), max(reg[3], g[3])]
                    changed = True
                else:
                    keep.append(g)
            rest = keep
        if reg[2] - reg[0] >= 15 and reg[3] - reg[1] >= 15:
            out[id(c)] = tuple(reg)
    return out


def _subcaptions(lines, fig: Figure) -> dict[str, str]:
    x0, y0, x1, y1 = fig.region
    out = {}
    for l in lines:
        if l.page != fig.page or l.kind == "caption":
            continue
        if _overlap(l.x0, l.x1, x0, x1) > 0 and y0 - 2 <= l.y0 <= max(y1, fig.caption_bbox[1]) + 2:
            parts = re.split(r"(?:^|\s)\(([a-z])\)\s+", " " + l.text)
            # parts = [prefix, letter, text, letter, text, ...]
            for i in range(1, len(parts) - 1, 2):
                txt = parts[i + 1].strip()
                if parts[i] not in out and txt:
                    out[parts[i]] = txt
    return out


def _mark_region_lines(lines, figures) -> None:
    """Body-sized text that sits inside a figure/table region is not running text."""
    for f in figures:
        if not f.region:
            continue
        x0, y0, x1, y1 = f.region
        for l in lines:
            if l.page == f.page and l.kind in ("body", "heading", "math"):
                if _overlap(l.x0, l.x1, x0, x1) > 0.6 * (l.x1 - l.x0) and _overlap(l.y0, l.y1, y0, y1) > 0.6 * (l.y1 - l.y0):
                    l.kind = "other"


# ------------------------------------------------------------------ reading order + paragraphs


def _column_of(l: Line, mid: float, two_col: bool) -> int:
    if not two_col:
        return 0
    if l.x1 <= mid + 8:
        return 0
    if l.x0 >= mid - 8:
        return 1
    return -1  # spans both columns


def _ordered(lines: list[Line], page_rects) -> list[tuple[Line, int]]:
    out = []
    for pno in sorted(page_rects):
        pl = [l for l in lines if l.page == pno and l.kind in ("body", "heading", "math")]
        if not pl:
            continue
        mid = page_rects[pno].width / 2
        bodyl = [l for l in pl if l.kind == "body"]
        two_col = sum(1 for l in bodyl if l.x1 <= mid + 8 or l.x0 >= mid - 8) > 0.7 * max(1, len(bodyl))
        cols = [(l, _column_of(l, mid, two_col)) for l in pl]
        # full-width lines split the page into horizontal bands
        spans = sorted([l.y0 for l, c in cols if c == -1])
        def band(l):
            return sum(1 for y in spans if y < l.y0 - 1)
        cols.sort(key=lambda lc: (band(lc[0]), lc[1] if lc[1] >= 0 else -1, lc[0].y0, lc[0].x0))
        out.extend(cols)
    return out


def _dehyphen_join(parts: list[str], vocab: set[str] | None = None) -> str:
    out = ""
    for p in parts:
        p = p.strip()
        if not out:
            out = p
            continue
        m = re.search(r"(\w+)-$", out)
        if m and p and p[0].islower():
            first = re.match(r"\w+", p)
            joined = m.group(1) + (first.group(0) if first else "")
            hyph = m.group(1) + "-" + (first.group(0) if first else "")
            if vocab is not None and hyph.lower() in vocab and joined.lower() not in vocab:
                out = out + p
            else:
                out = out[:-1] + p
        else:
            out = out + " " + p
    return normalize(out)


def _paragraphs(lines: list[Line], page_rects, body_size: float) -> list[Paragraph]:
    ordered = _ordered(lines, page_rects)
    vocab = set()
    for l in lines:
        if l.kind == "body":
            vocab.update(w.lower() for w in re.findall(r"\w+(?:-\w+)+|\w+", l.text))

    # left margin per (page, column) = most common x0 of body lines
    margins: dict[tuple[int, int], float] = {}
    tmp: dict[tuple[int, int], Counter] = {}
    for l, c in ordered:
        if l.kind == "body":
            tmp.setdefault((l.page, c), Counter())[round(l.x0)] += 1
    for k, cnt in tmp.items():
        margins[k] = cnt.most_common(1)[0][0]
    line_gap = body_size * 1.25
    # typical white space between consecutive body lines (leading); paragraph gaps are clearly larger
    gaps = []
    prev_l = None
    for l, c in ordered:
        if l.kind == "body" and prev_l is not None and prev_l.kind == "body" and prev_l.page == l.page \
                and 0 <= l.y0 - prev_l.y1 < body_size * 2 and _overlap(l.x0, l.x1, prev_l.x0, prev_l.x1) > 0:
            gaps.append(l.y0 - prev_l.y1)
        prev_l = l
    gaps.sort()
    lead = gaps[len(gaps) // 2] if gaps else body_size * 0.25
    para_gap = lead + max(0.45 * body_size, 2.5)

    paras: list[Paragraph] = []
    cur: list[Line] = []
    section = ""
    stopped = False
    prev: tuple[Line, int] | None = None
    in_bullet = False

    def flush():
        nonlocal cur
        if cur:
            text = _dehyphen_join([l.text for l in cur], vocab)
            if len(text) > 1:
                paras.append(Paragraph(len(paras), text, section, sorted({l.page for l in cur}),
                                       [(l.page, l.bbox) for l in cur]))
        cur = []

    for l, c in ordered:
        if stopped:
            break
        if l.kind == "heading":
            in_bullet = False
            # run-in headings that end with ':' or '.' followed by body on same line are not here (mixed fonts)
            flush()
            section = l.text if not (prev and prev[0].kind == "heading" and abs(prev[0].y1 - l.y0) < line_gap) else section + " " + l.text
            if STOP_SECTIONS.match(section.strip()) or STOP_SECTIONS.match(l.text):
                stopped = True
            prev = (l, c)
            continue
        if l.kind == "math":
            prev = (l, c)
            continue  # display equations: dropped, paragraph continues
        margin = margins.get((l.page, c), l.x0)
        indented = l.x0 > margin + 4
        bullet = bool(re.match(r"^[•●▪–\-\*]\s|^\(?\d+[).]\s", l.text))
        new_para = False
        if bullet:
            in_bullet = True
            new_para = True
        elif not cur:
            new_para = True
        elif in_bullet and indented:
            new_para = False
        else:
            in_bullet = False
            pl = cur[-1]
            same_col = pl.page == l.page and (prev and prev[1] == c)
            if indented:
                new_para = True
            elif same_col and l.y0 - pl.y1 > para_gap and (prev and prev[0].kind != "math"):
                new_para = True  # block-style paragraphs separated by vertical space
            elif same_col and pl.x1 < (margin + 0.75 * (_col_width(margins, l.page, page_rects) or 1)) and \
                    re.search(r"[.!?:]\W*$", pl.text):
                new_para = True  # previous line was a short, sentence-final last line
        if new_para:
            flush()
        cur.append(l)
        prev = (l, c)
    flush()
    return paras


def _col_width(margins, page, page_rects) -> float:
    xs = sorted(v for (p, c), v in margins.items() if p == page)
    w = page_rects[page].width
    if len(xs) >= 2:
        return xs[1] - xs[0] - 12
    return w - 2 * (xs[0] if xs else 50)


def _title(lines: list[Line]) -> str:
    first = [l for l in lines if l.page == 0 and l.kind != "header" and l.font != "rot" and len(l.text) > 3]
    if not first:
        return ""
    big = max(l.size for l in first)
    return normalize(" ".join(l.text for l in sorted(first, key=lambda l: l.y0) if abs(l.size - big) < 0.5))
