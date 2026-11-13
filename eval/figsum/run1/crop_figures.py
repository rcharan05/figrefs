"""Crop every scored figure of the 18 FigSum+ papers out of its PDF, for image-to-text (run 4 SPAN-VIS).

For figure F: find the page whose text holds F's caption start ("Fig. F" / "Figure F" label followed by the first
caption words, else just the label at a line start); the figure is the union of embedded images on that page that
lie above the caption (below the previous caption on the page, if any). If the page has no images there, the
region between the previous text block above and the caption is rendered instead (vector figures).
Rendered at 150 dpi to OUT/<paper>_<fig>.png; OUT/index.json maps "paper|fig" -> file or null.
usage: python crop_figures.py RUN1_DATA.jsonl FIGSUM_ROOT OUT_DIR
"""
from __future__ import annotations

import json
import os
import re
import sys

import pymupdf

LABEL = r"(?:Fig(?:ure)?\.?|FIG(?:URE)?\.?)\s*{n}\b"


def words(s, k=6):
    return [w for w in re.findall(r"[A-Za-z]{3,}", s)][:k]


def caption_hits(doc, n, cap):
    """[(page index, rect of caption start)], best first."""
    lab = re.compile(LABEL.format(n=n))
    hits = []
    cw = [w.lower() for w in words(cap)]
    for pn, page in enumerate(doc):
        for b in page.get_text("blocks"):
            lines = [l.strip() for l in b[4].split("\n") if l.strip()]
            k = next((j for j, l in enumerate(lines[:3]) if lab.match(l)), None)   # BMC: title line, then "Figure N"
            if k is None:
                continue
            t = " ".join(" ".join(lines[k:]).split())
            m = lab.match(t)
            rest = t[m.end():].lower()
            score = sum(w in rest[:300] for w in cw)
            hits.append((score, pn, pymupdf.Rect(b[:4])))
    hits.sort(key=lambda h: (-h[0], h[1]))
    return [(pn, r) for s, pn, r in hits if s >= 2] or [(pn, r) for s, pn, r in hits]


def region(page, cap_rect, other_caps):
    above = [r for r in other_caps if r.y1 <= cap_rect.y0 - 1]
    top = max([r.y1 for r in above], default=page.rect.y0 + 60)
    imgs = []
    for info in page.get_images(full=True):
        for r in page.get_image_rects(info[0]):
            if r.width < 60 or r.height < 60 or r.y1 < page.rect.y0 + 75:   # skip journal header logos
                continue
            if r.y1 <= cap_rect.y0 + 5 and r.y0 >= top - 5:
                imgs.append(r)
    if imgs:
        u = imgs[0]
        for r in imgs[1:]:
            u |= r
        return u
    # vector figure: from the last text block above (that is not a caption) to the caption
    blocks = [pymupdf.Rect(b[:4]) for b in page.get_text("blocks")
              if b[3] <= cap_rect.y0 - 2 and len(b[4].split()) > 25 and b[3] > top]
    y0 = max([b.y1 for b in blocks], default=top)
    if cap_rect.y0 - y0 < 80:
        return None
    return pymupdf.Rect(page.rect.x0 + 20, y0, page.rect.x1 - 20, cap_rect.y0)


def main():
    data, root, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    index, pdf_text = {}, {}
    for line in open(data):
        a = json.loads(line)
        pdf = os.path.join(root, a["id"], "paper.pdf")
        for f in a["figs"]:
            index[f"{a['id']}|{f}"] = None
        if not os.path.exists(pdf):
            continue
        doc = pymupdf.open(pdf)
        for f in a["figs"]:
            cap = a["captions"].get(str(f), "")
            hits = caption_hits(doc, f, cap)
            for pn, cr in hits[:3]:
                page = doc[pn]
                others = [r for g in a["figs"] if g != f for p2, r in caption_hits(doc, g, a["captions"].get(str(g), ""))[:1]
                          if p2 == pn]
                rg = region(page, cr, others)
                if rg is None or rg.width < 80 or rg.height < 80:
                    continue
                fn = f"{a['id']}_{f}.png"
                page.get_pixmap(clip=rg & page.rect, dpi=150).save(os.path.join(out, fn))
                index[f"{a['id']}|{f}"] = fn
                body = pymupdf.Rect(page.rect.x0, page.rect.y0 + 60, page.rect.x1, page.rect.y1 - 50)  # no headers
                txt = " ".join(page.get_text(clip=rg & body).split())             # text layer (vector figures)
                txt = re.sub(r"https?://\S+|BMC \w+ \d{4}, \d+:\d+|Vol \d+ No \d+|Available online", " ", txt)
                txt = " ".join(txt.split())
                if len(re.findall(r"[A-Za-z0-9]", txt)) >= 8:
                    pdf_text[f"{a['id']}|{f}"] = txt[:600]
                break
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)
    json.dump(pdf_text, open(os.path.join(out, "pdf_text.json"), "w"), indent=1, ensure_ascii=False)
    print(f"text layer found for {len(pdf_text)} figures")
    got = sum(v is not None for v in index.values())
    print(f"cropped {got} of {len(index)} figures; missing: {[k for k, v in index.items() if v is None]}")


if __name__ == "__main__":
    main()
