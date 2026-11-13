"""End-to-end: PDF -> figures + explicit paragraphs + implicit sentences -> JSON (+ figure crops, review HTML)."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pymupdf

from .doc import Doc, build_doc
from .explicit import find_mentions
from .implicit import score_implicit
from .pdfparse import parse_pdf


def process_pdf(pdf_path: str, out_dir: str, dpi: int = 200, threshold: float = 0.5,
                dense: str | None = "bge", rerank: str | None = "bge-reranker", save_images: bool = True,
                implicit_mode: str = "rule") -> dict:
    parsed = parse_pdf(pdf_path)
    doc = build_doc(parsed)
    stem = Path(pdf_path).stem
    pdir = Path(out_dir) / stem
    (pdir / "figures").mkdir(parents=True, exist_ok=True)

    if implicit_mode == "rule":
        from .implicit_rule import score_implicit_rule
        implicit = score_implicit_rule(doc)
    else:  # legacy v0.3 scorer
        implicit = score_implicit(doc, dense=dense, rerank=rerank, threshold=threshold)

    pdf = pymupdf.open(pdf_path)
    figures_out = []
    valid = {f.id for f in doc.figures}
    for f in doc.figures:
        img_rel = None
        if f.region and save_images:
            img_rel = f"figures/fig_{f.id}.png"
            pdf[f.page].get_pixmap(dpi=dpi, clip=pymupdf.Rect(f.region)).save(str(pdir / img_rel))

        # explicit: whole paragraphs containing a mention, with the mention sentences marked
        exp_paras: dict[int, dict] = {}
        for s in doc.sentences:
            for m in s.mentions:
                if m.figure != f.id:
                    continue
                p = doc.paragraphs[s.pid]
                e = exp_paras.setdefault(s.pid, {
                    "paragraph_id": p.id, "section": p.section, "pages": [x + 1 for x in p.pages],
                    "paragraph": p.text, "mention_sentences": []})
                e["mention_sentences"].append({"sentence_index": s.sid, "text": s.text, "subfigures": m.subfigs})

        # other figures' captions that point at this figure ("The original dataset is shown in Figure 18.")
        cap_mentions = []
        for g in doc.figures:
            if g.id == f.id:
                continue
            body = re.sub(r"^(Fig\.?|Figure|FIG\.?)\s*\d+\s*[.:|]?", "", g.caption)
            if any(m.figure == f.id for m in find_mentions(body, valid)):
                cap_mentions.append({"from_figure": g.id, "caption": g.caption})

        imp = []
        for h in implicit.get(f.id, []):
            imp.append({"paragraph_id": h.sentence.pid, "sentence_index": h.sentence.sid, "text": h.sentence.text,
                        "score": h.score, "selected": h.score >= threshold, "signals": h.signals})

        relevant = sorted(
            [(pid, -1, e["paragraph"]) for pid, e in exp_paras.items()] +
            [(i["paragraph_id"], i["sentence_index"], i["text"]) for i in imp if i["selected"]])
        figures_out.append({
            "figure_id": f.id,
            "page": f.page + 1,
            "bbox": [round(x, 1) for x in f.region] if f.region else None,
            "image": img_rel,
            "caption": f.caption,
            "subcaptions": f.subcaptions,
            "explicit": [exp_paras[k] for k in sorted(exp_paras)],
            "caption_mentions": cap_mentions,
            "implicit": imp,
            "relevant_text": [t for _, _, t in relevant],
        })

    result = {
        "paper": {"file": os.path.basename(pdf_path), "title": parsed.title, "pages": parsed.n_pages},
        "n_paragraphs": len(doc.paragraphs),
        "figures": figures_out,
        "unmatched_mentions": _unmatched(doc, valid),
    }
    (pdir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    from .report import write_report
    write_report(result, doc, pdir)
    return result


def _unmatched(doc: Doc, valid: set[str]) -> list[str]:
    """Figure numbers mentioned in text for which no caption was found (diagnostic for parsing failures)."""
    nums = set()
    for p in doc.paragraphs:
        for m in find_mentions(p.text, None):
            if m.figure not in valid:
                nums.add(m.figure)
    return sorted(nums, key=lambda x: (len(x), x))


def process_dir(in_dir: str, out_dir: str, **kw) -> list[dict]:
    pdfs = sorted(str(p) for p in Path(in_dir).glob("**/*.pdf")) if os.path.isdir(in_dir) else [in_dir]
    results = []
    for p in pdfs:
        try:
            r = process_pdf(p, out_dir, **kw)
            print(f"[ok] {os.path.basename(p)}: {len(r['figures'])} figures")
            results.append(r)
        except Exception as e:  # keep going on bad PDFs
            print(f"[fail] {os.path.basename(p)}: {e}")
            results.append({"paper": {"file": os.path.basename(p)}, "error": str(e)})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "all_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return results
