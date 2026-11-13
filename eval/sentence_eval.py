"""Sentence-level benchmark of explicit figure references: (figure number, sentence) pairs.

usage:
  python eval/sentence_eval.py --data eval/arxiv --gt latex   --out eval/results_sentence_arxiv.json
  python eval/sentence_eval.py --data ../bench/biorxiv --gt jats --out eval/results_sentence_biorxiv.json

Systems
  ours                 figrefs PDF parser + explicit matcher (sentences from our splitter)
  grobid<TAG>          GROBID TEI with segmentSentences=1: its own sentences, its own <ref type="figure">
  grobid<TAG>+split    GROBID paragraphs re-split with OUR sentence splitter (isolates splitter differences)

Matching: within a paper, a predicted pair and a GT pair match when the figure is the same and the sentence
token-F1 >= tau. One-to-one greedy matching by descending F1. tau = 0.5 ("lenient") and 0.9 ("strict").
Precision = matched predictions / predictions; recall = matched GT / GT (micro over papers).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentence_gt import jats_sentences, latex_sentences  # noqa: E402

TOK = re.compile(r"[a-z0-9]+")
TEI = "{http://www.tei-c.org/ns/1.0}"


def toks(s: str) -> frozenset:
    return frozenset(TOK.findall(s.lower()))


def tf1(a, b) -> float:
    if not a or not b:
        return 0.0
    i = len(a & b)
    return 2 * i / (len(a) + len(b))


# ---------------------------------------------------------------- systems

def run_ours(pdf: str, cache_tag: str | None = None) -> list[tuple[int, str]]:
    """cache_tag: reuse <dir>/ours_<tag>.json (delete it after changing the parser)."""
    cache = os.path.join(os.path.dirname(pdf), f"ours_{cache_tag}.json") if cache_tag else None
    if cache and os.path.exists(cache):
        return [tuple(x) for x in json.load(open(cache))]
    from figrefs.doc import build_doc
    from figrefs.pdfparse import parse_pdf
    d = build_doc(parse_pdf(pdf))
    out = [(int(m.figure), s.text) for s in d.sentences for m in s.mentions if m.figure.isdigit()]
    if cache:
        json.dump(out, open(cache, "w"))
    return out


def _tei_figmap(root) -> dict[str, int]:
    id2num = {}
    for fig in root.iter(TEI + "figure"):
        if fig.get("type") == "table":
            continue
        fid = fig.get("{http://www.w3.org/XML/1998/namespace}id")
        lab, head = fig.find(TEI + "label"), fig.find(TEI + "head")
        for cand in [lab.text if lab is not None else None, "".join(head.itertext()) if head is not None else None]:
            if re.search(r"(?i)\b(S\d|supp)", cand or ""):
                break  # supplementary figure: not a main figure number
            m = re.search(r"(\d+)", cand or "")
            if m and fid:
                id2num[fid] = int(m.group(1))
                break
    return id2num


def _ref_nums(r, id2num) -> list[int]:
    tgt = (r.get("target") or "").lstrip("#")
    if tgt in id2num:
        return [id2num[tgt]]
    if tgt:
        return []  # resolved to a figure we do not count (supplementary)
    # unresolved target: every number in the ref text ("Figures 2 and 3" may come as one ref), not "S2"
    txt = "".join(r.itertext())
    if re.search(r"(?i)supp", txt):
        return []
    return [int(n) for n in re.findall(r"(?<![\d.SsA-Za-z])(\d{1,2})(?![\d.]\d)", txt)]


def _text_containers(root):
    for path in (f".//{TEI}profileDesc/{TEI}abstract", f".//{TEI}text/{TEI}body"):
        for c in root.findall(path):
            yield c


def run_grobid_native(tei_path: str) -> list[tuple[int, str]]:
    root = ET.parse(tei_path).getroot()
    id2num = _tei_figmap(root)
    out = []
    for c in _text_containers(root):
        for s in c.iter(TEI + "s"):
            text = re.sub(r"\s+", " ", "".join(s.itertext())).strip()
            for r in s.iter(TEI + "ref"):
                if r.get("type") == "figure":
                    out += [(n, text) for n in _ref_nums(r, id2num)]
    return out


def run_grobid_oursplit(tei_path: str) -> list[tuple[int, str]]:
    from figrefs.textutil import split_sentences
    root = ET.parse(tei_path).getroot()
    id2num = _tei_figmap(root)
    out = []
    for c in _text_containers(root):
        for p in c.iter(TEI + "p"):
            parts = [p.text or ""]

            def walk(el):
                for ch in el:
                    if ch.tag == TEI + "ref" and ch.get("type") == "figure":
                        parts.append("".join(ch.itertext()) + "".join(f"ZZFR{n}ZZ" for n in _ref_nums(ch, id2num)))
                    else:
                        parts.append(ch.text or "")
                        walk(ch)
                    if ch.tag == TEI + "s":
                        parts.append(" ")  # <s> elements are not whitespace-separated in the TEI
                    parts.append(ch.tail or "")
            walk(p)
            text = re.sub(r"\s+", " ", "".join(parts)).strip()
            for s in split_sentences(text):
                for n in re.findall(r"ZZFR(\d+)ZZ", s):
                    out.append((int(n), re.sub(r"\s*ZZFR\d+ZZ", "", s)))
    return out


# ---------------------------------------------------------------- filters (arXiv only; system-independent)

def caption_alignment(gt: dict, pdfdoc) -> float:
    from figrefs.textutil import normalize
    raw = normalize(" ".join(p.get_text() for p in pdfdoc)).lower()
    ok = 0
    for f in gt["figures"]:
        words = re.findall(r"[a-z]{3,}", f["caption"].lower())[:3]
        if not words:
            ok += 1
            continue
        pat = r"fig(?:ure)?\.?\s*" + str(f["number"]) + r"(?!\d).{0,60}?" + r"\W+(?:\w{1,2}\W+)*".join(map(re.escape, words))
        ok += bool(re.search(pat, raw))
    return ok / max(1, len(gt["figures"]))


def filter_gt(gt: dict, pdfdoc) -> dict:
    """Drop GT sentences whose text is found in the PDF but without the figure number nearby
    (LaTeX source and PDF disagree: different version, duplicated label, ...)."""
    from figrefs.textutil import normalize
    raw = normalize(" ".join(p.get_text() for p in pdfdoc))
    raw_l = raw.lower()
    keep, dropped = [], 0
    for m in gt["mentions"]:
        words = re.findall(r"[A-Za-z]{3,}", m["sentence"])
        figpat = re.compile(r"(?i)fig(?:ure)?s?\.?\s*(?:[\d\s,and&–\-()a-z]{0,40}?)(?<!\d)" + str(m["figure"]) + r"(?!\d)")
        hits, ok = 0, None
        for k in range(0, max(1, len(words) - 4), 3):
            pat = r"\W+".join(map(re.escape, (w.lower() for w in words[k:k + 5])))
            for hit in re.finditer(pat, raw_l):
                hits += 1
                if figpat.search(raw[max(0, hit.start() - 400): hit.end() + 400]):
                    ok = True
                    break
            if ok:
                break
        if not ok and hits >= 2:
            ok = False
        if ok is False:
            dropped += 1
            continue
        keep.append(m)
    return {**gt, "mentions": keep, "dropped": dropped}


# ---------------------------------------------------------------- scoring

def match(G: list[tuple[int, frozenset]], P: list[tuple[int, frozenset]], tau: float) -> int:
    cand = []
    for i, (gf, gt_) in enumerate(G):
        for j, (pf, pt) in enumerate(P):
            if gf == pf:
                s = tf1(gt_, pt)
                if s >= tau:
                    cand.append((s, i, j))
    cand.sort(reverse=True)
    ug, up, n = set(), set(), 0
    for s, i, j in cand:
        if i not in ug and j not in up:
            ug.add(i)
            up.add(j)
            n += 1
    return n


def score(gt: dict, pred: list[tuple[int, str]]) -> dict:
    G = sorted({(m["figure"], toks(m["sentence"])) for m in gt["mentions"]}, key=str)
    P = sorted({(f, toks(t)) for f, t in pred}, key=str)
    return {"gt": len(G), "pred": len(P), "tp_05": match(G, P, 0.5), "tp_09": match(G, P, 0.9)}


def summarize(rows: list[dict]) -> dict:
    g, p = sum(r["gt"] for r in rows), sum(r["pred"] for r in rows)
    out = {"papers": len(rows), "gt_pairs": g, "pred_pairs": p}
    for tau in ("05", "09"):
        tp = sum(r[f"tp_{tau}"] for r in rows)
        P, R = tp / max(1, p), tp / max(1, g)
        out[f"tau{tau}"] = {"precision": round(P, 3), "recall": round(R, 3),
                            "f1": round(2 * P * R / max(1e-9, P + R), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gt", choices=["latex", "jats"], required=True)
    ap.add_argument("--tags", default="082,090")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated paper ids (default: all)")
    ap.add_argument("--cache", default=None, help="cache our predictions per paper under this tag")
    a = ap.parse_args()
    import pymupdf
    tags = [t for t in a.tags.split(",") if t]
    systems = ["ours"] + [f"grobid{t}" for t in tags] + [f"grobid{t}+split" for t in tags]
    rows = {s: [] for s in systems}
    per_paper = []
    only = set(a.only.split(",")) - {""}
    for d in sorted(os.listdir(a.data)):
        if only and d not in only:
            continue
        pdir = os.path.join(a.data, d)
        pdf = os.path.join(pdir, "paper.pdf")
        if not os.path.exists(pdf):
            continue
        try:
            gt = latex_sentences(os.path.join(pdir, "src")) if a.gt == "latex" else jats_sentences(os.path.join(pdir, "paper.xml"))
        except Exception as e:
            print(f"skip {d}: GT error {e}")
            continue
        if not gt or len(gt["figures"]) < 2 or not gt["mentions"]:
            print(f"skip {d}: no usable ground truth")
            continue
        pdfdoc = pymupdf.open(pdf)
        if pdfdoc.page_count > 60:
            print(f"skip {d}: {pdfdoc.page_count} pages")
            continue
        if a.gt == "latex":
            al = caption_alignment(gt, pdfdoc)
            if al < 0.6:
                print(f"skip {d}: LaTeX numbering does not match the PDF ({al:.2f})")
                continue
            gt = filter_gt(gt, pdfdoc)
        tei_missing = [t for t in tags if not os.path.exists(os.path.join(pdir, f"grobid_{t}_sent.tei.xml"))]
        if tei_missing:
            print(f"skip {d}: no GROBID output for {tei_missing}")
            continue
        rec = {"paper": d, "gt_pairs": len({(m['figure'], m['sentence']) for m in gt["mentions"]}),
               "gt_dropped": gt.get("dropped", 0)}
        for s in systems:
            try:
                if s == "ours":
                    pred = run_ours(pdf, a.cache)
                else:
                    tag = s[6:9]
                    tei = os.path.join(pdir, f"grobid_{tag}_sent.tei.xml")
                    pred = run_grobid_oursplit(tei) if s.endswith("+split") else run_grobid_native(tei)
                sc = score(gt, pred)
            except Exception as e:
                print(f"{s} failed on {d}: {e}")
                sc = {"gt": rec["gt_pairs"], "pred": 0, "tp_05": 0, "tp_09": 0, "error": str(e)[:200]}
            rows[s].append(sc)
            rec[s] = sc
        per_paper.append(rec)
        print(d, {s: (rec[s]["tp_05"], rec[s]["pred"], rec[s]["gt"]) for s in systems}, flush=True)
    summary = {s: summarize(rows[s]) for s in systems if rows[s]}
    json.dump({"summary": summary, "per_paper": per_paper}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
