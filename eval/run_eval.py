"""Benchmark explicit-reference extraction against LaTeX-derived ground truth.

Systems:  ours   — figrefs PDF parser + explicit matcher
          grobid — GROBID TEI (<ref type="figure">) via a running GROBID server (http://localhost:8070)

Metrics (per system, micro-averaged over papers):
  figures   : caption recall (GT figure number found with a similar caption)
  mentions  : (figure, paragraph) pairs. A predicted pair matches a GT pair for the same figure when their
              paragraph token-F1 >= 0.5. Recall over GT pairs, precision over predicted pairs.
  boundary  : mean token-F1 of matched paragraph pairs, and share with F1 >= 0.9 ("whole paragraph, no more, no less")
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
import uuid
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from latex_gt import ground_truth  # noqa: E402

from figrefs.doc import build_doc  # noqa: E402
from figrefs.pdfparse import parse_pdf  # noqa: E402

TOK = re.compile(r"[a-z]{3,}")


def toks(s: str) -> set[str]:
    return set(TOK.findall(s.lower()))


def f1(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    i = len(a & b)
    return 0 if i == 0 else 2 * i / (len(a) + len(b))


# ---------------------------------------------------------------- systems

def run_ours(pdf: str) -> dict:
    d = build_doc(parse_pdf(pdf))
    pairs = set()
    for s in d.sentences:
        for m in s.mentions:
            if m.figure.isdigit():
                pairs.add((int(m.figure), s.pid))
    return {"figures": {int(f.id): f.caption for f in d.figures if f.id.isdigit()},
            "mentions": [(f, d.paragraphs[pid].text) for f, pid in sorted(pairs)]}


def _post_pdf(url: str, pdf: str) -> bytes:
    boundary = uuid.uuid4().hex
    data = open(pdf, "rb").read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"input\"; filename=\"p.pdf\"\r\n"
            f"Content-Type: application/pdf\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def run_grobid(pdf: str, server: str = "http://localhost:8070", cache_dir: str | None = None) -> dict:
    cache = os.path.join(cache_dir or os.path.dirname(pdf), "grobid.tei.xml")
    if os.path.exists(cache):
        tei = open(cache, "rb").read()
    else:
        tei = _post_pdf(server + "/api/processFulltextDocument", pdf)
        open(cache, "wb").write(tei)
    ns = {"t": "http://www.tei-c.org/ns/1.0"}
    root = ET.fromstring(tei)
    figs, id2num = {}, {}
    for fig in root.iter("{http://www.tei-c.org/ns/1.0}figure"):
        if fig.get("type") == "table":
            continue
        fid = fig.get("{http://www.w3.org/XML/1998/namespace}id")
        lab = fig.find("t:label", ns)
        head = fig.find("t:head", ns)
        num = None
        for cand in [lab.text if lab is not None else None, "".join(head.itertext()) if head is not None else None]:
            if cand:
                m = re.search(r"(\d+)", cand)
                if m:
                    num = int(m.group(1))
                    break
        if num is None:
            continue
        desc = fig.find("t:figDesc", ns)
        figs[num] = " ".join(desc.itertext()) if desc is not None else ""
        if fid:
            id2num[fid] = num
    mentions = []
    body = root.find(".//t:text/t:body", ns)
    if body is not None:
        for p in body.iter("{http://www.tei-c.org/ns/1.0}p"):
            text = re.sub(r"\s+", " ", "".join(p.itertext())).strip()
            found = set()
            for r in p.iter("{http://www.tei-c.org/ns/1.0}ref"):
                if r.get("type") != "figure":
                    continue
                tgt = (r.get("target") or "").lstrip("#")
                if tgt in id2num:
                    found.add(id2num[tgt])
                else:  # unresolved target: fall back to the number in the ref text
                    m = re.search(r"(\d+)", "".join(r.itertext()))
                    if m:
                        found.add(int(m.group(1)))
            for f in found:
                mentions.append((f, text))
    return {"figures": figs, "mentions": mentions}


def caption_alignment(gt: dict, pdfdoc) -> float:
    """Share of GT figures whose number+caption opening appears in the raw PDF text (system-independent)."""
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
    """Drop GT mentions whose figure number does not appear in the PDF near that paragraph's text.

    arXiv source and PDF occasionally disagree (different versions, duplicated labels, appendix numbering).
    Check is system-independent: locate the paragraph's opening words in the raw PDF text and require
    "Fig..N" within the following window.
    """
    from figrefs.textutil import normalize
    raw = normalize(" ".join(p.get_text() for p in pdfdoc))
    raw_l = raw.lower()
    keep, dropped = [], 0
    for m in gt["mentions"]:
        words = re.findall(r"[A-Za-z]{3,}", m["paragraph"])
        figpat = re.compile(r"(?i)fig(?:ure)?s?\.?\s*(?:[\d\s,and&–\-()a-z]{0,40}?)(?<!\d)" + str(m["figure"]) + r"(?!\d)")
        hits, ok = 0, None
        for k in range(0, max(1, len(words) - 5), 5):  # every anchor of 5 consecutive words
            pat = r"\W+".join(map(re.escape, (w.lower() for w in words[k:k + 5])))
            for hit in re.finditer(pat, raw_l):
                hits += 1
                if figpat.search(raw[max(0, hit.start() - 700): hit.end() + 700]):
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

def score(gt: dict, pred: dict) -> dict:
    gt_figs = {f["number"]: f["caption"] for f in gt["figures"]}
    fig_hit = sum(1 for n, c in gt_figs.items() if n in pred["figures"] and f1(toks(c), toks(pred["figures"][n])) >= 0.4)
    G = [(m["figure"], toks(m["paragraph"])) for m in gt["mentions"]]
    P = [(f, toks(t)) for f, t in pred["mentions"]]
    g_matched, p_matched, bf = set(), set(), []
    for i, (gf, gt_t) in enumerate(G):
        best, bj = 0.0, None
        for j, (pf, pt) in enumerate(P):
            if pf != gf:
                continue
            s = f1(gt_t, pt)
            if s > best:
                best, bj = s, j
        if best >= 0.5:
            g_matched.add(i)
            p_matched.add(bj)
            bf.append(best)
    for j, (pf, pt) in enumerate(P):
        if j in p_matched:
            continue
        if any(gf == pf and f1(gt_t, pt) >= 0.5 for gf, gt_t in G):
            p_matched.add(j)
    return {"gt_figs": len(gt_figs), "fig_hit": fig_hit, "gt_m": len(G), "pred_m": len(P),
            "tp_g": len(g_matched), "tp_p": len(p_matched), "bf": bf}


def summarize(rows: list[dict]) -> dict:
    s = {k: sum(r[k] for r in rows) for k in ("gt_figs", "fig_hit", "gt_m", "pred_m", "tp_g", "tp_p")}
    bf = [x for r in rows for x in r["bf"]]
    rec = s["tp_g"] / max(1, s["gt_m"])
    prec = s["tp_p"] / max(1, s["pred_m"])
    return {"papers": len(rows), "figure_recall": round(s["fig_hit"] / max(1, s["gt_figs"]), 3),
            "mention_recall": round(rec, 3), "mention_precision": round(prec, 3),
            "mention_f1": round(2 * rec * prec / max(1e-9, rec + prec), 3),
            "boundary_f1_mean": round(sum(bf) / max(1, len(bf)), 3),
            "boundary_exact_rate(F1>=0.9)": round(sum(x >= 0.9 for x in bf) / max(1, len(bf)), 3),
            "gt_figures": s["gt_figs"], "gt_mentions": s["gt_m"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval/arxiv")
    ap.add_argument("--systems", default="ours,grobid")
    ap.add_argument("--out", default="eval/results.json")
    a = ap.parse_args()
    systems = a.systems.split(",")
    rows = {s: [] for s in systems}
    per_paper = []
    for d in sorted(os.listdir(a.data)):
        pdir = os.path.join(a.data, d)
        pdf = os.path.join(pdir, "paper.pdf")
        if not os.path.exists(pdf):
            continue
        gt = ground_truth(os.path.join(pdir, "src"))
        if not gt or len(gt["figures"]) < 2 or not gt["mentions"]:
            print(f"skip {d}: no usable LaTeX ground truth")
            continue
        import pymupdf
        pdfdoc = pymupdf.open(pdf)
        if pdfdoc.page_count > 60:
            print(f"skip {d}: {pdfdoc.page_count} pages (thesis/book)")
            continue
        al = caption_alignment(gt, pdfdoc)
        if al < 0.6:
            print(f"skip {d}: LaTeX figure numbering does not match the PDF (alignment {al:.2f})")
            continue
        gt = filter_gt(gt, pdfdoc)
        rec = {"paper": d, "gt_figures": len(gt["figures"]), "gt_mentions": len(gt["mentions"]),
               "gt_dropped_inconsistent": gt["dropped"]}
        for s in systems:
            try:
                pred = run_ours(pdf) if s == "ours" else run_grobid(pdf)
                sc = score(gt, pred)
                rows[s].append(sc)
                rec[s] = {k: v for k, v in sc.items() if k != "bf"}
            except Exception as e:
                rec[s] = {"error": str(e)[:200]}
                print(f"{s} failed on {d}: {e}")
        per_paper.append(rec)
        print(d, {s: rec.get(s) for s in systems})
    summary = {s: summarize(rows[s]) for s in systems if rows[s]}
    json.dump({"summary": summary, "per_paper": per_paper}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
