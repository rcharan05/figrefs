"""Any paper -> the article model used by every method (run1_data.jsonl format), from LaTeX, JATS XML or PDF.

  from_latex(src_dir)  arXiv source: figures numbered as LaTeX numbers them (latex_gt.ground_truth), \\ref-like
                       commands to figure labels become "Fig. N", figure/table/equation environments removed,
                       paragraphs = blank-line blocks, sections from \\section / \\subsection
  from_jats(xml)       eval/figsum/data.load_article (PMC / bioRxiv XML)
  from_pdf(pdf)        the tool's own PyMuPDF parser (figrefs.pdfparse + doc.build_doc): paragraphs, sections,
                       captions and explicit mentions with the tool's grammar
Output dict: id, split "new", source, title, sentences, sent_para, para_section, captions {fig: text}, cue, figs
(every detected figure), gold {} and the empty training-only fields the trainers expect.
usage: python demo/article_from_source.py OUT.jsonl ID=PATH [ID=PATH ...]   (PATH: a LaTeX source dir, .xml or .pdf)
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "eval", "figsum"))
from figrefs.explicit import find_mentions  # noqa: E402
from figrefs.textutil import normalize, split_sentences  # noqa: E402

MAJOR = re.compile(r"(?i)^(\d+(\.\d+)*\.?\s+)?(abstract|introduction|background|related work|results?|discussion|"
                   r"conclusions?|methods?|materials and methods|evaluation|experiments?|implementation|"
                   r"results and discussion|overview|method|approach|applications?|limitations)\b")


def _finish(pid, source, title, paras, para_sec, figures):
    sents, sp, cue = [], [], []
    for k, t in enumerate(paras):
        for s in split_sentences(t):
            if len(re.findall(r"[A-Za-z]{2,}", s)) < 3:
                continue
            figs = {int(m.figure) for m in find_mentions(s) if m.figure.isdigit()} & set(figures)
            sents.append(s)
            sp.append(k)
            cue.append(sorted(figs))
    return {"id": pid, "split": "new", "source": source, "title": title, "sentences": sents, "sent_para": sp,
            "para_section": para_sec, "captions": {str(f): c for f, c in figures.items()}, "cue": cue,
            "figs": sorted(figures), "gold": {}, "masked": sents, "bt": {}, "bt_masked": {}, "cdel": {}, "fold": -1}


def from_latex(pid, src_dir):
    from latex_gt import FIG_ENVS, find_main, flatten, ground_truth, ref_macros, to_text
    base = ground_truth(src_dir)
    if not base:
        raise SystemExit(f"{pid}: no LaTeX main file / figures found in {src_dir}")
    label2fig = {lab: f["number"] for f in base["figures"] for lab in f["labels"]}
    figures = {f["number"]: normalize(f["caption"])[:1500] for f in base["figures"]}
    tex = flatten(find_main(src_dir))
    macros = ref_macros(tex)
    tm = re.search(r"\\title\s*(\[[^\]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}", tex)
    title = to_text(tm.group(2)) if tm else pid
    b = tex.find("\\begin{document}")
    body = tex[b:] if b >= 0 else tex
    cut = len(body)
    for pat in (r"\\bibliography\s*\{", r"\\printbibliography", r"\\begin\{thebibliography\}", r"\\appendix\b",
                r"\\section\*?\{\s*(Acknowledge?ments?|References)"):
        m = re.search(pat, body)
        if m:
            cut = min(cut, m.start())
    body = body[:cut]
    am = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    abstract = am.group(1) if am else ""
    body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", " ", body, flags=re.S)
    body = re.sub(r"\\begin\{(" + FIG_ENVS + r"|table\*?|algorithm\*?|equation\*?|align\*?|eqnarray\*?|lstlisting|"
                  r"tabular\*?)\}.*?\\end\{\1\}", "\n\n", body, flags=re.S)
    body = re.sub(r"\\teaser\s*\{", "\\\\teaserX{", body)
    ref_re = re.compile(r"\\(" + "|".join(sorted(map(re.escape, macros), key=len, reverse=True)) + r")\*?\s*\{([^}]+)\}")

    def rep(m):
        nums = [str(label2fig[l.strip()]) for l in m.group(2).split(",") if l.strip() in label2fig]
        if not nums:
            return "??"
        return ", ".join(nums) if m.group(1) == "ref" else "Fig. " + ", ".join(nums)

    paras, para_sec = [], []
    if abstract.strip():
        paras.append(normalize(to_text(ref_re.sub(rep, abstract))))
        para_sec.append("Abstract")
    sec = "Introduction"
    parts = re.split(r"(\\(?:sub)*section\*?\s*(?:\[[^\]]*\])?\s*\{(?:[^{}]|\{[^{}]*\})*\})", body)
    for part in parts:
        hm = re.match(r"\\((?:sub)*)section\*?\s*(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}", part)
        if hm:
            if not hm.group(1):                       # top-level \section only
                sec = to_text(hm.group(2)) or sec
            continue
        for p in re.split(r"\n\s*\n|\\par\b|\\paragraph\*?\s*\{[^}]*\}", part):
            t = normalize(to_text(ref_re.sub(rep, p)))
            if len(re.findall(r"[A-Za-z]{3,}", t)) >= 5:
                paras.append(t)
                para_sec.append(sec)
    return _finish(pid, "latex", title, paras, para_sec, figures)


def from_jats(pid, xml):
    from data import load_article
    a = load_article(pid, xml)
    figures = {f: normalize(c)[:1500] for f, c in a.figures.items()}
    out = _finish(pid, "jats", pid, a.paragraphs, [a.top_section[k] for k in range(len(a.paragraphs))], figures)
    return out


def from_pdf(pid, pdf):
    from figrefs.doc import build_doc
    from figrefs.pdfparse import parse_pdf
    d = build_doc(parse_pdf(pdf))
    figures = {int(f.id): normalize(f.caption)[:1500] for f in d.figures if str(f.id).isdigit()}
    paras, para_sec, cur = [], [], "Abstract"
    for p in d.paragraphs:
        s = (p.section or "").strip()
        if s and (MAJOR.match(s) or re.match(r"^\d+\.?\s+\S", s)) and not re.match(r"^\d+\.\d+", s):
            cur = s
        paras.append(normalize(p.text))
        para_sec.append(cur)
    return _finish(pid, "pdf", d.parsed.title or pid, paras, para_sec, figures)


def build(pid, path):
    if os.path.isdir(path):
        return from_latex(pid, path)
    if path.lower().endswith(".xml"):
        return from_jats(pid, path)
    if path.lower().endswith(".pdf"):
        return from_pdf(pid, path)
    raise SystemExit(f"unknown input type: {path}")


if __name__ == "__main__":
    out = sys.argv[1]
    with open(out, "w") as fh:
        for spec in sys.argv[2:]:
            pid, path = spec.split("=", 1)
            a = build(pid, path)
            n_cite = sum(1 for c in a["cue"] if c)
            print(f"{pid}: {a['source']}  '{a['title'][:70]}'  sentences {len(a['sentences'])}  figures {len(a['figs'])}  "
                  f"citing sentences {n_cite}  uncited figures {[f for f in a['figs'] if not any(f in c for c in a['cue'])]}")
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
