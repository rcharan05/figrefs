"""Sentence-level ground truth for explicit figure references.

Two sources, both independent of every system under test:
  latex_sentences(src_dir)  arXiv LaTeX source: every \\ref-like command to a figure label
  jats_sentences(xml_path)  publisher JATS XML (bioRxiv / PMC): every <xref ref-type="fig">

The reference sentence is found by splitting the source paragraph with pysbd (a third-party rule-based splitter,
not ours) after placing an alphanumeric marker at each reference. Literal mentions typed without a \\ref / xref
("Figure 3") are added too, so that a system is not penalised for finding them.
Output: {"figures": [{"number", "caption"}], "mentions": [{"figure", "sentence", "paragraph", "source"}]}
"""
from __future__ import annotations

import re

import pysbd

from latex_gt import FIG_ENVS, _arg, find_main, flatten, ground_truth, ref_macros, to_text

_seg = pysbd.Segmenter(language="en", clean=False)
MARK = re.compile(r"ZZFR(\d+)ZZ")
LITERAL = re.compile(r"\b(?:Fig(?:ure)?s?\.?)\s*(\d+)(?![\d.]\d)")


def _split(text: str) -> list[str]:
    return [s.strip() for s in _seg.segment(text) if s.strip()]


def _mentions_from_paragraph(marked: str, valid: set[int], source: str) -> list[dict]:
    """marked: paragraph text with ZZFR<n>ZZ placed right after each tagged reference."""
    out = []
    plain_para = re.sub(r"\s+", " ", MARK.sub("", marked)).strip()
    for s in _split(marked):
        figs = {int(n) for n in MARK.findall(s)}
        clean = re.sub(r"\s+", " ", MARK.sub("", s)).strip()
        tagged = {f for f in figs if f in valid}
        literal = {int(n) for n in LITERAL.findall(clean) if int(n) in valid} - tagged
        for f in sorted(tagged):
            out.append({"figure": f, "sentence": clean, "paragraph": plain_para, "source": source})
        for f in sorted(literal):
            out.append({"figure": f, "sentence": clean, "paragraph": plain_para, "source": source + "-literal"})
    return out


# ---------------------------------------------------------------- LaTeX (arXiv)

def latex_sentences(src_dir: str) -> dict | None:
    base = ground_truth(src_dir)  # figure numbering + labels, as in the paragraph-level benchmark
    if not base:
        return None
    label2fig = {lab: f["number"] for f in base["figures"] for lab in f["labels"]}
    valid = {f["number"] for f in base["figures"]}
    tex = flatten(find_main(src_dir))
    macros = ref_macros(tex)
    b = tex.find("\\begin{document}")
    body = tex[b:] if b >= 0 else tex
    cut = len(body)
    for pat in (r"\\bibliography\s*\{", r"\\printbibliography", r"\\begin\{thebibliography\}", r"\\appendix\b",
                r"\\section\*?\{\s*(Acknowledge?ments?|References)"):
        m = re.search(pat, body)
        if m:
            cut = min(cut, m.start())
    body = body[:cut]
    # remove figure environments (their captions are not body text)
    spans = []
    for m in re.finditer(r"\\begin\{(" + FIG_ENVS + r")\}", body):
        env = m.group(1)
        if env.startswith("subfigure"):
            continue
        e = body.find("\\end{" + env + "}", m.end())
        if e >= 0:
            spans.append((m.start(), e + len(env) + 6))
    text = body
    for s, e in sorted(spans, reverse=True):
        text = text[:s] + "\n\n" + text[e:]
    text = re.sub(r"\\teaser\s*\{", "\\\\teaserX{", text)
    text = re.sub(r"\\begin\{(table\*?|algorithm\*?|equation\*?|align\*?)\}.*?\\end\{\1\}", "\n", text, flags=re.S)
    paras = re.split(r"\n\s*\n|\\par\b|\\(?:sub)*section\*?\s*\{[^}]*\}|\\paragraph\*?\s*\{[^}]*\}", text)
    ref_re = re.compile(r"\\(" + "|".join(sorted(map(re.escape, macros), key=len, reverse=True)) + r")\*?\s*\{([^}]+)\}")
    mentions = []
    for p in paras:
        if not ref_re.search(p) and not LITERAL.search(p):
            continue

        def rep(m):
            nums = [label2fig[l.strip()] for l in m.group(2).split(",") if l.strip() in label2fig]
            if not nums:
                return "??"
            pref = "" if m.group(1) == "ref" else "Fig. "
            return pref + ", ".join(f"{n}ZZFR{n}ZZ" for n in nums)
        marked = to_text(ref_re.sub(rep, p))
        mentions += _mentions_from_paragraph(marked, valid, "ref")
    return {"figures": [{"number": f["number"], "caption": f["caption"]} for f in base["figures"]],
            "mentions": mentions}


# ---------------------------------------------------------------- JATS (bioRxiv / PMC)

FIG_LABEL = re.compile(r"^\s*(?:Fig(?:ure)?\.?)\s*(\d+)\s*[.:|]?\s*$", re.I)


def _local(tag) -> str:
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def jats_sentences(xml_path: str) -> dict:
    from lxml import etree
    root = etree.parse(xml_path).getroot()
    rid2num, figures = {}, []
    for f in root.iter("{*}fig"):
        lab = f.find("{*}label")
        lab_t = "".join(lab.itertext()) if lab is not None else ""
        m = FIG_LABEL.match(lab_t)
        if not m:
            continue  # supplementary ("Figure S1"), schemes, unlabeled
        n = int(m.group(1))
        cap = f.find("{*}caption")
        figures.append({"number": n, "caption": re.sub(r"\s+", " ", "".join(cap.itertext())).strip() if cap is not None else ""})
        if f.get("id"):
            rid2num[f.get("id")] = n
    valid = {f["number"] for f in figures}

    def serialise(el) -> str:
        parts = [el.text or ""]
        for ch in el:
            name = _local(ch.tag)
            if name in ("fig", "table-wrap", "disp-formula", "fn"):
                pass
            elif name == "xref" and ch.get("ref-type") == "fig":
                nums = [rid2num[r] for r in (ch.get("rid") or "").split() if r in rid2num]
                parts.append("".join(ch.itertext()) + "".join(f"ZZFR{n}ZZ" for n in nums))
            else:
                parts.append(serialise(ch))
            parts.append(ch.tail or "")
        return "".join(parts)

    mentions = []
    containers = list(root.iter("{*}abstract")) + list(root.iter("{*}body"))
    for c in containers:
        for p in c.iter("{*}p"):
            if any(_local(a.tag) in ("fig", "table-wrap", "caption", "fn", "boxed-text") for a in p.iterancestors()):
                continue
            if any(_local(d.tag) == "p" for d in p.iterdescendants()):
                continue  # take the innermost paragraphs only
            marked = re.sub(r"\s+", " ", serialise(p)).strip()
            if marked:
                mentions += _mentions_from_paragraph(marked, valid, "xref")
    return {"figures": sorted(figures, key=lambda f: f["number"]), "mentions": mentions}


if __name__ == "__main__":
    import json
    import sys
    p = sys.argv[1]
    g = jats_sentences(p) if p.endswith(".xml") else latex_sentences(p)
    print(len(g["figures"]), "figures", len(g["mentions"]), "mentions")
    for m in g["mentions"][:8]:
        print(m["figure"], m["source"], "|", m["sentence"][:150])
