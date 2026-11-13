"""Derive ground truth for explicit figure references from a paper's LaTeX source.

For each figure (numbered in source order of top-level \\caption in figure-like environments, plus a
TVCG \\teaser), collect its labels; then every \\ref-like command to those labels in the body text gives a
(figure number, paragraph) pair. Paragraphs are LaTeX paragraphs (blank-line separated), converted to plain text.
"""
from __future__ import annotations

import os
import re

from pylatexenc.latex2text import LatexNodes2Text

FIG_ENVS = r"(?:figure\*?|teaserfigure|wrapfigure|SCfigure\*?|subfigure\*?)"
REF_CMDS = ["ref", "autoref", "cref", "Cref", "figref", "Figref", "fref", "Fref", "subref", "vref", "Vref",
            "cpageref", "nameref", "figureref", "Figureref", "fig", "Fig", "reffig", "figurename"]


def _read(path: str) -> str:
    try:
        return open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return ""


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def find_main(src_dir: str) -> str | None:
    cands = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith(".tex"):
                t = _read(os.path.join(root, f))
                if "\\begin{document}" in t:
                    cands.append((len(t), os.path.join(root, f)))
    return max(cands)[1] if cands else None


def flatten(path: str, depth: int = 0) -> str:
    base = os.path.dirname(path)
    s = strip_comments(_read(path))
    if depth > 6:
        return s

    def rep(m):
        name = m.group(2).strip()
        for cand in (name, name + ".tex"):
            for root in (base, os.path.dirname(base)):
                p = os.path.join(root, cand)
                if os.path.isfile(p):
                    return flatten(p, depth + 1)
        return ""
    return re.sub(r"\\(input|include|subfile)\s*\{([^}]+)\}", rep, s)


def _balanced(s: str, i: int) -> int:
    """index just past the balanced {...} starting at s[i] == '{'"""
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == "\\":
            continue
        if c == "{" and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif c == "}" and (j == 0 or s[j - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return j + 1
    return len(s)


def _arg(s: str, cmd_end: int) -> tuple[str, int]:
    i = cmd_end
    while i < len(s) and s[i] in " \t\n*":
        i += 1
    if i < len(s) and s[i] == "[":  # optional arg
        depth = 0
        for j in range(i, len(s)):
            if s[j] == "[":
                depth += 1
            elif s[j] == "]":
                depth -= 1
                if depth == 0:
                    i = j + 1
                    break
    while i < len(s) and s[i] in " \t\n":
        i += 1
    if i < len(s) and s[i] == "{":
        e = _balanced(s, i)
        return s[i + 1:e - 1], e
    return "", i


def ref_macros(tex: str) -> set[str]:
    """User macros wrapping \\ref, e.g. \\newcommand{\\figref}[1]{Fig.~\\ref{#1}}"""
    out = set(REF_CMDS)
    for m in re.finditer(r"\\(?:re)?newcommand\*?\s*\{?\\(\w+)\}?\s*(?:\[\d\])?\s*\{", tex):
        body, _ = _arg(tex, m.end() - 1)
        if re.search(r"\\(c|C|auto)?ref\s*\{\s*(fig:)?#1", body):
            out.add(m.group(1))
    for m in re.finditer(r"\\def\\(\w+)#1\{", tex):
        body, _ = _arg(tex, m.end() - 1)
        if re.search(r"\\(c|C|auto)?ref\s*\{\s*(fig:)?#1", body):
            out.add(m.group(1))
    return out


_l2t = LatexNodes2Text(math_mode="text", strict_latex_spaces=False)


def to_text(s: str) -> str:
    s = re.sub(r"\\(cite[tp]?|citep|citet|citeauthor|nocite)\s*(\[[^\]]*\])?\{[^}]*\}", " [1]", s)
    s = re.sub(r"\\(label|vspace|hspace|includegraphics|centering|small|footnotesize)\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", s)
    try:
        t = _l2t.latex_to_text(s)
    except Exception:
        t = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
        t = re.sub(r"[{}$]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def ground_truth(src_dir: str) -> dict | None:
    main = find_main(src_dir)
    if not main:
        return None
    tex = flatten(main)
    macros = ref_macros(tex)
    # body up to bibliography / appendix (our PDF parser stops at References)
    b = tex.find("\\begin{document}")
    body = tex[b:] if b >= 0 else tex
    cut = len(body)
    for pat in (r"\\bibliography\s*\{", r"\\printbibliography", r"\\begin\{thebibliography\}", r"\\appendix\b",
                r"\\section\*?\{\s*(Acknowledge?ments?|References)"):
        m = re.search(pat, body)
        if m:
            cut = min(cut, m.start())
    body = body[:cut]
    pre = tex[:b] if b >= 0 else ""

    # ---- figures in numbering order
    figures: list[dict] = []
    label2fig: dict[str, int] = {}
    spans: list[tuple[int, int]] = []
    items = []
    for m in re.finditer(r"\\teaser\s*\{", pre + body):
        items.append((-1 if m.start() < len(pre) else m.start() - len(pre), "teaser", m))
    for m in re.finditer(r"\\begin\{(" + FIG_ENVS + r")\}", body):
        items.append((m.start(), m.group(1), m))
    items.sort(key=lambda x: x[0])
    for pos, kind, m in items:
        if kind == "teaser":
            src = pre + body
            content, end = _arg(src, m.end() - 1)
        else:
            if kind.startswith("subfigure"):
                continue
            env = kind
            e = body.find("\\end{" + env + "}", m.end())
            if e < 0:
                continue
            content = body[m.end():e]
            spans.append((m.start(), e + len(env) + 6))
        # top-level captions (a figure env may hold several minipage figures); sub-captions don't count
        top = re.sub(r"\\begin\{subfigure\*?\}.*?\\end\{subfigure\*?\}", lambda m: " " * len(m.group(0)), content, flags=re.S)
        caps = [m for m in re.finditer(r"\\caption\s*(\[[^\]]*\])?\s*\{", top)]
        if not caps:
            continue
        starts = [c.start() for c in caps]
        nums = []
        for c in caps:
            cap, _ = _arg(top, c.end() - 1)
            num = len(figures) + 1
            figures.append({"number": num, "caption": to_text(cap), "labels": []})
            nums.append(num)
        for lm in re.finditer(r"\\label\s*\{([^}]+)\}", content):
            k = max([i for i, st in enumerate(starts) if st <= lm.start()] or [0])
            # a label placed before its caption inside a minipage belongs to the next caption
            if lm.start() < starts[0]:
                k = 0
            lab = lm.group(1).strip()
            label2fig[lab] = nums[k]
            figures[nums[k] - 1]["labels"].append(lab)

    # ---- body text without figure/table environments
    text = body
    for s, e in sorted(spans, reverse=True):
        text = text[:s] + "\n\n" + text[e:]
    text = re.sub(r"\\begin\{(table\*?|algorithm\*?|equation\*?|align\*?)\}.*?\\end\{\1\}", "\n", text, flags=re.S)
    paras = re.split(r"\n\s*\n|\\par\b|\\(?:sub)*section\*?\s*\{[^}]*\}|\\paragraph\*?\s*\{[^}]*\}", text)

    ref_re = re.compile(r"\\(" + "|".join(sorted(map(re.escape, macros), key=len, reverse=True)) + r")\*?\s*\{([^}]+)\}")
    mentions = []
    for p in paras:
        found: dict[int, list[str]] = {}
        for m in ref_re.finditer(p):
            for lab in m.group(2).split(","):
                lab = lab.strip()
                if lab in label2fig:
                    found.setdefault(label2fig[lab], []).append(lab)
        if not found:
            continue
        # replace refs to figures by their number so the plain text is comparable with the PDF
        def rep(m):
            nums = [str(label2fig[l.strip()]) for l in m.group(2).split(",") if l.strip() in label2fig]
            if not nums:
                return "??"
            pref = "" if m.group(1) == "ref" else "Fig. "
            return pref + ", ".join(nums)
        plain = to_text(ref_re.sub(rep, p))
        for fig in found:
            mentions.append({"figure": fig, "paragraph": plain})
    return {"main": os.path.relpath(main, src_dir), "figures": figures, "mentions": mentions}


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(ground_truth(sys.argv[1]), indent=1)[:4000])
