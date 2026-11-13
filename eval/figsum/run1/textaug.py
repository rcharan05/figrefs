"""Sentence-level text operations shared by the data export and back-translation.

mention_spans(s)  merged character spans of explicit figure mentions (figrefs grammar + attached sub-figure letters)
mask(s)           every figure mention (any figure, incl. supplementary) -> "Fig. [X]" (stage-1 input: the model
                  must match on content, not on the number; one token for all figures keeps the grammar intact)
cite_delete(s,f)  s with its parenthetical citation(s) of figure f removed, e.g. "... higher (Fig. 3A)." ->
                  "... higher."; None if f is also cited outside a parenthesis or the parenthesis holds other text
                  (A3 augmentation; non-parenthetical deletion leaves telltale fragments, see the M2 gotcha)
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from figrefs.explicit import find_mentions  # noqa: E402

LETTERS = re.compile(r"(?:[A-Za-z](?:\s?[,–\-&]\s?[A-Za-z])*|\s?\([a-z]\))(?![A-Za-z])")
MASK = "Fig. [X]"


def mention_spans(s):
    spans = []
    for m in find_mentions(s):
        st, en = m.start, m.end
        tail = LETTERS.match(s, en)
        if tail:
            en = tail.end()
        spans.append((st, en, m.figure))
    spans.sort()
    merged = []
    for st, en, f in spans:
        if merged and st <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], en)
            merged[-1][2].add(f)
        else:
            merged.append([st, en, {f}])
    return [(st, en, figs) for st, en, figs in merged]


def mask(s):
    out, last = [], 0
    for st, en, _ in mention_spans(s):
        out.append(s[last:st])
        out.append(MASK)
        last = en
    out.append(s[last:])
    return "".join(out)


def cite_delete(s, f):
    f = str(f)
    spans = [sp for sp in mention_spans(s) if f in sp[2]]
    if not spans:
        return None
    cuts = []
    for st, en, figs in spans:
        if figs != {f}:
            return None
        lo, hi = s.rfind("(", 0, st), s.find(")", en)
        if lo < 0 or hi < 0 or ")" in s[lo:st] or "(" in s[en:hi]:
            return None
        inner = (s[lo + 1:st] + s[en:hi]).strip(" ,;")
        if inner and not re.fullmatch(r"(?i)(see|cf\.?|also|e\.g\.,?)\s*", inner):
            return None
        a = lo - 1 if lo > 0 and s[lo - 1] == " " else lo
        cuts.append((a, hi + 1))
    out, last = [], 0
    for a, b in sorted(cuts):
        out.append(s[last:a])
        last = b
    out.append(s[last:])
    t = re.sub(r"\s+([.,;:])", r"\1", "".join(out)).strip()
    return t if t != s else None


if __name__ == "__main__":
    for s in ["IL-17 levels were higher (Fig. 1).", "As shown in Figs. 2 and 3B, the band (Figure 4A,B) and Fig. S2 were",
              "Figure 5 shows that (Figs 3-5) were up", "data (Fig. 2; Table 1)", "was reduced (see Fig. 3B), and",
              "levels rose (Fig. 2A) and fell (Fig. 2C)."]:
        print(repr(s), "\n   mask:", repr(mask(s)), "\n   del1:", cite_delete(s, 1), "| del2:", cite_delete(s, 2),
              "| del3:", cite_delete(s, 3), "| del4:", cite_delete(s, 4))
