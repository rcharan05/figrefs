"""Implicit references, rule B0f (validated on FigSum+ GS1).

For every sentence that cites figure F, the sentence immediately before and the one immediately after are taken as
implicit references to F when they are in the same paragraph, and they are dropped when they:
  - cite a different figure,
  - are 'offtarget' (tables, supplementary figures, "data not shown", additional files, web site),
  - lie in a biomedical-style methods section ("Materials and Methods", "Methods", "Experimental Procedures").
    A CS "Method"/"Proposed method" section is NOT dropped: that is where CS figures are discussed.
"""
from __future__ import annotations

import re

from .doc import Doc
from .implicit import ImplicitHit

OFFTARGET = re.compile(r"data not shown|not shown|supplement|additional file|figure s\d|fig\.? s\d|\btables?\b|web site",
                       re.I)
METHODS_HEAD = re.compile(r"^\s*(?:[\dIVX]+\.?\s*)*(?:materials?\s+and\s+methods|methods\s+and\s+materials|methods|"
                          r"experimental\s+procedures|methods\s+summary|online\s+methods)\s*$", re.I)


def score_implicit_rule(doc: Doc, **_ignored) -> dict[str, list[ImplicitHit]]:
    out: dict[str, list[ImplicitHit]] = {f.id: [] for f in doc.figures}
    sents = doc.sentences
    sections = {p.id: (p.section or "") for p in doc.paragraphs}
    for f in doc.figures:
        cites = [s for s in sents if any(m.figure == f.id for m in s.mentions)]
        seen: set[int] = set()
        for s in cites:
            for side, g in (("before", s.gid - 1), ("after", s.gid + 1)):
                if not (0 <= g < len(sents)) or g in seen:
                    continue
                c = sents[g]
                if c.pid != s.pid or any(m.figure == f.id for m in c.mentions):
                    continue
                if c.mentions:                                   # cites another figure
                    continue
                if OFFTARGET.search(c.text):
                    continue
                if METHODS_HEAD.match(sections.get(c.pid, "").strip().split("\n")[-1]):
                    continue
                seen.add(g)
                out[f.id].append(ImplicitHit(c, 1.0, {"rule": "B0f", "side_of_citation": 1.0 if side == "after" else -1.0}))
        out[f.id].sort(key=lambda h: h.sentence.gid)
    return out
