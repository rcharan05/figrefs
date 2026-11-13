"""Document model shared by all stages: paragraphs -> sentences, figures, explicit mention index."""
from __future__ import annotations

from dataclasses import dataclass, field

from .explicit import Mention, find_bare_mentions, find_mentions
from .pdfparse import Figure, ParsedDoc
from .textutil import split_sentences


@dataclass
class Sentence:
    pid: int     # paragraph id
    sid: int     # index within paragraph
    gid: int     # global index in document order
    text: str
    mentions: list[Mention] = field(default_factory=list)


@dataclass
class Doc:
    parsed: ParsedDoc
    sentences: list[Sentence]
    by_para: dict[int, list[Sentence]]

    @property
    def figures(self) -> list[Figure]:
        return [f for f in self.parsed.figures if f.kind == "figure"]

    @property
    def paragraphs(self):
        return self.parsed.paragraphs


def build_doc(parsed: ParsedDoc) -> Doc:
    valid = {f.id for f in parsed.figures if f.kind == "figure"}
    sents: list[Sentence] = []
    by_para: dict[int, list[Sentence]] = {}
    recent_by_para: list[set[str]] = []
    for p in parsed.paragraphs:
        recent: set[str] = set().union(*recent_by_para[-2:]) if recent_by_para else set()
        own: set[str] = set()
        for i, s in enumerate(split_sentences(p.text)):
            ms = _dedupe(find_mentions(s, valid))
            recent |= {m.figure for m in ms}
            own |= {m.figure for m in ms}
            ms = _dedupe(ms + find_bare_mentions(s, valid, recent))
            st = Sentence(p.id, i, len(sents), s, ms)
            sents.append(st)
            by_para.setdefault(p.id, []).append(st)
        recent_by_para.append(own)
    return Doc(parsed, sents, by_para)


def _dedupe(ms: list[Mention]) -> list[Mention]:
    """One Mention per figure per sentence, with the union of sub-figure letters."""
    by: dict[str, Mention] = {}
    for m in ms:
        if m.figure in by:
            by[m.figure].subfigs += [x for x in m.subfigs if x not in by[m.figure].subfigs]
        else:
            by[m.figure] = Mention(m.figure, list(m.subfigs), m.start, m.end, m.text)
    return list(by.values())
