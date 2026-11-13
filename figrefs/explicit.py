"""Explicit figure mentions: "Fig. 3", "Figure 3(b)", "Figs. 4–6", "Figures 9(a), 10(a), and 11(a)", "Fig. 5a,b"."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TRIG_PL = r"(?:Figs\.?|Figures|FIGS\.?|FIGURES|figs\.|figures)"
TRIG_SG = r"(?:Fig\.?|Figure|FIG\.?|FIGURE|fig\.|figure)"
TRIGGER = r"(?:" + TRIG_PL + "|" + TRIG_SG + ")"
# sub-figure letters: "(b)", "(a)-(c)", "(a) and (b)", "(a,b)", or bare suffix "5a", "5a,b", "5a-c"
SUBL = (r"(?:\s?\([a-z]\)(?:\s*(?:,\s*(?:and\s*)?|and|&|–|—|-|to)\s*\([a-z]\))*"
        r"|\s?\([a-z](?:\s*[,–\-]\s*[a-z])*\)"
        r"|[a-z](?:\s*[,–\-]\s*[a-z])*(?![A-Za-z]))")
NUMID = r"S?\d+(?:\.\d+)?"          # 3, 3.2 (chapter-numbered), S1 (supplementary)
ITEM = NUMID + r"(?:" + SUBL + r")?(?![\d])"
SEP_PL = r"(?:\s*(?:,\s*(?:and|&)?|and|&|to|–|—|-|through)\s*)"
SEP_SG = r"(?:\s*(?:and|&|to|–|—|-|through)\s*)"   # "Fig. 7, 17 chart types" is not a list
MENTION_RE = re.compile(
    r"(?<![A-Za-z])(?:" + TRIG_PL + r"\s*(?P<pl>" + ITEM + r"(?:" + SEP_PL + r"(?:" + TRIGGER + r"\s*)?" + ITEM + r")*)"
    r"|" + TRIG_SG + r"\s*(?P<sg>" + ITEM + r"(?:" + SEP_SG + r"(?:" + TRIGGER + r"\s*)?" + ITEM + r")*))")
_ITEM_RE = re.compile(r"(?P<n>" + NUMID + r")(?P<s>(?:" + SUBL + r")?)")
_RANGE_RE = re.compile(r"(?<![\d.])(\d+)(?:" + SUBL + r")?\s*(?:–|—|-|to|through)\s*(?:" + TRIGGER + r"\s*)?(\d+)(?![\d.])")


@dataclass
class Mention:
    figure: str
    subfigs: list[str]
    start: int
    end: int
    text: str


@dataclass
class FigureRefs:
    mentions: list[tuple[int, int, Mention]] = field(default_factory=list)  # (para_id, sent_idx, mention)


def _letters(s: str) -> list[str]:
    s = re.sub(r"\s*(and|&)\s*", ",", s)
    s = s.replace("(", "").replace(")", "").replace(" ", "").replace("to", "-").replace("\u2014", "-")
    s = re.sub(r",+", ",", s).strip(",")
    if not s:
        return []
    out = []
    for part in s.split(","):
        m = re.fullmatch(r"([a-z])[–\-–]([a-z])", part)
        if m:
            out += [chr(c) for c in range(ord(m.group(1)), ord(m.group(2)) + 1)]
        elif re.fullmatch(r"[a-z]", part):
            out.append(part)
    return out


def find_mentions(text: str, valid: set[str] | None = None) -> list[Mention]:
    out: list[Mention] = []
    for m in MENTION_RE.finditer(text):
        body = m.group("pl") or m.group("sg")
        nums: list[tuple[str, list[str]]] = []
        for it in _ITEM_RE.finditer(body):
            nums.append((it.group("n"), _letters(it.group("s") or "")))
        # expand numeric ranges "4–6" / "4 to 6"
        for r in _RANGE_RE.finditer(body):
            a, b = int(r.group(1)), int(r.group(2))
            if a < b <= a + 30:
                for k in range(a + 1, b):
                    nums.append((str(k), []))
        seen = {}
        for n, subs in nums:
            seen.setdefault(n, [])
            seen[n] += [s for s in subs if s not in seen[n]]
        for n, subs in seen.items():
            if valid is not None and n not in valid:
                continue
            out.append(Mention(n, subs, m.start(), m.end(), m.group(0)))
    return out


_BARE_RE = re.compile(r"(?<![\w.])(\d+)\(([a-z])\)")  # "(11(a))", "Compared to 4(a)" after the figure was named


def find_bare_mentions(text: str, valid: set[str], context_figs: set[str]) -> list[Mention]:
    out = []
    for m in _BARE_RE.finditer(text):
        if m.group(1) in valid and m.group(1) in context_figs:
            out.append(Mention(m.group(1), [m.group(2)], m.start(), m.end(), m.group(0)))
    return out
