"""Text normalisation and sentence splitting tuned for scientific prose."""
from __future__ import annotations

import re
import unicodedata

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}


def normalize(text: str) -> str:
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen
    return re.sub(r"\s+", " ", text).strip()


# Abbreviations after which a period does NOT end a sentence.
_ABBREV = [
    "Fig", "Figs", "fig", "figs", "Eq", "Eqs", "eq", "eqs", "Sec", "Secs", "sec", "Sect", "Tab", "Tabs",
    "al", "e.g", "i.e", "vs", "cf", "approx", "resp", "Ref", "Refs", "ref", "Dr", "Prof", "Mr", "Ms",
    "Ch", "Chap", "No", "Nos", "Vol", "pp", "ca", "Alg", "Def", "Thm", "Lem", "Prop", "Appx", "App",
    "w.r.t", "i.i.d", "a.k.a", "U.S", "U.K", "Ph.D", "St", "Jr", "Inc", "Corp", "viz", "Eqn", "Eqns", "Supp", "Suppl",
]
_ABBREV_RE = re.compile(r"(?<![\w.])(" + "|".join(re.escape(a) for a in sorted(_ABBREV, key=len, reverse=True)) + r")\.")
_INITIAL_RE = re.compile(r"(?<![\w.])([A-Z])\.(?=\s+[A-Z][a-z])")  # "J. Smith"
_DOT = "․"  # one-dot leader used as placeholder
_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences; returns sentences whose join (with single spaces) equals the text."""
    if not text:
        return []
    protected = _ABBREV_RE.sub(lambda m: m.group(1) + _DOT, text)
    protected = _INITIAL_RE.sub(lambda m: m.group(1) + _DOT, protected)
    # decimals / version numbers never split ("0.2", "5.1.2") — they have no whitespace after the dot so are safe.
    parts, last = [], 0
    for m in _SPLIT_RE.finditer(protected):
        cut = m.end()
        seg = protected[last:cut].strip()
        if seg:
            parts.append(seg)
        last = cut
    tail = protected[last:].strip()
    if tail:
        parts.append(tail)
    # merge fragments that are too short to be sentences (e.g. "[15].")
    merged: list[str] = []
    for p in parts:
        if merged and (len(p) < 4 or re.fullmatch(r"[\[\]\d,\s–\-.;)(]+", p)):
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return [p.replace(_DOT, ".") for p in merged]
