"""Re-implementation of GROBID's own end-to-end metric for the `reference_figure` field, to check that our
GROBID setup reproduces the published bioRxiv numbers (GROBID 0.9.0: soft P 71.0 / R 77.85 / F1 74.26).

Follows grobid-trainer EndToEndEvaluation.java (tag 0.9.0):
  gold   = //xref[@ref-type="fig"]/text()     in the JATS
  grobid = //ref[@type="figure"]/text()       in the TEI (plain run, no sentence segmentation)
  each string -> basicNormalizationFullText (drop "figure"/"fig."/"fig", collapse spaces, strip brackets/commas)
  soft = removeFullPunct (lowercase, drop punctuation and spaces)
  per document: for each gold string, if it is in the (remaining) grobid list -> match, remove it (multiset)
  TP = matches, FN = gold - matches, FP = grobid - matches; micro-averaged over documents.
usage: python eval/repro_grobid.py ../bench/biorxiv 090
"""
import os
import sys
import unicodedata

from lxml import etree

FULL_PUNCT = "(（[ •*,:;?.!/)）-−–‐«»„\"“”‘’'`$#@]*♦♥♣♠ 。、，・" + "‐ \t\n\r " + \
    "·◼▲►◆○◇●◎◽◸◹◺"


def basic_norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s.strip())
    for a, b in (("\n", " "), ("\t", " "), ("_", " "), (" ", " ")):
        s = s.replace(a, b)
    for w in ("figure", "Figure", "fig.", "Fig.", "fig", "Fig"):
        s = s.replace(w, "")
    while "  " in s:
        s = s.replace("  ", " ")
    if s.startswith("[") or s.startswith("("):
        s = s[1:]
    while s.endswith("]") or s.endswith(")") or s.endswith(","):
        s = s[:-1]
    return s.strip()


def soft(s: str) -> str:
    return "".join(c for c in s.lower() if c not in FULL_PUNCT)


def values(tree, xpath, ns=None):
    out = []
    for v in tree.xpath(xpath, namespaces=ns or {}):
        n = basic_norm(str(v))
        if n:
            out.append(n)
    return out


def main():
    root, tag = sys.argv[1], sys.argv[2]
    tot = {"strict": [0, 0, 0], "soft": [0, 0, 0]}  # tp, n_gold, n_pred
    docs = 0
    for d in sorted(os.listdir(root)):
        tei_p = os.path.join(root, d, f"grobid_{tag}.tei.xml")
        xml_p = os.path.join(root, d, "paper.xml")
        if not (os.path.exists(tei_p) and os.path.exists(xml_p)):
            continue
        gold = values(etree.parse(xml_p), '//xref[@ref-type="fig"]/text()')
        tei = etree.parse(tei_p)
        # GROBID's XPath runs without a namespace context on a namespace-aware DOM; use local-name() for the same nodes
        pred = values(tei, '//*[local-name()="ref"][@type="figure"]/text()')
        docs += 1
        for mode, f in (("strict", lambda x: x), ("soft", soft)):
            P = [f(x) for x in pred]
            tp = 0
            for g in gold:
                if f(g) in P:
                    P.remove(f(g))
                    tp += 1
            tot[mode][0] += tp
            tot[mode][1] += len(gold)
            tot[mode][2] += len(pred)
    print(f"documents: {docs}")
    for mode, (tp, g, p) in tot.items():
        P, R = tp / max(1, p), tp / max(1, g)
        print(f"{mode:6s} precision {100*P:.2f}  recall {100*R:.2f}  f1 {200*P*R/max(1e-9, P+R):.2f}  support {g}")


if __name__ == "__main__":
    main()
