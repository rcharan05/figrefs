"""Labelling pack for AI annotators (Claude, Gemini): one prompt per paper, built from the SAME sentence list the
methods are run on, so the labels line up exactly. The annotator's JSON answer is loaded straight into the viewer.
usage: python demo/make_label_pack.py ARTICLES.jsonl OUT_DIR
"""
from __future__ import annotations

import json
import os
import sys

PDF_HINT = {"prof_tvcg2019": "the attached PDF (TVCG 2019, Biswas et al.)"}

INSTR = """You are annotating a research paper for a benchmark of figure-associated text, following the FigSum+
annotation guideline (Ramesh, Sethi & Yu, PLoS ONE 2015, gold standard "GS1").

TASK. For EVERY figure listed below, select ALL sentences of the paper that are associated with that figure:
  - sentences that explicitly cite the figure ("Fig. 3 shows ...", "(Figure 2b)"), and
  - sentences that do not name the figure but describe, explain, set up or interpret what the figure shows
    (its method, data, result or conclusion), e.g. the sentences right after a citation that keep discussing it,
    a sentence before the citation that introduces the experiment shown, or Discussion sentences interpreting it.
Do NOT select sentences that only mention the same general topic, or that are about a different figure or a table.
A sentence may belong to more than one figure. Use the figure images in {pdf} to understand each figure.

Use ONLY the sentence numbers in the list below (they come from our own sentence splitter; do not re-split).

ANSWER with exactly one JSON object and nothing else:
{{"paper": "{pid}", "labeler": "<your model name, e.g. claude or gemini>",
  "figures": {{"1": [sentence numbers], "2": [ ... ], ...}}}}
Every figure number below must appear as a key (use [] if no sentence is associated)."""


def main():
    arts = [json.loads(l) for l in open(sys.argv[1])]
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for a in arts:
        pid = a["id"]
        pdf = PDF_HINT.get(pid, f"the attached PDF (arXiv {pid}, https://arxiv.org/pdf/{pid})")
        lines = [INSTR.format(pid=pid, pdf=pdf), "", f"PAPER: {a.get('title', pid)}", "", "FIGURES:"]
        for f in a["figs"]:
            lines.append(f"Figure {f}: {a['captions'][str(f)]}")
        lines += ["", "SENTENCES (number: text; '¶' marks a new paragraph, '§' a new section):"]
        last_p, last_s = None, None
        for i, s in enumerate(a["sentences"]):
            p = a["sent_para"][i]
            sec = a["para_section"][p] if p < len(a["para_section"]) else ""
            if sec != last_s:
                lines.append(f"§ {sec}")
                last_s = sec
            mark = "¶ " if p != last_p else "  "
            last_p = p
            lines.append(f"{mark}{i}: {s}")
        open(os.path.join(out, f"prompt_{pid}.txt"), "w").write("\n".join(lines) + "\n")
        print(pid, "figures", len(a["figs"]), "sentences", len(a["sentences"]))
    open(os.path.join(out, "README.txt"), "w").write(
        "AI gold labels for the 5 CS papers (10 chats: 5 papers x Claude, Gemini)\n\n"
        "1. For each paper open a NEW chat in Claude, attach the paper's PDF (arXiv: https://arxiv.org/pdf/<id>;\n"
        "   professor's paper: tvcg2019_sampling.pdf in this folder), paste the WHOLE prompt_<id>.txt, send.\n"
        "   Use the strongest model with extended thinking on. Repeat in Gemini.\n"
        "2. Save each JSON answer exactly as  <id>_claude.json  /  <id>_gemini.json  (e.g. 2009.02122_claude.json,\n"
        "   prof_tvcg2019_gemini.json). If an answer is cut off, ask it to continue and join the JSON.\n"
        "3a. Quick look: in the Figure Reference Explorer pick the paper, click 'Load results file...' and select its\n"
        "    _claude.json and _gemini.json together. Each AI becomes a method; Gold = sentences both agree on; the\n"
        "    table then shows P/R/F1 of every method against it.\n"
        "3b. Benchmark table: put all 10 files in one folder L and run (CPU, anywhere)\n"
        "    python demo/demo_infer.py report cs_articles.jsonl OUT --labels L        (OUT = folder with scores_*.json)\n"
        "    -> per-paper F1 of every method vs gold, Claude-vs-Gemini agreement, and OUT/results.json for the viewer.\n"
        "    --gold union | claude | gemini  changes how gold is formed (default: both agree).\n")


if __name__ == "__main__":
    main()
