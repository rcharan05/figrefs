"""Zero-shot LLM judge for implicit figure references (Qwen2.5-1.5B-Instruct, bf16, CPU).

For a (figure F, candidate sentence) pair the model sees F's caption and a short passage, and answers
"Does sentence [k] describe, explain or interpret the experiment or results shown in Figure F?"
score = logit("Yes") - logit("No") at the first answer token.
  window candidates (within 3 sentences of a mention of F): passage = the sentences spanning the candidate and its
      nearest mention of F, plus one sentence of context either side
  far candidates: the top FAR_N sentences per paper by the 'mix' similarity score (each sentence with its best figure);
      passage = up to 3 sentences that cite F, then the candidate on its own
Cached to llm_scores.json ("art|fig|i" -> score), resumable.  usage: python llm_judge.py [window|far]...
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase2 as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
PROMPT = os.environ.get("JUDGE_PROMPT", "v1")
OUT = os.path.join(HERE, os.environ.get("JUDGE_OUT", "llm_scores.json"))
FAR_N = 20
SYS = "You are an expert scientific annotator. Answer only Yes or No."


def clip(s, n):
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " ..."


def prompt_window(a, f, i):
    ms = a._ment[f]
    m = min(ms, key=lambda j: abs(j - i))
    lo, hi = max(0, min(i, m) - 1), min(len(a.sentences) - 1, max(i, m) + 1)
    lines, k = [], None
    for n, j in enumerate(range(lo, hi + 1), 1):
        lines.append(f"[{n}] {clip(a.sentences[j], 350)}")
        if j == i:
            k = n
    return (f"Figure {f} caption: {clip(a.figures[f], 600)}\n\nPassage from the paper:\n" + "\n".join(lines)
            + (f"\n\nQuestion: Does sentence [{k}] describe, explain or interpret the experiment or the results shown in "
               f"Figure {f}, even though it does not name the figure? Answer Yes or No." if PROMPT == "v1" else
               f"\n\nA sentence belongs to Figure {f} if it is part of the discussion of what Figure {f} shows: the "
               f"question or purpose of that experiment, how that experiment was set up or measured, its results, or the "
               f"direct conclusion drawn from them. It does not belong if it is general background, describes a different "
               f"experiment, figure or table, or only concerns methods details used throughout the paper.\n"
               f"Question: Does sentence [{k}] belong to Figure {f}? Answer Yes or No."))


def prompt_far(a, f, i):
    cites = "\n".join(f"- {clip(a.sentences[j], 300)}" for j in a._ment[f][:3])
    return (f"Figure {f} caption: {clip(a.figures[f], 600)}\n\nSentences in the paper that cite Figure {f}:\n{cites}\n\n"
            f"Candidate sentence (from elsewhere in the paper): {clip(a.sentences[i], 350)}\n\n"
            f"Question: Does the candidate sentence describe, explain or interpret the experiment or the results shown "
            f"in Figure {f}, even though it does not name the figure? Answer Yes or No.")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(2)
    D = P.load()
    arts = D.split["dev"] + D.split["test"]
    jobs = []
    if "window" in sys.argv[1:]:
        for aid in arts:
            for k in D.by_art[aid]:
                r = D.rows[k]
                if r["d_sent"] <= P.WIN and r["fig"] in D.figs[aid]:
                    jobs.append(("w", aid, r["fig"], r["i"]))
    if "far" in sys.argv[1:]:
        for aid in arts:
            sc = P.plain_scores(D, [aid], "mix")
            best = {}
            for (x, f, i), s in sc.items():
                if f in D.figs[aid] and (i not in best or s > best[i][0]):
                    best[i] = (s, f)
            for s, f, i in sorted(((s, f, i) for i, (s, f) in best.items()), reverse=True)[:FAR_N]:
                jobs.append(("f", aid, f, i))
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    yes, no = tok.encode("Yes")[0], tok.encode("No")[0]
    t0, done = time.time(), 0
    for kind, aid, f, i in jobs:
        key = f"{aid}|{f}|{i}"
        if key in res:
            continue
        a = D.A[aid]
        user = prompt_window(a, f, i) if kind == "w" else prompt_far(a, f, i)
        kw = {"enable_thinking": False} if "Qwen3" in MODEL else {}
        text = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True, **kw)
        ids = tok(text, return_tensors="pt").input_ids
        with torch.no_grad():
            lg = model(ids).logits[0, -1].float()
        res[key] = float(lg[yes] - lg[no])
        done += 1
        if done % 25 == 0:
            json.dump(res, open(OUT, "w"))
            print(f"{done}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(OUT, "w"))
    print("finished", len(res), flush=True)


if __name__ == "__main__":
    main()
