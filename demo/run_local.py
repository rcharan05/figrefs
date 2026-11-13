"""Offline demo on a laptop (no GPU needed; uses the Apple-silicon GPU if present): any paper in, viewer out.

  python demo/run_local.py PAPER [PAPER ...] [--models DIR] [--only SPAN_ES,C0L] [--out DIR] [--no-open]

PAPER: a .pdf, a JATS .xml, a LaTeX source folder, or an articles .jsonl (e.g. cs_articles.jsonl).
--models DIR: folder holding the downloaded model folders (each has input_format.json), default ./models.
--only: score only these models (SPAN_ES alone is the fastest good choice for a live demo).
Steps: paper -> sentences/figures -> each model scores every sentence for every figure -> FigSum+ Paragraph, B0f,
models and ensembles with the evaluated cut-offs -> OUT/explorer.html (opens in the browser).
Without models it still runs Paragraph and B0f.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        raise SystemExit(__doc__)

    def opt(name, default=None):
        if name in args:
            i = args.index(name)
            v = args[i + 1]
            del args[i:i + 2]
            return v
        return default

    models = os.path.abspath(opt("--models", "models"))
    only = opt("--only")
    out = os.path.abspath(opt("--out", "demo_out"))
    no_open = "--no-open" in args
    papers = [a for a in args if a != "--no-open"]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    # 1. paper(s) -> articles.jsonl
    arts = os.path.join(out, "articles.jsonl")
    specs = []
    with open(arts, "w") as fh:
        for p in papers:
            if p.endswith(".jsonl"):
                fh.write(open(p).read().rstrip("\n") + "\n")
            else:
                pid = os.path.splitext(os.path.basename(os.path.normpath(p)))[0]
                specs.append(f"{pid}={p}")
    if specs:
        tmp = os.path.join(out, "_new.jsonl")
        subprocess.run([sys.executable, os.path.join(HERE, "article_from_source.py"), tmp] + specs, check=True)
        with open(arts, "a") as fh:
            fh.write(open(tmp).read())
    print(f"[1/3] papers parsed ({time.time() - t0:.0f}s)")

    # 2. models
    import demo_infer as DI
    for fp in [f for f in os.listdir(out) if f.startswith("scores_")]:
        os.remove(os.path.join(out, fp))
    if os.path.isdir(models):
        os.environ["DEMO_MODELS"] = models
        DI.cmd_score(arts, out, set(only.split(",")) if only else None)
    else:
        print(f"no model folder at {models}: running the rule methods only")
    print(f"[2/3] models done ({time.time() - t0:.0f}s)")

    # 3. report + viewer
    DI.cmd_report(arts, out)
    base = os.path.join(HERE, "viewer_data.json")
    if not os.path.exists(base):
        base = os.path.join(out, "_empty.json")
        json.dump({"papers": [], "labels": {}, "order": []}, open(base, "w"))
    html = os.path.join(out, "explorer.html")
    subprocess.run([sys.executable, os.path.join(HERE, "assemble_viewer.py"), base, html, os.path.join(out, "results.json")],
                   check=True)
    print(f"[3/3] viewer: {html}  (total {time.time() - t0:.0f}s)")
    if not no_open:
        webbrowser.open("file://" + html)


if __name__ == "__main__":
    main()
