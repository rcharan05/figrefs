"""Build the single-file viewer: viewer_template.html + viewer_data.json (+ any results.json files to pre-load).
usage: python demo/assemble_viewer.py viewer_data.json OUT.html [results.json ...]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(sys.argv[1]))
for fp in sys.argv[3:]:
    R = json.load(open(fp))
    have = {p["id"]: p for p in D["papers"]}
    for p in R["papers"]:
        if p["id"] in have:
            have[p["id"]]["methods"].update(p["methods"])
        else:
            D["papers"].insert(0, p)
    for k, v in R.get("labels", {}).items():
        D["labels"].setdefault(k, v)
        if k not in D["order"]:
            D["order"].append(k)
blob = json.dumps(D, ensure_ascii=False).replace("</", "<\\/")
html = open(os.path.join(HERE, "viewer_template.html")).read().replace("__DATA__", blob)
open(sys.argv[2], "w").write(html)
print(len(D["papers"]), "papers,", round(len(html) / 1e6, 1), "MB ->", sys.argv[2])
