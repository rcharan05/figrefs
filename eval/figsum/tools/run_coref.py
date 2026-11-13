"""Maverick coreference over every paragraph of the FigSum articles.
Each paragraph is fed as the article's own sentences joined by a space, so mention spans map exactly to sentence ids.
Output: {article: [cluster, ...]}, cluster = [[sentence_index, mention_text], ...] (only clusters spanning >= 2 sentences
are useful later, but all are kept).
usage: python run_coref.py OUT.json [article_id ...]
"""
import json
import os
import sys
import time

sys.path.insert(0, "/home/charan/proj/figrefs/eval/figsum")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import mav_load  # noqa: E402
from data import load_all  # noqa: E402

out_path = sys.argv[1]
only = set(sys.argv[2:])
arts, _ = load_all("/home/charan/proj/figrefs/eval/figsum/gold", "/home/charan/bench/figsum")
m = mav_load.load()
m.model.float()
res = json.load(open(out_path)) if os.path.exists(out_path) else {}
for a in arts:
    if (only and a.id not in only) or a.id in res:
        continue
    t0 = time.time()
    by_para = {}
    for i, p in enumerate(a.sent_para):
        by_para.setdefault(p, []).append(i)
    clusters = []
    for p, ids in by_para.items():
        if len(ids) < 2:
            continue
        starts, text = [], ""
        for i in ids:
            starts.append(len(text))
            text += a.sentences[i] + " "
        try:
            with torch.no_grad():
                o = m.predict(text.strip())
        except Exception as e:  # noqa: BLE001
            print("fail", a.id, p, e, flush=True)
            continue
        for cl in o["clusters_char_offsets"]:
            mm = []
            for (s, e) in cl:
                k = max(j for j, st in enumerate(starts) if st <= s)
                mm.append([ids[k], text[s:e + 1]])
            clusters.append(mm)
    res[a.id] = clusters
    json.dump(res, open(out_path, "w"))
    print(a.id, len(clusters), f"{time.time() - t0:.0f}s", flush=True)
