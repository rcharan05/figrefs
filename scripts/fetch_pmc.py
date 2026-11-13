"""Sample PMC Open Access articles (JATS) from the PMC open-data S3 bucket and store a compact parsed form.

Random listing pages over the modern PMC id range; each article's XML is parsed with eval/figsum/data.load_article and
kept only if it is a usable research article (>= 3 figures with captions, >= 5 sentences citing a figure).
The 13 FigSum+ articles (and anything with the same PMC number) are excluded.
Output: OUT/<pmcid>.json = {sentences, sent_para, top_section(per para), cue(list of fig lists), figures{num: caption}}
usage: python fetch_pmc.py OUT N_ARTICLES [SEED]
"""
import json
import os
import random
import re
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/charan/proj/figrefs/eval/figsum")
from data import load_article  # noqa: E402

B = "https://pmc-oa-opendata.s3.amazonaws.com/"
FIGSUM = {"1064895", "1065313", "1080127", "1087843", "111020", "1134656", "1156890", "115840", "1159166",
          "20459090", "2542994", "2569962", "2891020"}
out, N = sys.argv[1], int(sys.argv[2])
rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
os.makedirs(out, exist_ok=True)


def get(url, t=60):
    return urllib.request.urlopen(url, timeout=t).read()


def one(key):
    pid = key.split("/")[0].split(".")[0][3:]
    if pid in FIGSUM or os.path.exists(os.path.join(out, pid + ".json")):
        return None
    try:
        x = get(B + key, 120)
        with tempfile.NamedTemporaryFile(suffix=".xml") as f:
            f.write(x)
            f.flush()
            a = load_article(pid, f.name)
    except Exception:  # noqa: BLE001
        return False
    figs = {int(k): v for k, v in a.figures.items() if v and len(v) > 20}
    n_cite = sum(1 for c in a.cue if c & set(figs))
    if len(figs) < 3 or n_cite < 5:
        return False
    json.dump({"sentences": a.sentences, "sent_para": a.sent_para, "top_section": a.top_section,
               "cue": [sorted(c) for c in a.cue], "figures": {str(k): v for k, v in figs.items()}},
              open(os.path.join(out, pid + ".json"), "w"))
    return True


have = len(os.listdir(out))
pages = 0
with ThreadPoolExecutor(3) as ex:
    while have < N and pages < 400:
        start = f"PMC{rng.randint(2_000_000, 11_500_000)}"
        try:
            lst = get(f"{B}?list-type=2&max-keys=1000&start-after={start}").decode()
        except Exception:  # noqa: BLE001
            continue
        pages += 1
        keys = [k for k in re.findall(r"<Key>([^<]+)</Key>", lst) if re.fullmatch(r"PMC\d+\.\d+/PMC\d+\.\d+\.xml", k)]
        # keep the latest version per article
        best = {}
        for k in keys:
            pid, v = k.split("/")[0].split(".")
            if pid not in best or int(v) > int(best[pid].split("/")[0].split(".")[1]):
                best[pid] = k
        res = list(ex.map(one, best.values()))
        have = len(os.listdir(out))
        print(f"page {pages} start {start}: ok {sum(r is True for r in res)} / {len(res)}  total {have}", flush=True)
print("done", have)
