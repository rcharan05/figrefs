"""Fetch a deterministic random sample of the GROBID bioRxiv-10k test set (PDF + JATS XML) via HTTP range requests."""
import os, random, sys
from remotezip import RemoteZip
URL = "https://zenodo.org/records/3873702/files/biorxiv-10k-test-2000.zip?download=1"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
out = sys.argv[2] if len(sys.argv) > 2 else "/home/charan/bench/biorxiv"
with RemoteZip(URL) as z:
    ids = sorted({x.split('/')[0] for x in z.namelist() if x.endswith('.pdf')})
    random.Random(230819).shuffle(ids)
    for i, pid in enumerate(ids[:n]):
        d = os.path.join(out, pid)
        os.makedirs(d, exist_ok=True)
        for ext in ("pdf", "xml"):
            dst = os.path.join(d, f"paper.{ext}")
            if not os.path.exists(dst):
                with open(dst, "wb") as f:
                    f.write(z.read(f"{pid}/{pid}.{ext}"))
        if i % 20 == 0:
            print(i, pid, flush=True)
print("done")
