"""Re-download paper.pdf for every eval/arxiv/<id>/ folder (PDFs are not shipped in the zip)."""
import os
import sys
import time
import urllib.request

root = sys.argv[1] if len(sys.argv) > 1 else "eval/arxiv"
for d in sorted(os.listdir(root)):
    dst = os.path.join(root, d, "paper.pdf")
    if os.path.exists(dst):
        continue
    req = urllib.request.Request(f"https://arxiv.org/pdf/{d}", headers={"User-Agent": "figrefs-eval/0.3"})
    try:
        data = urllib.request.urlopen(req, timeout=120).read()
        if data.startswith(b"%PDF"):
            open(dst, "wb").write(data)
            print("ok", d)
        else:
            print("not a pdf", d)
    except Exception as e:
        print("fail", d, e)
    time.sleep(3)
