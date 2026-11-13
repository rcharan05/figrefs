"""Run GROBID on every <dir>/*/paper.pdf, caching TEI next to the PDF.

usage: run_grobid.py ROOT PORT TAG [--sentences]
  writes ROOT/<id>/grobid_<TAG>[_sent].tei.xml
"""
import os
import sys
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor


def post(url, pdf, fields):
    b = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"input\"; filename=\"p.pdf\"\r\n"
                 f"Content-Type: application/pdf\r\n\r\n".encode() + open(pdf, "rb").read() + b"\r\n")
    body = b"".join(parts) + f"--{b}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def main():
    root, port, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    sent = "--sentences" in sys.argv
    url = f"http://localhost:{port}/api/processFulltextDocument"
    fields = {"segmentSentences": "1"} if sent else {}
    jobs = []
    for d in sorted(os.listdir(root)):
        pdf = os.path.join(root, d, "paper.pdf")
        out = os.path.join(root, d, f"grobid_{tag}{'_sent' if sent else ''}.tei.xml")
        if os.path.exists(pdf) and not os.path.exists(out):
            jobs.append((d, pdf, out))
    t0 = time.time()

    def one(job):
        d, pdf, out = job
        for attempt in range(3):
            try:
                tei = post(url, pdf, fields)
                open(out, "wb").write(tei)
                return d, "ok"
            except Exception as e:  # 503 = server busy -> retry
                err = e
                time.sleep(5)
        return d, f"fail {err}"

    with ThreadPoolExecutor(2) as ex:
        for i, (d, st) in enumerate(ex.map(one, jobs)):
            print(i, d, st, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
