"""Download visualization/graphics papers from arXiv with their LaTeX source.

Usage: python eval/fetch_arxiv.py --n 40 --out eval/arxiv
Layout: <out>/<id>/paper.pdf, <out>/<id>/src/ (extracted source), meta.json
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import tarfile
import time
import urllib.parse
import urllib.request

import feedparser

QUERIES = [
    'cat:cs.GR AND abs:"visualization"',
    'cat:cs.HC AND abs:"visual analytics"',
    'cat:cs.GR AND abs:"volume rendering"',
    'cat:cs.GR AND abs:"scientific visualization"',
]


def fetch(url: str, tries: int = 3) -> bytes:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "figrefs-eval/0.1 (student research)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))
    return b""


def extract_source(blob: bytes, dst: str) -> bool:
    os.makedirs(dst, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
            for m in tf.getmembers():
                if m.isfile() and not m.name.startswith("/") and ".." not in m.name:
                    tf.extract(m, dst, filter="data")
        return True
    except tarfile.TarError:
        pass
    try:  # single gzipped .tex
        data = gzip.decompress(blob)
    except OSError:
        data = blob
    if b"\\documentclass" in data or b"\\begin{document}" in data:
        open(os.path.join(dst, "main.tex"), "wb").write(data)
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="eval/arxiv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ids: list[tuple[str, str]] = []
    per_q = a.n // len(QUERIES) + 4
    for q in QUERIES:
        url = ("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": q, "start": 0, "max_results": per_q, "sortBy": "relevance"}))
        feed = feedparser.parse(fetch(url))
        for e in feed.entries:
            aid = e.id.split("/abs/")[-1]
            if aid not in [i for i, _ in ids]:
                ids.append((aid, e.title.replace("\n", " ")))
        time.sleep(3)
    got = 0
    for aid, title in ids:
        if got >= a.n:
            break
        safe = aid.replace("/", "_")
        d = os.path.join(a.out, safe)
        if os.path.exists(os.path.join(d, "meta.json")):
            got += 1
            continue
        try:
            src = fetch(f"https://arxiv.org/e-print/{aid}")
            if not extract_source(src, os.path.join(d, "src")):
                print("no tex source", aid)
                continue
            pdf = fetch(f"https://arxiv.org/pdf/{aid}")
            if not pdf.startswith(b"%PDF"):
                print("no pdf", aid)
                continue
            open(os.path.join(d, "paper.pdf"), "wb").write(pdf)
            json.dump({"id": aid, "title": title}, open(os.path.join(d, "meta.json"), "w"))
            got += 1
            print("ok", aid, title[:70], flush=True)
        except Exception as e:
            print("fail", aid, e)
        time.sleep(3)


if __name__ == "__main__":
    main()
