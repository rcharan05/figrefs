"""Download PDF + JATS XML for the FigSum+ articles from the PMC open-data S3 bucket."""
import csv, io, os, re, urllib.request
root = "/home/charan/bench/figsum"
raw = open("/home/charan/figsum_data/f_1293142.csv", "rb").read().decode("latin1").replace("\r", "\n")
ids = [r[0] for r in list(csv.reader(io.StringIO(raw)))[1:] if r and r[0].strip()]
B = "https://pmc-oa-opendata.s3.amazonaws.com/"
for i in ids:
    lst = urllib.request.urlopen(f"{B}?list-type=2&prefix=PMC{i}.", timeout=60).read().decode()
    keys = re.findall(r"<Key>([^<]+)</Key>", lst)
    vers = sorted({k.split("/")[0] for k in keys}, key=lambda v: int(v.split(".")[1]))
    if not vers:
        print("missing", i); continue
    v = vers[0]  # earliest version = closest to what the 2015 annotators saw
    d = os.path.join(root, i); os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "xml"):
        k = f"{v}/{v}.{ext}"
        if k in keys:
            open(os.path.join(d, f"paper.{ext}"), "wb").write(urllib.request.urlopen(B + k, timeout=120).read())
    print(i, v, "versions:", vers, sorted(os.listdir(d)))
