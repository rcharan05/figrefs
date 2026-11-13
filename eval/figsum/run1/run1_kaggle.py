"""Run 1: whole-paper figure-reference models on FigSum+ GS1 — baseline vs two-stage training with augmentation.

ONE command runs everything and uses every visible GPU (Kaggle 2xT4: two worker processes, one per GPU):
    python run1_kaggle.py launch
Inputs (found under /kaggle/input/** or R1_DATA): run1_data.jsonl (18 articles; split train/test; fold 0-4).

Configs (every sentence of the paper is a candidate; nothing is capped by distance at prediction time):
  C0         phase-7 baseline: (caption, "[tag] Citing: <nearest citation> Candidate: <s>") for every sentence,
             citing sentences included; train keeps all positives + 20% of far negatives; 3 epochs; max_len 320.
  S1         stage 1, figure-sentence MATCHING: (caption, sentence with every figure mention -> "Fig. [X]"),
             no tags, no position; labels = all gold sentences of F (citing + implicit) vs every other
             sentence (A1: all negatives kept). Paraphrase swap p=0.3. max_len 256.
  S2s / S2f  stage 2, IMPLICIT GIVEN EXPLICIT: F's citing sentences E are known (detector) and go straight to the
             output; candidates = every other sentence. Input: (caption, "[position/paragraph/section/cites tag]
             Candidate: <s> Citing: <up to 4 nearest sentences of E>"). max_len 384.
             S2s starts from plain DeBERTa, S2f from the S1 model of the same fold and seed.
             Training augmentations: A2 tag dropout p=0.3 and citing-context dropout p=0.3; detector noise
             (each sentence of E dropped p=0.12 -> it becomes a candidate with its gold label; p=0.05 a random
             non-gold same-paragraph sentence is added to E); A3 citation-deleted copies of F's citing sentences
             as candidates (gold label kept); paraphrase swap p=0.3; far negatives (> 3 from E) kept p=0.3.
  S2f_noA3   S2f without A3 (A3 is the risky augmentation: deleted-citation sentences look like results prose).
Protocol: 5-fold rotation over the 15 training papers x seeds 1,2,3 -> out-of-fold scores (mean logit over
seeds) for every training paper. Then a FINAL model per config on all 15 papers (seed 1; S2f-final from
S1-final), saved, and scored on the 3 sealed test papers. Thresholds are chosen offline (phase8.py).
Outputs (R1_OUT, default /kaggle/working/run1):
  oof/<cfg>.json            {"art|fig|i": mean logit}                     <- bring back
  test_SEALED/<cfg>.json    final-model scores on the 3 test papers       <- bring back, do not open
  models/<cfg>/             safetensors fp32 + tokenizer + input_format.json   (about 0.74 GB each)
  jobs/*.json, log.txt      per-job timing, losses, example counts
Resumable: re-running `launch` skips finished jobs (if the output folder survived).
Env: R1_SEEDS=1,2,3  R1_CONFIGS=C0,S1,S2s,S2f,S2f_noA3  R1_EPOCHS=3  R1_BASE=microsoft/deberta-v3-base
     R1_SMOKE=1 (tiny CPU check: 2 folds, 1 seed, 1 epoch, 40 sentences per article)
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time

import numpy as np

OUT = os.environ.get("R1_OUT", "/kaggle/working/run1")
TMP = os.environ.get("R1_TMP", os.path.join(OUT, "_ckpt"))    # stage-1 fold checkpoints (deleted once used)
SMOKE = os.environ.get("R1_SMOKE") == "1"
SEEDS = [int(s) for s in os.environ.get("R1_SEEDS", "1" if SMOKE else "1,2,3").split(",")]
CONFIGS = os.environ.get("R1_CONFIGS", "C0,S1,S2s,S2f,S2f_noA3").split(",")
EPOCHS = int(os.environ.get("R1_EPOCHS", 1 if SMOKE else 3))
BASE = os.environ.get("R1_BASE", "microsoft/deberta-v3-base")
LR, WD, BATCH, WARM = 2e-5, 0.01, 16, 0.1
MAXLEN = {"C0": 320, "S1": 256, "S2": 384}
NFOLD = 2 if SMOKE else 5
STAGE = {"C0": "C0", "S1": "S1", "S2s": "S2", "S2f": "S2", "S2f_noA3": "S2"}
INIT = {"S2f": "S1", "S2f_noA3": "S1"}           # config -> config whose same-fold/seed model it starts from
P_SWAP, P_TAGDROP, P_CTXDROP, P_EDROP, P_EADD = 0.3, 0.3, 0.3, 0.12, 0.05
FAR_KEEP = {"C0": 0.2, "S1": 1.0, "S2": 0.3}
TAB = re.compile(r"\btables?\b", re.I)
SUP = re.compile(r"supplement|additional file|figure s\d|fig\.? ?s\d|data not shown|not shown", re.I)
CAPPFX = re.compile(r"^\s*(?:Fig(?:ure)?\.?|FIG\.?)\s*\d+\s*[.:|]?\s*", re.I)
VERSION = "run1-v1"


# ------------------------------------------------------------------ data
def find(name):
    d = os.environ.get("R1_DATA", "")
    if d and os.path.exists(os.path.join(d, name)):
        return os.path.join(d, name)
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if hits:
        return hits[0]
    if os.path.exists(name):
        return name
    raise SystemExit(f"cannot find {name} (attach the dataset or set R1_DATA)")


def load_articles():
    arts = [json.loads(l) for l in open(find("run1_data.jsonl"))]
    for a in arts:
        a["cue"] = [set(c) for c in a["cue"]]
        a["gold"] = {int(f): set(v) for f, v in a["gold"].items()}
        a["captions"] = {int(k): CAPPFX.sub("", v).strip()[:700] for k, v in a["captions"].items()}   # as phase 7
        if SMOKE:
            n = min(40, len(a["sentences"]))
            for k in ("sentences", "masked", "sent_para", "cue"):
                a[k] = a[k][:n]
            a["gold"] = {f: {i for i in g if i < n} for f, g in a["gold"].items()}
            a["figs"] = [f for f in a["figs"] if a["gold"][f] and any(f in c for c in a["cue"])][:2]
    return {a["id"]: a for a in arts}


def _others(a, i, f):
    c = [f"Figure {g}" for g in sorted(a["cue"][i]) if g != f]
    s = a["sentences"][i]
    if TAB.search(s):
        c.append("table")
    if SUP.search(s):
        c.append("supplementary/not shown")
    return ", ".join(c) if c else "none"


def _sec(a, i):
    return (a["para_section"][a["sent_para"][i]] or "")[:40]


def _pos(a, i, E):
    """(position text, nearest index of E or None, distance)."""
    if not E:
        return "no citation of this figure in the paper", None, 10 ** 6
    m = min(E, key=lambda j: (abs(j - i), j))
    d = abs(i - m)
    return f"{d} sentence{'s' if d > 1 else ''} {'after' if i > m else 'before'} a citation of this figure", m, d


def examples(a, f, cfg, rng=None):
    """List of (text_a, text_b, label, key). rng=None -> evaluation (deterministic, no augmentation)."""
    st, train = STAGE[cfg], rng is not None
    S, gold, cap = a["sentences"], a["gold"].get(f, set()), a["captions"].get(f, "")
    n = len(S)
    cit = [i for i in range(n) if f in a["cue"][i]]
    out = []
    key = lambda i: f"{a['id']}|{f}|{i}"
    swap = lambda i, txt, masked: ((a["bt_masked"] if masked else a["bt"]).get(str(i), txt)
                                   if train and rng.random() < P_SWAP else txt)
    if st == "C0":
        for i in range(n):
            y = int(i in gold)
            cites = f in a["cue"][i]
            pos, m, d = _pos(a, i, cit)
            if train and not cites and d > 3 and y == 0 and rng.random() > FAR_KEEP["C0"]:
                continue
            pos_txt = "this is a citation" if cites else pos
            par = "same" if (m is not None and a["sent_para"][i] == a["sent_para"][m]) else "other"
            tag = (f"[cites this figure: {'yes' if cites else 'no'} | {pos_txt} | {par} paragraph | "
                   f"section: {_sec(a, i)} | cites: {_others(a, i, f)}]")
            ctx = "(this sentence)" if cites else (S[m] if m is not None else "(none)")
            out.append((cap, f"{tag} Citing: {ctx} Candidate: {S[i]}", y, key(i)))
        return out
    if st == "S1":
        for i in range(n):
            out.append((cap, swap(i, a["masked"][i], True), int(i in gold), key(i)))
        return out
    # ---- stage 2: implicit given explicit
    E = set(cit)
    if train:
        E = {c for c in cit if rng.random() >= P_EDROP}
        if rng.random() < P_EADD:
            paras = {a["sent_para"][c] for c in cit}
            pool = [j for j in range(n) if a["sent_para"][j] in paras and j not in gold and j not in E]
            if pool:
                E.add(rng.choice(pool))

    def b_text(i, cand, Ex):
        pos, m, d = _pos(a, i, Ex)
        par = "same" if (m is not None and a["sent_para"][i] == a["sent_para"][m]) else "other"
        tag = f"[{pos} | {par} paragraph | section: {_sec(a, i)} | cites: {_others(a, i, f)}]"
        near = sorted(Ex, key=lambda j: (abs(j - i), j))[:4]
        ctx = " | ".join(S[j] for j in near) if near else "(none)"
        if train and rng.random() < P_TAGDROP:
            tag = "[position unknown]"
        if train and rng.random() < P_CTXDROP:
            ctx = "(none)"
        return f"{tag} Candidate: {cand} Citing: {ctx}", d

    for i in range(n):
        if i in E:
            continue
        y = int(i in gold)
        txt, d = b_text(i, swap(i, S[i], False), E)
        if train and d > 3 and y == 0 and rng.random() > FAR_KEEP["S2"]:
            continue
        out.append((cap, txt, y, key(i)))
    if train and cfg != "S2f_noA3":
        for c in cit:
            t = a["cdel"].get(f"{c}|{f}")
            if t:
                txt, _ = b_text(c, t, E - {c})
                out.append((cap, txt, int(c in gold), key(c) + "|A3"))
    return out


def build(arts, ids, cfg, rng=None):
    ex = []
    for pid in ids:
        a = arts[pid]
        for f in a["figs"]:
            ex += examples(a, f, cfg, rng)
    return ex


# ------------------------------------------------------------------ model
def load_model(path):
    import torch
    from transformers import AutoModelForSequenceClassification
    try:
        m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, dtype=torch.float32)
    except TypeError:
        m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, torch_dtype=torch.float32)
    return m.float()


def batches(tok, ex, maxlen, rng=None):
    """Length-grouped batches (shuffled chunks of 50 batches when training) to cut padding."""
    import torch
    idx = list(range(len(ex)))
    if rng is not None:
        rng.shuffle(idx)
        chunk = BATCH * 50
        idx = [j for k in range(0, len(idx), chunk) for j in sorted(idx[k:k + chunk], key=lambda j: len(ex[j][1]))]
        groups = [idx[k:k + BATCH] for k in range(0, len(idx), BATCH)]
        rng.shuffle(groups)
    else:
        idx.sort(key=lambda j: len(ex[j][0]) + len(ex[j][1]))
        groups = [idx[k:k + BATCH * 2] for k in range(0, len(idx), BATCH * 2)]
    for g in groups:
        e = tok([ex[j][0] for j in g], [ex[j][1] for j in g], truncation="longest_first", max_length=maxlen,
                padding=True, return_tensors="pt")
        e["labels"] = torch.tensor([float(ex[j][2]) for j in g])
        yield g, e


def train(arts, ids, cfg, seed, init, log):
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    maxlen = MAXLEN[STAGE[cfg]]
    epochs = [build(arts, ids, cfg, random.Random(seed * 1000 + ep)) for ep in range(EPOCHS)]
    npos = sum(e[2] for e in epochs[0])
    pw = torch.tensor((len(epochs[0]) - npos) / max(1, npos))
    steps = sum(math.ceil(len(e) / BATCH) for e in epochs)
    model = load_model(init).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = get_linear_schedule_with_warmup(opt, int(WARM * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw.to(dev))
    model.train()
    losses = []
    for ep, ex in enumerate(epochs):
        tot, k = 0.0, 0
        for _, e in batches(tok, ex, maxlen, random.Random(seed * 7 + ep)):
            e = {kk: v.to(dev) for kk, v in e.items()}
            y = e.pop("labels")
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                lg = model(**e).logits.float().squeeze(-1)
            loss = lossf(lg, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss in {cfg} seed {seed}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            tot, k = tot + loss.item(), k + 1
        losses.append(round(tot / max(1, k), 4))
    log.update({"n_train": [len(e) for e in epochs], "n_pos": npos, "pos_weight": round(float(pw), 3),
                "steps": steps, "epoch_loss": losses})
    return model, tok


def score(model, tok, arts, ids, cfg):
    import torch
    dev = next(model.parameters()).device
    ex = build(arts, ids, cfg, None)
    out = {}
    model.eval()
    with torch.no_grad():
        for g, e in batches(tok, ex, MAXLEN[STAGE[cfg]]):
            e = {k: v.to(dev) for k, v in e.items() if k != "labels"}
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                lg = model(**e).logits.float().squeeze(-1).cpu().numpy()
            for j, v in zip(g, np.atleast_1d(lg)):
                out[ex[j][3]] = float(v)
    model.train()
    return out


def save_model(model, tok, path, cfg):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    json.dump({"version": VERSION, "config": cfg, "stage": STAGE[cfg], "max_len": MAXLEN[STAGE[cfg]], "base": BASE,
               "input": __doc__.split("Configs")[1].split("Protocol")[0].strip(),
               "decision": "per-article z-score of the logit >= threshold (threshold from phase8.py)"},
              open(os.path.join(path, "input_format.json"), "w"), indent=1)


# ------------------------------------------------------------------ jobs
def job_list(arts):
    train_ids = sorted(p for p, a in arts.items() if a["split"] == "train")
    folds = sorted({arts[p]["fold"] for p in train_ids})[:NFOLD]
    jobs = []
    order = [c for c in ("S1", "S2f", "S2f_noA3", "C0", "S2s") if c in CONFIGS]
    chained = [c for c in order if c == "S1" or c in INIT]      # S1 then its stage-2 jobs, fold by fold: few live
    for group in (chained, [c for c in order if c not in chained]):   # checkpoints on disk at any time
        for seed in SEEDS:
            for k in folds:
                for cfg in group:
                    dep = [f"{INIT[cfg]}-s{seed}-f{k}"] if cfg in INIT else []
                    jobs.append({"id": f"{cfg}-s{seed}-f{k}", "cfg": cfg, "seed": seed, "fold": k, "deps": dep,
                                 "train": [p for p in train_ids if arts[p]["fold"] != k],
                                 "score": [p for p in train_ids if arts[p]["fold"] == k]})
    for cfg in order:     # finals: all training papers, seed 1, scored on the sealed test papers
        dep = [f"{INIT[cfg]}-final"] if cfg in INIT else []
        jobs.append({"id": f"{cfg}-final", "cfg": cfg, "seed": SEEDS[0], "fold": None, "deps": dep,
                     "train": train_ids, "score": sorted(p for p, a in arts.items() if a["split"] == "test")})
    return jobs


def done(jid):
    return os.path.exists(os.path.join(OUT, "jobs", jid + ".json"))


def run_job(job, arts, gpu):
    t0 = time.time()
    log = {"id": job["id"], "gpu": gpu, "train_papers": job["train"], "scored_papers": job["score"]}
    init = BASE
    if job["cfg"] in INIT:
        init = os.path.join(TMP, job["deps"][0]) if job["fold"] is not None else os.path.join(OUT, "models", INIT[job["cfg"]])
    model, tok = train(arts, job["train"], job["cfg"], job["seed"], init, log)
    log["scores"] = score(model, tok, arts, job["score"], job["cfg"])
    if job["fold"] is None:
        save_model(model, tok, os.path.join(OUT, "models", job["cfg"]), job["cfg"])
        os.makedirs(os.path.join(OUT, "test_SEALED"), exist_ok=True)
        json.dump(log["scores"], open(os.path.join(OUT, "test_SEALED", job["cfg"] + ".json"), "w"))
    elif job["cfg"] in INIT.values():                       # keep the stage-1 fold model for its stage-2 jobs
        model.save_pretrained(os.path.join(TMP, job["id"]), safe_serialization=True)
    log["minutes"] = round((time.time() - t0) / 60, 1)
    tmp = os.path.join(OUT, "jobs", job["id"] + ".json.tmp")
    json.dump(log, open(tmp, "w"))
    os.replace(tmp, os.path.join(OUT, "jobs", job["id"] + ".json"))
    # stage-1 fold checkpoint no longer needed once every job that starts from it is done
    for d in glob.glob(os.path.join(TMP, "*")):
        users = [j for j in job_list(arts) if j["deps"] == [os.path.basename(d)]]
        if users and all(done(j["id"]) for j in users):
            shutil.rmtree(d, ignore_errors=True)
    return log


def worker(gpu):
    import torch  # noqa: F401
    arts = load_articles()
    jobs = job_list(arts)
    logf = open(os.path.join(OUT, "log.txt"), "a")
    while True:
        todo = [j for j in jobs if not done(j["id"])]
        if not todo:
            return
        ready = [j for j in todo if all(done(d) for d in j["deps"])]
        claimed = None
        for j in ready:
            try:
                os.mkdir(os.path.join(OUT, "claims", j["id"]))
                claimed = j
                break
            except FileExistsError:
                continue
        if claimed is None:
            if all(os.path.isdir(os.path.join(OUT, "claims", j["id"])) for j in todo if all(done(d) for d in j["deps"])) \
                    and not any(not os.path.isdir(os.path.join(OUT, "claims", j["id"])) for j in todo):
                return                                     # everything left is being run by the other worker
            time.sleep(20)
            continue
        msg = f"[gpu {gpu}] start {claimed['id']}  train {len(claimed['train'])} papers  {time.strftime('%H:%M:%S')}"
        print(msg, flush=True), logf.write(msg + "\n"), logf.flush()
        try:
            lg = run_job(claimed, arts, gpu)
        except Exception as ex:                             # release the claim so a re-launch retries it
            shutil.rmtree(os.path.join(OUT, "claims", claimed["id"]), ignore_errors=True)
            logf.write(f"[gpu {gpu}] FAILED {claimed['id']}: {ex!r}\n"), logf.flush()
            raise
        msg = (f"[gpu {gpu}] done  {claimed['id']}  {lg['minutes']} min  n_train {lg['n_train']}  "
               f"loss {lg['epoch_loss']}  ({sum(done(j['id']) for j in jobs)}/{len(jobs)} jobs)")
        print(msg, flush=True), logf.write(msg + "\n"), logf.flush()


def merge():
    arts = load_articles()
    jobs = job_list(arts)
    os.makedirs(os.path.join(OUT, "oof"), exist_ok=True)
    for cfg in CONFIGS:
        acc = {}
        for j in jobs:
            if j["cfg"] == cfg and j["fold"] is not None and done(j["id"]):
                for k, v in json.load(open(os.path.join(OUT, "jobs", j["id"] + ".json")))["scores"].items():
                    acc.setdefault(k, []).append(v)
        if acc:
            json.dump({k: float(np.mean(v)) for k, v in acc.items()}, open(os.path.join(OUT, "oof", cfg + ".json"), "w"))
            print(f"oof/{cfg}.json  {len(acc)} pairs  seeds per pair: {sorted({len(v) for v in acc.values()})}")
    left = [j["id"] for j in jobs if not done(j["id"])]
    print("unfinished jobs:", left if left else "none")
    import zipfile
    zp = os.path.join(os.path.dirname(OUT.rstrip("/")) or ".", "run1_scores.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("oof", "test_SEALED", "jobs"):
            for fn in glob.glob(os.path.join(OUT, sub, "*.json")):
                z.write(fn, os.path.join(sub, os.path.basename(fn)))
        if os.path.exists(os.path.join(OUT, "log.txt")):
            z.write(os.path.join(OUT, "log.txt"), "log.txt")
    print(f"DOWNLOAD {zp} (scores + logs, a few MB). Weights: {OUT}/models/<cfg>/ (about 0.74 GB each) stay in the "
          f"notebook output; 'New Dataset' from the output keeps them on Kaggle.")


def launch():
    for d in ("jobs", "claims"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    for c in glob.glob(os.path.join(OUT, "claims", "*")):   # stale claims from a previous session
        if not done(os.path.basename(c)):
            shutil.rmtree(c, ignore_errors=True)
    arts = load_articles()
    jobs = job_list(arts)
    print(f"{len(jobs)} jobs ({sum(done(j['id']) for j in jobs)} already done); configs {CONFIGS}; seeds {SEEDS}; "
          f"epochs {EPOCHS}; base {BASE}", flush=True)
    for cfg in CONFIGS:
        tr = [p for p, a in arts.items() if a["split"] == "train"]
        ex = build(arts, tr, cfg, random.Random(0))
        print(f"  {cfg:9s} training examples on all 15 papers (one epoch): {len(ex)}  positives {sum(e[2] for e in ex)}")
    try:
        import torch
        ngpu = torch.cuda.device_count()
    except Exception:
        ngpu = 0
    procs = []
    for g in range(max(1, ngpu)):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g) if ngpu else "")
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker", str(g)], env=env))
    codes = [p.wait() for p in procs]
    merge()
    if any(codes):
        raise SystemExit(f"a worker failed (exit codes {codes}); see {OUT}/log.txt, then re-run launch to resume")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "launch"
    if cmd == "worker":
        worker(int(sys.argv[2]))
    elif cmd == "merge":
        merge()
    else:
        launch()
