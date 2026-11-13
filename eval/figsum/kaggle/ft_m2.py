"""Idea 2: fine-tune a cross-encoder on FigSum GS1 window candidates, leave-one-article-out (13 folds x 3 seeds).

Kaggle: GPU, Internet ON, dataset attached containing ft_window.jsonl, train_m2.py, train.jsonl, valid.jsonl,
figsum_pairs.jsonl. Starting points:
  m2    the M2 checkpoint (free-label cross-encoder). Looked for in /kaggle/working/m2_deberta or under /kaggle/input;
        if missing, it is retrained first with train_m2.py (~35 min on a T4).
  base  plain microsoft/deberta-v3-base (control: does the M2 starting point matter?)
For each held-out article: train on the other 12 (EPOCHS epochs), score the held-out article's candidates; scores are
averaged over SEEDS seeds. Outputs (/kaggle/working):
  ft_m2_scores.json, ft_base_scores.json   {"art|fig|i": mean logit}   <- bring back
  ft_log.json                              per-fold timing / train loss
"""
import glob
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

OUT = os.environ.get("FT_OUT", "/kaggle/working")
DATA = os.environ.get("FT_DATA", "")
INITS = os.environ.get("FT_INITS", "m2,base").split(",")
SEEDS = [int(s) for s in os.environ.get("FT_SEEDS", "1,2,3").split(",")]
EPOCHS = int(os.environ.get("FT_EPOCHS", 4))
LR = float(os.environ.get("FT_LR", 2e-5))
BATCH = int(os.environ.get("FT_BATCH", 16))
MAX_LEN = int(os.environ.get("FT_MAX_LEN", 320))
FOLDS = int(os.environ.get("FT_FOLDS", 0))  # 0 = all articles; smaller = smoke test
BASE = os.environ.get("FT_BASE", "microsoft/deberta-v3-base")
FILE = os.environ.get("FT_FILE", "ft_window.jsonl")
FAR_NEG = float(os.environ.get("FT_FAR_NEG_FRAC", 1.0))  # fraction of far negatives kept for TRAINING (all are scored)


def find(name):
    if DATA and os.path.exists(os.path.join(DATA, name)):
        return os.path.join(DATA, name)
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None


def load_model(path):
    try:
        m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, dtype=torch.float32)
    except TypeError:
        m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, torch_dtype=torch.float32)
    return m.float()


def m2_checkpoint():
    for c in [os.path.join(OUT, "m2_deberta")] + glob.glob("/kaggle/input/**/m2_deberta", recursive=True):
        if os.path.exists(os.path.join(c, "config.json")):
            return c
    script = find("train_m2.py")
    if not script:
        raise SystemExit("no M2 checkpoint and no train_m2.py to rebuild it")
    print("M2 checkpoint not found -> retraining it (about 35 min on a T4)", flush=True)
    env = dict(os.environ, M2_MODEL=BASE, M2_TAG="deberta", M2_OUT=OUT, M2_DATA=os.path.dirname(script))
    subprocess.run([sys.executable, script], env=env, check=True)
    return os.path.join(OUT, "m2_deberta")


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recs = [json.loads(l) for l in open(find(FILE))]
    arts = sorted({r["art"] for r in recs})
    if FOLDS:
        arts = arts[:FOLDS]
    print(f"examples {len(recs)}  articles {len(arts)}  device {dev}", flush=True)
    starts = {"m2": m2_checkpoint() if "m2" in INITS else None, "base": BASE}
    tok = AutoTokenizer.from_pretrained(BASE)

    def enc(batch):
        e = tok([b["caption"] for b in batch], [b["text"] for b in batch], truncation="longest_first",
                max_length=MAX_LEN, padding=True, return_tensors="pt")
        e["labels"] = torch.tensor([float(b["label"]) for b in batch])
        return e

    log = []
    for init in INITS:
        scores = {}
        t_init = time.time()
        for h in arts:
            rng = random.Random(int(h) % 10_000)
            tr = [r for r in recs if r["art"] != h and not (r.get("far") and r["label"] == 0 and rng.random() > FAR_NEG)]
            te = [r for r in recs if r["art"] == h]
            acc = np.zeros(len(te))
            for seed in SEEDS:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                model = load_model(starts[init]).to(dev)
                tl = DataLoader(tr, batch_size=BATCH, shuffle=True, collate_fn=enc)
                steps = len(tl) * EPOCHS
                opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
                sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
                scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
                pw = torch.tensor((len(tr) - sum(r["label"] for r in tr)) / max(1, sum(r["label"] for r in tr)))
                lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw.to(dev))
                model.train()
                last = 0.0
                for _ in range(EPOCHS):
                    for e in tl:
                        e = {k: v.to(dev) for k, v in e.items()}
                        y = e.pop("labels")
                        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                            lg = model(**e).logits.float().squeeze(-1)
                        loss = lossf(lg, y)
                        opt.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(opt)
                        scaler.update()
                        sch.step()
                        last = loss.item()
                model.eval()
                out = []
                with torch.no_grad():
                    for e in DataLoader(te, batch_size=BATCH * 2, collate_fn=enc):
                        e = {k: v.to(dev) for k, v in e.items() if k != "labels"}
                        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                            out.append(model(**e).logits.float().squeeze(-1).cpu())
                acc += torch.cat(out).numpy() / len(SEEDS)
                log.append({"init": init, "held_out": h, "seed": seed, "last_loss": last,
                            "minutes": round((time.time() - t_init) / 60, 1)})
                del model, opt
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            for r, s in zip(te, acc):
                scores[r["key"]] = float(s)
            print(f"[{init}] held-out {h} done (train {len(tr)}, scored {len(te)})  {(time.time() - t_init) / 60:.1f} min", flush=True)
            json.dump(scores, open(os.path.join(OUT, f"{os.environ.get('FT_TAG', 'ft')}_{init}_scores.json"), "w"))
            json.dump(log, open(os.path.join(OUT, "ft_log.json"), "w"), indent=1)
    tg = os.environ.get("FT_TAG", "ft")
    print("DONE. Download " + ", ".join(f"{OUT}/{tg}_{i}_scores.json" for i in INITS) + f" and {OUT}/ft_log.json")


if __name__ == "__main__":
    main()
