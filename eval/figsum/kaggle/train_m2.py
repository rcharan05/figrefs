"""M2 content cross-encoder: train on free-label (caption, sentence) pairs, then score FigSum candidates.

Run on Kaggle (GPU T4/P100, Internet ON) as a single notebook cell, with the dataset folder attached that contains
train.jsonl, valid.jsonl, figsum_pairs.jsonl. Outputs in /kaggle/working:
  m2_<tag>_scores.json   {"art|fig|i": logit} for every FigSum candidate   <- bring this file back
  m2_<tag>_valid.json    valid AUCs (overall and per negative kind) logged during training
  m2_<tag>/              the fine-tuned model (optional to bring back)
Settings via the CONFIG block below. Defaults: microsoft/deberta-v3-base, 1 epoch, max_len 192, batch 32, lr 2e-5.
Time guard: training stops after MAX_TRAIN_MIN minutes, then scoring always runs.
"""
import glob
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

# ----------------------------------------------------------------------------- CONFIG
MODEL = os.environ.get("M2_MODEL", "microsoft/deberta-v3-base")
TAG = os.environ.get("M2_TAG", "deberta")
DATA = os.environ.get("M2_DATA", "")            # folder with train.jsonl; auto-detected under /kaggle/input if empty
OUT = os.environ.get("M2_OUT", "/kaggle/working")
MAX_LEN = int(os.environ.get("M2_MAX_LEN", 192))
BATCH = int(os.environ.get("M2_BATCH", 32))
LR = float(os.environ.get("M2_LR", 2e-5))
EPOCHS = float(os.environ.get("M2_EPOCHS", 1))
MAX_TRAIN_MIN = float(os.environ.get("M2_MAX_TRAIN_MIN", 300))
MAX_TRAIN_EX = int(os.environ.get("M2_MAX_TRAIN_EX", 0))   # 0 = all
EVAL_EVERY = int(os.environ.get("M2_EVAL_EVERY", 1000))
SEED = 230819
# -----------------------------------------------------------------------------


def load_model(path):
    """always float32 master weights (newer transformers load the checkpoint dtype, e.g. fp16, by default)"""
    try:
        m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, dtype=torch.float32)
    except TypeError:
        m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, torch_dtype=torch.float32)
    return m.float()


def find_data():
    if DATA:
        return DATA
    hits = glob.glob("/kaggle/input/**/train.jsonl", recursive=True)
    if not hits:
        raise SystemExit("train.jsonl not found: attach the dataset (Add Input) or set M2_DATA")
    return os.path.dirname(hits[0])


def load(p, limit=0):
    out = [json.loads(l) for l in open(p)]
    return out[:limit] if limit else out


class Pairs(Dataset):
    def __init__(self, recs):
        self.r = recs

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        return self.r[i]


def collate(tok):
    def f(batch):
        enc = tok([b["caption"] for b in batch], [b["sentence"] for b in batch], truncation="longest_first",
                  max_length=MAX_LEN, padding=True, return_tensors="pt")
        if "label" in batch[0]:
            enc["labels"] = torch.tensor([float(b["label"]) for b in batch])
        return enc
    return f


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def score(model, loader, dev):
    model.eval()
    out = []
    for enc in loader:
        enc = {k: v.to(dev) for k, v in enc.items() if k != "labels"}
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out.append(model(**enc).logits.float().squeeze(-1).cpu())
    model.train()
    return torch.cat(out).numpy()


def valid_report(model, va, loader, dev):
    s = score(model, loader, dev)
    y = [r["label"] for r in va]
    kinds = np.array([r["kind"] for r in va])
    rep = {"all": auc(y, s)}
    for k in ("N-other", "N-pre-hard", "N-pre-rand"):
        m = (kinds == "P") | (kinds == k)
        rep[k] = auc(np.array(y)[m], s[m])
    return rep


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    d = find_data()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("data", d, "| device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    tr = load(os.path.join(d, "train.jsonl"))
    random.shuffle(tr)
    if MAX_TRAIN_EX:
        tr = tr[:MAX_TRAIN_EX]
    va = load(os.path.join(d, "valid.jsonl"))
    pairs = load(os.path.join(d, "figsum_pairs.jsonl"))
    print(f"train {len(tr)}  valid {len(va)}  figsum pairs {len(pairs)}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = load_model(MODEL).to(dev)
    col = collate(tok)
    tl = DataLoader(Pairs(tr), batch_size=BATCH, shuffle=True, collate_fn=col, num_workers=2)
    vl = DataLoader(Pairs(va), batch_size=BATCH * 2, shuffle=False, collate_fn=col, num_workers=2)
    pl = DataLoader(Pairs(pairs), batch_size=BATCH * 2, shuffle=False, collate_fn=col, num_workers=2)
    steps = int(math.ceil(len(tl) * EPOCHS))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    log, step, t0, best = [], 0, time.time(), -1.0
    os.makedirs(os.path.join(OUT, f"m2_{TAG}"), exist_ok=True)
    model.train()
    done = False
    while not done:
        for enc in tl:
            enc = {k: v.to(dev) for k, v in enc.items()}
            y = enc.pop("labels")
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                logit = model(**enc).logits.float().squeeze(-1)
            loss = lossf(logit, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            step += 1
            el = (time.time() - t0) / 60
            if step % 100 == 0:
                print(f"step {step}/{steps}  loss {loss.item():.4f}  {el:.1f} min  "
                      f"(eta {el / step * (steps - step):.0f} min)", flush=True)
            if step % EVAL_EVERY == 0 or step >= steps or el > MAX_TRAIN_MIN:
                rep = valid_report(model, va, vl, dev)
                rep.update(step=step, minutes=round(el, 1))
                log.append(rep)
                print("VALID", rep, flush=True)
                if rep["all"] > best:
                    best = rep["all"]
                    model.save_pretrained(os.path.join(OUT, f"m2_{TAG}"))
                    tok.save_pretrained(os.path.join(OUT, f"m2_{TAG}"))
                json.dump(log, open(os.path.join(OUT, f"m2_{TAG}_valid.json"), "w"), indent=1)
            if step >= steps or el > MAX_TRAIN_MIN:
                done = True
                break
    # score FigSum with the best checkpoint
    model = load_model(os.path.join(OUT, f"m2_{TAG}")).to(dev)
    s = score(model, pl, dev)
    json.dump({p["key"]: float(x) for p, x in zip(pairs, s)}, open(os.path.join(OUT, f"m2_{TAG}_scores.json"), "w"))
    print(f"DONE. best valid AUC {best:.3f}. Download {OUT}/m2_{TAG}_scores.json and m2_{TAG}_valid.json")


if __name__ == "__main__":
    main()
