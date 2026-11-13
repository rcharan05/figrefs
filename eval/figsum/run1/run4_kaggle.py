"""Run 4: span tagger with EARLY STOPPING, without (SPAN_ES) and with (SPAN_VIS) text from the figure image.

One script; each Kaggle account runs ONE config (2xT4):
    python run4_kaggle.py launch SPAN_ES     |     python run4_kaggle.py launch SPAN_VIS
Inputs: run1_data.jsonl; for SPAN_VIS also figtext.py and the figimg/ folder (cropped figures + index.json).
Output: /kaggle/working/run4_<cfg>/ and run4_<cfg>_scores.zip.

Model = run 3's SPAN (ModernBERT-large span tagger: "[CLS] Figure F. <caption> [SEP]" + 2,048-token windows tiling
the whole paper, one [MASK] marker per sentence, A3 + paraphrase swap, pos_weight <= 8, batch 8, lr 2e-5).
Early stopping (both configs), deliberately conservative: validation = the 3 former test papers (17483019,
19673075, 23041342; never trained on). Up to 8 epochs, linear decay over 8 epochs; every half epoch the model
scores the validation papers; metric = average precision over (figure, sentence) pairs (threshold-free), smoothed
over the last two checks. Never stops before epoch 3; stops only after 4 checks (2 epochs) without a new smoothed
best; then restores the best weights. Validation loss is logged too. The held-out fold of the 5-fold rotation is never looked
at during training, so out-of-fold scores stay honest; validation papers are used only for stopping.
SPAN_VIS: before training, Qwen2-VL-2B-Instruct (figtext.py, fp32, one shard per GPU) writes for every cropped
figure (a) the text inside it (OCR) and (b) a literal visual description; the header becomes
"Figure F. <caption> Figure text: <ocr> Figure shows: <description>" (up to 400 tokens instead of 200). Figures with
no crop get "Figure shows: (no image available)".
Protocol: 5 folds x seeds 1,2,3 on the 15 training papers -> oof/<cfg>.json; final model on all 15 (early-stopped
on the validation papers) -> models/<cfg>/ and val/<cfg>.json (its scores on the validation papers).
Robust: failed jobs are logged and skipped; re-running launch resumes.
Env: R1_SEEDS R1_OUT; R1_SMOKE=1 quick check (tiny models, 2 folds, 1 seed, 2 epochs, 40 sentences);
     R1_SMOKE_REAL=1 with real models; R1_FIGTEXT_LIMIT=n caps figures for figtext in smoke tests.
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

if len(sys.argv) > 2 and sys.argv[1] == "launch":      # `launch <cfg>` selects the experiment for this account
    os.environ["R1_CONFIGS"] = sys.argv[2]
    os.environ.setdefault("R1_OUT", ("/kaggle/working" if os.path.isdir("/kaggle/working") else ".") + f"/run4_{sys.argv[2]}")

OUT = os.environ.get("R1_OUT", "/kaggle/working/run4")
TMP = os.environ.get("R1_TMP", os.path.join(OUT, "_ckpt"))    # stage-1 fold checkpoints (deleted once used)
SMOKE = os.environ.get("R1_SMOKE") == "1"
SMOKE_REAL = os.environ.get("R1_SMOKE_REAL") == "1"      # smoke limits, but the real large models / 2,048 windows
SEEDS = [int(s) for s in os.environ.get("R1_SEEDS", "1" if SMOKE else "1,2,3").split(",")]
CONFIGS = os.environ.get("R1_CONFIGS", "SPAN_ES").split(",")
EPOCHS = int(os.environ.get("R1_EPOCHS", 1 if SMOKE else 3))
BASE = os.environ.get("R1_BASE", "microsoft/deberta-v3-base")
LR, WD, BATCH, WARM = 2e-5, 0.01, 16, 0.1
MAXLEN = {"C0": 320, "S1": 256, "S2": 384, "S1b": 256, "S2b": 320}
LR_STAGE = {"S1b": 1e-5}                       # everything else LR
LARGE = ("C0L", "C0aL", "C0Lctx")
LR_CFG = {c: 1e-5 for c in LARGE}
BASE_CFG = {c: "microsoft/deberta-v3-large" for c in LARGE} if (not SMOKE or SMOKE_REAL) else {}
MICRO = {c: 4 for c in LARGE}
GRAD_CKPT = set(LARGE)
EVAL_BS = {c: 16 for c in LARGE}
MAXLEN_CFG = {"C0Lctx": 448}
SPAN_BASE = os.environ.get("R1_SPAN_BASE", "hf-internal-testing/tiny-random-ModernBertModel" if (SMOKE and not SMOKE_REAL)
                           else "answerdotai/ModernBERT-large")
SPAN_LEN = 256 if (SMOKE and not SMOKE_REAL) else 2048
SPAN_EPOCHS = 2 if SMOKE else 8          # maximum; early stopping usually ends sooner
ES_PATIENCE = 4                          # half-epoch validation checks without improvement (= 2 epochs)
ES_MIN_EPOCHS = 1 if SMOKE else 3        # never stop before this (run 3 trained a fixed 5 epochs, still improving)
HEAD_TOK = {"SPAN_VIS": 400}
SPAN_ACCUM, SPAN_LR, SPAN_FAR_KEEP, SPAN_PW_CAP, P_A3_SPAN, SPAN_SENT_TOK = 8, 2e-5, 0.3, 8.0, 0.3, 96                              # micro-batch; gradients accumulated up to BATCH
S1B_NEG = 4                                     # stage-1 negatives per positive, resampled every epoch
P2B = {"tag": 0.1, "ctx": 0.1, "edrop": 0.05, "eadd": 0.02}
NFOLD = 2 if SMOKE else 5
STAGE = {"C0": "C0", "S1": "S1", "S2s": "S2", "S2f": "S2", "S2f_noA3": "S2", "S1b": "S1b", "S2b_f": "S2b", "S2b_s": "S2b",
         "C0a": "C0", "C0L": "C0", "C0aL": "C0", "C0Lctx": "C0", "SPAN": "SPAN", "SPAN_ES": "SPAN",
         "SPAN_VIS": "SPAN"}
INIT = {"S2f": "S1", "S2f_noA3": "S1", "S2b_f": "S1b"}           # config -> config whose same-fold/seed model it starts from
P_SWAP, P_TAGDROP, P_CTXDROP, P_EDROP, P_EADD = 0.3, 0.3, 0.3, 0.12, 0.05
FAR_KEEP = {"C0": 0.2, "S1": 1.0, "S2": 0.3, "S2b": 0.3}
TAB = re.compile(r"\btables?\b", re.I)
SUP = re.compile(r"supplement|additional file|figure s\d|fig\.? ?s\d|data not shown|not shown", re.I)
CAPPFX = re.compile(r"^\s*(?:Fig(?:ure)?\.?|FIG\.?)\s*\d+\s*[.:|]?\s*", re.I)
VERSION = "run4-v1"


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
    if st == "S1b":
        pos = [i for i in range(n) if i in gold]
        neg = [i for i in range(n) if i not in gold]
        if train:
            neg = rng.sample(neg, min(len(neg), S1B_NEG * max(1, len(pos))))
        for i in pos + neg:
            out.append((cap, swap(i, a["masked"][i], True), int(i in gold), key(i)))
        return out
    if st == "S2b":
        E = set(cit)
        if train:
            E = {c for c in cit if rng.random() >= P2B["edrop"]}
            if rng.random() < P2B["eadd"]:
                paras = {a["sent_para"][c] for c in cit}
                pool = [j for j in range(n) if a["sent_para"][j] in paras and j not in gold and j not in E]
                if pool:
                    E.add(rng.choice(pool))

        def b2(i, cand, Ex):
            pos_txt, m, d = _pos(a, i, Ex)
            par = "same" if (m is not None and a["sent_para"][i] == a["sent_para"][m]) else "other"
            tag = (f"[cites this figure: no | {pos_txt} | {par} paragraph | "   # candidates are never in E (detector view)
                   f"section: {_sec(a, i)} | cites: {_others(a, i, f)}]")
            ctx = S[m] if m is not None else "(none)"
            if train and rng.random() < P2B["tag"]:
                tag = "[position unknown]"
            if train and rng.random() < P2B["ctx"]:
                ctx = "(none)"
            return f"{tag} Citing: {ctx} Candidate: {cand}", d

        for i in range(n):
            if i in E:
                continue
            y = int(i in gold)
            txt, d = b2(i, swap(i, S[i], False), E)
            if train and d > 3 and y == 0 and rng.random() > FAR_KEEP["S2b"]:
                continue
            out.append((cap, txt, y, key(i)))
        if train:
            for c in cit:
                t = a["cdel"].get(f"{c}|{f}")
                if t:
                    tag_txt, _ = b2(c, t, E - {c})
                    out.append((cap, tag_txt, int(c in gold), key(c) + "|A3"))
        return out
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
            cand = swap(i, S[i], False) if cfg in ("C0a", "C0aL") else S[i]
            nb = (f" Before: {S[i - 1] if i > 0 else '(start of paper)'} After: {S[i + 1] if i + 1 < n else '(end of paper)'}"
                  if cfg == "C0Lctx" else "")
            out.append((cap, f"{tag} Citing: {ctx} Candidate: {cand}{nb}", y, key(i)))
        if train and cfg in ("C0a", "C0aL"):
            for c in cit:
                t = a["cdel"].get(f"{c}|{f}")
                if not t:
                    continue
                pos, m, _ = _pos(a, c, [j for j in cit if j != c])
                par = "same" if (m is not None and a["sent_para"][c] == a["sent_para"][m]) else "other"
                tag = (f"[cites this figure: no | {pos} | {par} paragraph | section: {_sec(a, c)} | "
                       f"cites: {_others(a, c, f)}]")
                ctx = S[m] if m is not None else "(none)"
                out.append((cap, f"{tag} Citing: {ctx} Candidate: {t}", int(c in gold), key(c) + "|A3"))
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


def base_of(cfg):
    return SPAN_BASE if STAGE.get(cfg) == "SPAN" else BASE_CFG.get(cfg, BASE)


def maxlen_of(cfg):
    return MAXLEN_CFG.get(cfg, MAXLEN.get(STAGE[cfg], SPAN_LEN))


def batches(tok, ex, maxlen, rng=None, bs=BATCH):
    """Length-grouped batches (shuffled chunks of 50 batches when training) to cut padding."""
    import torch
    idx = list(range(len(ex)))
    if rng is not None:
        rng.shuffle(idx)
        chunk = bs * 50
        idx = [j for k in range(0, len(idx), chunk) for j in sorted(idx[k:k + chunk], key=lambda j: len(ex[j][1]))]
        groups = [idx[k:k + bs] for k in range(0, len(idx), bs)]
        rng.shuffle(groups)
    else:
        idx.sort(key=lambda j: len(ex[j][0]) + len(ex[j][1]))
        groups = [idx[k:k + bs * 2] for k in range(0, len(idx), bs * 2)]
    for g in groups:
        e = tok([ex[j][0] for j in g], [ex[j][1] for j in g], truncation="longest_first", max_length=maxlen,
                padding=True, return_tensors="pt")
        e["labels"] = torch.tensor([float(ex[j][2]) for j in g])
        yield g, e


def train(arts, ids, cfg, seed, init, log):
    if STAGE[cfg] == "SPAN":
        return train_span(arts, ids, cfg, seed, init, log)
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(base_of(cfg))
    maxlen = maxlen_of(cfg)
    bs = MICRO.get(cfg, BATCH)
    accum = max(1, BATCH // bs)
    epochs = [build(arts, ids, cfg, random.Random(seed * 1000 + ep)) for ep in range(EPOCHS)]
    npos = sum(e[2] for e in epochs[0])
    pw = torch.tensor((len(epochs[0]) - npos) / max(1, npos))
    steps = sum(math.ceil(len(e) / BATCH) for e in epochs)
    model = load_model(init).to(dev)
    if cfg in GRAD_CKPT:
        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=LR_CFG.get(cfg, LR_STAGE.get(STAGE[cfg], LR)), weight_decay=WD)
    sch = get_linear_schedule_with_warmup(opt, int(WARM * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw.to(dev))
    model.train()
    losses, skipped, seen = [], 0, 0
    opt.zero_grad(set_to_none=True)
    for ep, ex in enumerate(epochs):
        tot, k = 0.0, 0
        for mb, (_, e) in enumerate(batches(tok, ex, maxlen, random.Random(seed * 7 + ep), bs)):
            e = {kk: v.to(dev) for kk, v in e.items()}
            y = e.pop("labels")
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                lg = model(**e).logits.float().squeeze(-1)
            loss = lossf(lg, y)
            seen += 1
            if not torch.isfinite(loss):            # fp16 overflow in the forward pass: skip this micro-batch
                skipped += 1
                if skipped > max(5, 0.02 * seen):
                    raise RuntimeError(f"non-finite loss in {cfg} seed {seed} ({skipped}/{seen} batches)")
                continue
            scaler.scale(loss / accum).backward()
            tot, k = tot + loss.item(), k + 1
            if (mb + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sch.step()
                opt.zero_grad(set_to_none=True)
        losses.append(round(tot / max(1, k), 4))
    log.update({"n_train": [len(e) for e in epochs], "n_pos": npos, "pos_weight": round(float(pw), 3),
                "steps": steps, "epoch_loss": losses, "skipped_batches": skipped, "base": base_of(cfg)})
    return model, tok


def score(model, tok, arts, ids, cfg):
    if STAGE[cfg] == "SPAN":
        return score_span(model, tok, arts, ids, cfg)
    import torch
    dev = next(model.parameters()).device
    ex = build(arts, ids, cfg, None)
    out = {}
    model.eval()
    with torch.no_grad():
        for g, e in batches(tok, ex, maxlen_of(cfg), None, EVAL_BS.get(cfg, BATCH) // 2):
            e = {k: v.to(dev) for k, v in e.items() if k != "labels"}
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                lg = model(**e).logits.float().squeeze(-1).cpu().numpy()
            for j, v in zip(g, np.atleast_1d(lg)):
                out[ex[j][3]] = float(v)
    model.train()
    return out


# ------------------------------------------------------------------ span tagger
def span_pieces(a, f, rng):
    """[(text, label, sentence index)] for the whole paper, with inline structure and figure flags."""
    S, gold, out = a["sentences"], a["gold"].get(f, set()), []
    prev_para = prev_sec = None
    for i in range(len(S)):
        pre = ""
        if a["sent_para"][i] != prev_para:
            sec = _sec(a, i)
            if sec != prev_sec:
                pre += f"§ {sec} "
                prev_sec = sec
            pre += "¶ "
            prev_para = a["sent_para"][i]
        cites, txt = f in a["cue"][i], S[i]
        if rng is not None and cites and a["cdel"].get(f"{i}|{f}") and rng.random() < P_A3_SPAN:
            txt, cites = a["cdel"][f"{i}|{f}"], False
        elif rng is not None and rng.random() < P_SWAP:
            txt = a["bt"].get(str(i), txt)
        others = [g for g in sorted(a["cue"][i]) if g != f]
        flag = ("[cites this figure] " if cites else "") + (f"[cites Figure {', '.join(map(str, others))}] " if others else "")
        out.append((pre + flag + txt, int(i in gold), i))
    return out


FIGTEXT = {}


def load_figtext():
    p = os.path.join(OUT, "figtext.json")
    if os.path.exists(p):
        FIGTEXT.update(json.load(open(p)))


def header(a, f, cfg):
    h = f"Figure {f}. " + a["captions"].get(f, "")
    if cfg == "SPAN_VIS":
        ft = FIGTEXT.get(f"{a['id']}|{f}")
        if ft:
            h += f" Figure text: {ft.get('ocr', '')} Figure shows: {ft.get('desc', '')}"
        else:
            h += " Figure shows: (no image available)"
    return h


def span_items(a, f, tok, rng, cfg="SPAN"):
    """Overlapping windows tiling the whole paper: (input ids, marker positions, labels, sentence indices)."""
    pcs = span_pieces(a, f, rng)
    ids = [x[:SPAN_SENT_TOK] for x in tok([t for t, _, _ in pcs], add_special_tokens=False)["input_ids"]]
    head = tok(header(a, f, cfg), add_special_tokens=False)["input_ids"][:min(HEAD_TOK.get(cfg, 200), SPAN_LEN // 4)]
    cls, sep, mk = tok.cls_token_id, tok.sep_token_id, tok.mask_token_id
    budget = SPAN_LEN - len(head) - 3
    n, s, items = len(pcs), 0, []
    while s < n:
        e, used = s, 0
        while e < n and used + len(ids[e]) + 1 <= budget:
            used += len(ids[e]) + 1
            e += 1
        e = max(e, s + 1)
        seq, pos, lab, idx = [cls] + head + [sep], [], [], []
        for j in range(s, e):
            pos.append(len(seq))
            seq += [mk] + ids[j]
            lab.append(pcs[j][1])
            idx.append(pcs[j][2])
        items.append((seq[:SPAN_LEN - 1] + [sep], [q for q in pos if q < SPAN_LEN - 1], lab, idx))
        if e >= n:
            break
        s += max(1, (2 * (e - s)) // 3)
    items = [(q, p, l[:len(p)], ix[:len(p)]) for q, p, l, ix in items]
    if rng is not None:
        cit = {i for i, c in enumerate(a["cue"]) if f in c}
        items = [it for it in items if any(it[2]) or cit & set(it[3]) or rng.random() < SPAN_FAR_KEEP]
    return items


def make_span_model(path):
    import torch
    from transformers import AutoModel

    class SpanModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc, err = None, None
            for kw in ({"dtype": torch.float32, "reference_compile": False}, {"dtype": torch.float32},
                       {"torch_dtype": torch.float32, "reference_compile": False}, {"torch_dtype": torch.float32}, {}):
                try:                                    # kwargs differ across transformers versions
                    self.enc = AutoModel.from_pretrained(path, attn_implementation="sdpa", **kw)
                    break
                except TypeError as ex:
                    err = ex
            if self.enc is None:
                raise err
            if hasattr(self.enc.config, "reference_compile"):
                self.enc.config.reference_compile = False      # no torch.compile on T4
            self.enc = self.enc.float()
            self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
            hp = os.path.join(path, "span_head.pt")
            if os.path.exists(hp):
                self.head.load_state_dict(torch.load(hp, map_location="cpu"))

        def forward(self, ids, pos):
            h = self.enc(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0]
            return self.head(h[pos].float()).squeeze(-1)

        def save(self, out):
            self.enc.save_pretrained(out, safe_serialization=True)
            torch.save(self.head.state_dict(), os.path.join(out, "span_head.pt"))

    return SpanModel()


def val_ids(arts):
    return sorted(p for p, a in arts.items() if a["split"] == "test")


def avg_precision(arts, scores):
    ys, vs = [], []
    for k, v in scores.items():
        p, f, i = k.split("|")[:3]
        ys.append(int(int(i) in arts[p]["gold"][int(f)]))
        vs.append(v)
    o = np.argsort(-np.array(vs))
    y = np.array(ys)[o]
    if y.sum() == 0:
        return 0.0
    hits = np.cumsum(y)
    return float((hits / np.arange(1, len(y) + 1))[y == 1].mean())


def val_loss(arts, scores):
    tot = 0.0
    for k, v in scores.items():
        p, f, i = k.split("|")[:3]
        y = int(int(i) in arts[p]["gold"][int(f)])
        tot += float(np.logaddexp(0, -v) if y else np.logaddexp(0, v))
    return tot / max(1, len(scores))


def train_span(arts, ids, cfg, seed, init, log):
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    if cfg == "SPAN_VIS":
        load_figtext()
    tok = AutoTokenizer.from_pretrained(SPAN_BASE)
    VAL = [p for p in val_ids(arts) if p not in ids]
    gen = lambda ep: [it for p in ids for f in arts[p]["figs"]
                      for it in span_items(arts[p], f, tok, random.Random(seed * 1000 + ep), cfg)]
    first = gen(0)
    npos = sum(sum(it[2]) for it in first)
    nmark = sum(len(it[2]) for it in first)
    pw = torch.tensor(min(SPAN_PW_CAP, (nmark - npos) / max(1, npos)))
    steps = math.ceil(len(first) / SPAN_ACCUM) * SPAN_EPOCHS
    model = make_span_model(init).to(dev)
    try:
        model.enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.enc.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=SPAN_LR, weight_decay=WD)
    sch = get_linear_schedule_with_warmup(opt, int(WARM * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw.to(dev))
    model.train()
    losses, skipped, seen, curve = [], 0, 0, []
    best, best_ap, bad, stop = None, -1.0, 0, False
    opt.zero_grad(set_to_none=True)

    def check(ep_frac):
        """Smoothed validation AP (mean of this and the previous check) decides both the best weights and stopping;
        one noisy check on 18 validation figures can neither fake a best nor trigger a stop."""
        nonlocal best, best_ap, bad, stop
        if not VAL:
            return
        sc = score_span(model, tok, arts, VAL, cfg)
        ap, vl = avg_precision(arts, sc), val_loss(arts, sc)
        sm = float(np.mean([ap] + [c[1] for c in curve[-1:]]))
        curve.append((round(ep_frac, 2), round(ap, 4), round(vl, 4), round(sm, 4)))
        if sm > best_ap + 1e-4:
            best_ap, bad = sm, 0
            best = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            stop = bad >= ES_PATIENCE and ep_frac >= ES_MIN_EPOCHS

    for ep in range(SPAN_EPOCHS):
        items = first if ep == 0 else gen(ep)
        order = list(range(len(items)))
        random.Random(seed * 7 + ep).shuffle(order)
        tot, k, half = 0.0, 0, len(order) // 2
        for mb, j in enumerate(order):
            seq, pos, lab, _ = items[j]
            x = torch.tensor([seq], device=dev)
            y = torch.tensor(lab, dtype=torch.float32, device=dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                lg = model(x, torch.tensor(pos, device=dev)).float()
            loss = lossf(lg, y)
            seen += 1
            if not torch.isfinite(loss):
                skipped += 1
                if skipped > max(5, 0.02 * seen):
                    raise RuntimeError(f"non-finite loss in {cfg} seed {seed} ({skipped}/{seen} windows)")
                continue
            scaler.scale(loss / SPAN_ACCUM).backward()
            tot, k = tot + loss.item(), k + 1
            if (mb + 1) % SPAN_ACCUM == 0 or mb + 1 == len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sch.step()
                opt.zero_grad(set_to_none=True)
            if mb + 1 == half:
                check(ep + 0.5)
                if stop:
                    break
        losses.append(round(tot / max(1, k), 4))
        if stop:
            break
        check(ep + 1.0)
        if stop:
            break
    if best is not None:
        model.load_state_dict(best)
    log.update({"n_train": len(first), "n_pos": npos, "n_markers": nmark, "pos_weight": round(float(pw), 3),
                "epoch_loss": losses, "skipped_batches": skipped, "base": SPAN_BASE, "val_papers": VAL,
                "val_curve": curve, "best_val_ap": round(best_ap, 4),
                "best_epoch": max(curve, key=lambda c: c[3])[0] if curve else None,
                "stopped_at": curve[-1][0] if curve else None})
    return model, tok


def score_span(model, tok, arts, ids, cfg="SPAN"):
    import torch
    dev = next(model.parameters()).device
    out = {}
    model.eval()
    with torch.no_grad():
        for p in ids:
            a = arts[p]
            for f in a["figs"]:
                acc = {}
                for seq, pos, _, idx in span_items(a, f, tok, None, cfg):
                    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                        lg = model(torch.tensor([seq], device=dev), torch.tensor(pos, device=dev)).float().cpu().numpy()
                    for i, v in zip(idx, np.atleast_1d(lg)):
                        acc.setdefault(i, []).append(float(v))
                for i, v in acc.items():
                    out[f"{p}|{f}|{i}"] = float(np.mean(v))
    model.train()
    return out


def window_auc(arts, scores):
    """AUC of held-out scores over window sentences (1-3 sentences from a citation of the figure)."""
    pos, neg = [], []
    for k, v in scores.items():
        p, f, i = k.split("|")[:3]
        a, f, i = arts[p], int(f), int(i)
        cit = [j for j, c in enumerate(a["cue"]) if f in c]
        if not cit or i in cit or min(abs(i - j) for j in cit) > 3:
            continue
        (pos if i in a["gold"][f] else neg).append(v)
    if not pos or not neg:
        return None
    neg = np.array(neg)
    return round(float(np.mean([(neg < x).mean() + 0.5 * (neg == x).mean() for x in pos])), 3)


def save_model(model, tok, path, cfg):
    os.makedirs(path, exist_ok=True)
    if STAGE[cfg] == "SPAN":
        model.save(path)
    else:
        model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    json.dump({"version": VERSION, "config": cfg, "stage": STAGE[cfg], "max_len": maxlen_of(cfg), "base": base_of(cfg),
               "input": __doc__.strip(),
               "decision": "per-article z-score of the logit >= threshold (threshold from phase8.py)"},
              open(os.path.join(path, "input_format.json"), "w"), indent=1)


# ------------------------------------------------------------------ jobs
def job_list(arts):
    train_ids = sorted(p for p, a in arts.items() if a["split"] == "train")
    folds = sorted({arts[p]["fold"] for p in train_ids})[:NFOLD]
    jobs = []
    order = [c for c in ("S1", "S2f", "S2f_noA3", "S1b", "S2b_f", "C0L", "C0aL", "C0Lctx", "SPAN", "SPAN_ES", "SPAN_VIS", "C0", "C0a", "S2s",
                         "S2b_s") if c in CONFIGS]
    chained = [c for c in order if c in INIT.values() or c in INIT]      # S1 then its stage-2 jobs, fold by fold: few live
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


def failed(jid):
    return os.path.exists(os.path.join(OUT, "failed", jid + ".txt"))


def mark_failed(jid, why):
    with open(os.path.join(OUT, "failed", jid + ".txt"), "w") as fh:
        fh.write(why)


def run_job(job, arts, gpu):
    t0 = time.time()
    if os.environ.get("R1_TEST_FAIL") == job["id"]:          # test hook for the failure path
        raise RuntimeError("forced test failure")
    log = {"id": job["id"], "gpu": gpu, "train_papers": job["train"], "scored_papers": job["score"]}
    init = base_of(job["cfg"])
    if job["cfg"] in INIT:
        init = os.path.join(TMP, job["deps"][0]) if job["fold"] is not None else os.path.join(OUT, "models", INIT[job["cfg"]])
    model, tok = train(arts, job["train"], job["cfg"], job["seed"], init, log)
    log["scores"] = score(model, tok, arts, job["score"], job["cfg"])
    if STAGE[job["cfg"]] in ("S1", "S1b") and job["fold"] is not None:
        log["s1_window_auc"] = window_auc(arts, log["scores"])
    if job["fold"] is None:
        save_model(model, tok, os.path.join(OUT, "models", job["cfg"]), job["cfg"])
        os.makedirs(os.path.join(OUT, "val"), exist_ok=True)
        json.dump(log["scores"], open(os.path.join(OUT, "val", job["cfg"] + ".json"), "w"))
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
    import torch
    arts = load_articles()
    jobs = job_list(arts)
    logf = open(os.path.join(OUT, "log.txt"), "a")
    while True:
        for j in jobs:                                      # a job whose prerequisite failed cannot run
            if not done(j["id"]) and not failed(j["id"]) and any(failed(d) for d in j["deps"]):
                mark_failed(j["id"], "prerequisite failed")
        todo = [j for j in jobs if not done(j["id"]) and not failed(j["id"])]
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
        except Exception as ex:                             # log, mark, free memory, carry on
            msg = f"[gpu {gpu}] FAILED {claimed['id']}: {ex!r}"[:2000]
            print(msg, flush=True), logf.write(msg + "\n"), logf.flush()
            mark_failed(claimed["id"], repr(ex)[:5000])
            ex = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        msg = (f"[gpu {gpu}] done  {claimed['id']}  {lg['minutes']} min  n_train {lg['n_train']}  "
               f"loss {lg['epoch_loss']}" + (f"  window_auc {lg['s1_window_auc']}" if "s1_window_auc" in lg else "")
               + f"  ({sum(done(j['id']) for j in jobs)}/{len(jobs)} jobs)")
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
    zp = os.path.join(os.path.dirname(OUT.rstrip("/")) or ".", os.path.basename(OUT.rstrip("/")) + "_scores.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("oof", "val", "jobs", "failed"):
            for fn in glob.glob(os.path.join(OUT, sub, "*.json")) + glob.glob(os.path.join(OUT, sub, "*.txt")):
                z.write(fn, os.path.join(sub, os.path.basename(fn)))
        for extra in ("log.txt", "figtext.json"):
            if os.path.exists(os.path.join(OUT, extra)):
                z.write(os.path.join(OUT, extra), extra)
    print(f"DOWNLOAD {zp} (scores + logs, a few MB). Weights: {OUT}/models/<cfg>/ (0.7-1.7 GB) stay in the "
          f"notebook output; 'New Dataset' from the output keeps them on Kaggle.")


def run_figtext(ngpu):
    """Image -> text with figtext.py, one shard per GPU, merged into OUT/figtext.json."""
    hits = [h for h in glob.glob("/kaggle/input/**/index.json", recursive=True) + glob.glob("**/figimg/index.json", recursive=True)
            + ([os.path.join(os.environ["R1_FIGIMG"], "index.json")] if os.environ.get("R1_FIGIMG") else [])
            if os.path.exists(h) and glob.glob(os.path.join(os.path.dirname(h), "*.png"))]
    if not hits:
        raise SystemExit("SPAN_VIS needs the figimg folder (index.json + PNGs) attached as input")
    img_dir = os.path.dirname(hits[0])
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figtext.py")
    if not os.path.exists(script):
        found = glob.glob("/kaggle/input/**/figtext.py", recursive=True)
        script = found[0] if found else script
    print(f"figtext: images from {img_dir}, {ngpu} shard(s)", flush=True)
    env_extra = {"FIGTEXT_LIMIT": os.environ.get("R1_FIGTEXT_LIMIT", "0")}
    procs = []
    for g in range(ngpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g) if ngpu > 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "", **env_extra)
        procs.append(subprocess.Popen([sys.executable, script, img_dir, os.path.join(OUT, f"figtext_{g}.json"), str(g), str(ngpu)], env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"figtext failed (exit codes {codes})")
    merged = {}
    for g in range(ngpu):
        merged.update(json.load(open(os.path.join(OUT, f"figtext_{g}.json"))))
    json.dump(merged, open(os.path.join(OUT, "figtext.json"), "w"), indent=1, ensure_ascii=False)
    print(f"figtext: {len(merged)} figures described", flush=True)


def launch():
    for d in ("jobs", "claims", "failed"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    for c in glob.glob(os.path.join(OUT, "claims", "*")):   # stale claims from a previous session
        if not done(os.path.basename(c)):
            shutil.rmtree(c, ignore_errors=True)
    for fl in glob.glob(os.path.join(OUT, "failed", "*.txt")):   # retry previously failed jobs
        os.remove(fl)
    arts = load_articles()
    jobs = job_list(arts)
    print(f"{len(jobs)} jobs ({sum(done(j['id']) for j in jobs)} already done); configs {CONFIGS}; seeds {SEEDS}; "
          f"epochs {SPAN_EPOCHS if CONFIGS == ['SPAN'] else EPOCHS}; base {[base_of(c) for c in CONFIGS]}", flush=True)
    for cfg in CONFIGS:
        tr = [p for p, a in arts.items() if a["split"] == "train"]
        if STAGE[cfg] == "SPAN":
            from transformers import AutoTokenizer
            tk = AutoTokenizer.from_pretrained(SPAN_BASE)
            its = [it for p in tr for f in arts[p]["figs"] for it in span_items(arts[p], f, tk, random.Random(0), cfg)]
            print(f"  SPAN      training windows on all 15 papers (one epoch): {len(its)}  markers "
                  f"{sum(len(it[2]) for it in its)}  positives {sum(sum(it[2]) for it in its)}")
            continue
        ex = build(arts, tr, cfg, random.Random(0))
        print(f"  {cfg:9s} training examples on all 15 papers (one epoch): {len(ex)}  positives {sum(e[2] for e in ex)}")
    try:
        import torch
        ngpu = torch.cuda.device_count()
    except Exception:
        ngpu = 0
    if "SPAN_VIS" in CONFIGS and not os.path.exists(os.path.join(OUT, "figtext.json")):
        run_figtext(max(1, ngpu))
    procs = []
    for g in range(max(1, ngpu)):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g) if ngpu else "",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker", str(g)], env=env))
    codes = [p.wait() for p in procs]
    merge()
    fails = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(OUT, "failed", "*.txt")))
    if fails:
        print("FAILED jobs:", fails, "- see log.txt; re-run launch to retry them")
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
