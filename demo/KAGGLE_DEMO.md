# Any-paper demo on Kaggle (about 10 min)

## Setup (once)
1. Upload `figrefs_demo.zip` as a new Kaggle dataset named **figrefs-demo**.
2. Create a new notebook. Set Accelerator to **GPU T4 x2** (one GPU is used). Turn Internet **on**; it is only needed for `pip` when you add a PDF.
3. Click **Add Input → Notebooks** and add the notebooks that hold the models:
   - `figrefs-run2b`, which holds `models/C0L`;
   - `figrefs-run3-span`, which holds `models/SPAN`;
   - `figrefs-run4-span-es`, which holds `models/SPAN_ES`.
   If a notebook belongs to a friend's account, they need to share it with you first (Share → add your username). Otherwise, see "Models in different accounts" below.
   Also add the **figrefs-demo** dataset.

## Cells
```python
# 1. code + the 5 CS papers
import glob, shutil
src = glob.glob('/kaggle/input/**/figrefs_demo/demo/demo_infer.py', recursive=True)[0].rsplit('/demo/', 1)[0]
shutil.copytree(src, '/kaggle/working/fd', dirs_exist_ok=True)
%cd /kaggle/working/fd
```
```python
# 2. (optional) add any other paper: LaTeX source folder, JATS .xml or .pdf  ->  new.jsonl
# !pip -q install pymupdf
# !python demo/article_from_source.py new.jsonl mypaper=/kaggle/input/<dataset>/paper.pdf
# !cat cs_articles.jsonl >> new.jsonl      # keep the CS papers too
```
```python
# 3. score with every model found, then build the viewer file
!python demo/demo_infer.py score  cs_articles.jsonl /kaggle/working/demo_out
!python demo/demo_infer.py report cs_articles.jsonl /kaggle/working/demo_out
```
(If you used cell 2, put `new.jsonl` in place of `cs_articles.jsonl` in both lines.)

The log first prints `models found: {...}`. Check that C0L, SPAN and SPAN_ES are all listed. Download `/kaggle/working/demo_out/results.json`, plus the `scores_*.json` files if you want to keep them. In the viewer, click **Load results file…** and select it.

## What you get per paper
For each paper, the file holds these methods:
- **PARA**: FigSum+ Paragraph.
- **B0F**: our rule.
- **C0L, SPAN, SPAN_ES**: each model on its own, scored as raw logit ≥ t.
- **ENS** (C0L+SPAN, the test winner) and **ENS_ES** (C0L+SPAN_ES): the anchored ensembles.

The cut-offs `t` come from `demo/thresholds.json`, chosen on the 15 training papers. Any method whose models were not found is skipped.

ENS3 is not available here. It needs SPAN_VIS, and SPAN_VIS needs figure crops plus Qwen2-VL figure text.

## Models in different accounts
1. In each account, run cells 1 and 3 with only that account's model attached. The `score` step writes `scores_<cfg>.json`.
2. Download every `scores_*.json` into one folder `OUT`.
3. Run `python demo/demo_infer.py report cs_articles.jsonl OUT` anywhere. It needs only a CPU (numpy), and a laptop is fine.

## AI gold for the CS benchmark
See `label_pack/README.txt`. Put the 10 answers in a folder `L`, then run:

`python demo/demo_infer.py report cs_articles.jsonl OUT --labels L`

This prints the benchmark table: per-paper F1 of every method against gold, and Claude-vs-Gemini agreement.
