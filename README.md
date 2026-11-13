# figrefs: finding every sentence that talks about a figure

For every figure in a research paper, `figrefs` returns all the text that discusses it:

- **explicit references**: sentences that name the figure ("as shown in Fig. 3B");
- **implicit references**: sentences that discuss what the figure shows without naming it.

Undergraduate project by Charan (230819), supervised by Prof. Soumya Dutta, IIT Kanpur, Fall 2026. The full write-up is in [`report/main.pdf`](report/main.pdf).

## Results in brief

| What | Result |
|---|---|
| Explicit references, sentence F1 | 0.93 (28 arXiv papers), 0.85 (177 held-out bioRxiv papers) |
| Implicit references, rule `B0f` | F1 0.752 on 15 FigSum+ papers (best published method: 0.659 on the same papers) |
| Learned ensemble (DeBERTa-v3-large + ModernBERT-large) | F1 0.766 out-of-fold |
| Pre-registered test, 3 unseen papers | **0.694 vs 0.622** for the rule (+0.071, 95% interval [+0.006, +0.144]) |
| Sanity check on one CS paper (LLM-agreement reference, not a benchmark) | model 0.807, rule 0.776, Paragraph 0.663 |

Benchmark: FigSum+ GS1 (Ramesh, Sethi & Yu, PLoS ONE 2015). All caveats are in the report.

Trained models: https://huggingface.co/iLuvU2/figrefs-models

## Setup

Python 3.9 or newer.

```bash
git clone <this repository>
cd figrefs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # the tool (CPU only)
pip install -r requirements-models.txt   # only if you want to run the learned models
```

## 1. Run the tool (rule-based, CPU, seconds per paper)

```bash
python -m figrefs path/to/paper.pdf -o out/      # one PDF
python -m figrefs path/to/folder/   -o out/      # every PDF in a folder
```

For each paper, `out/<paper>/` contains:

- `result.json`: for every figure, its caption, the explicit sentences, the implicit sentences and the combined `relevant_text`;
- `figures/fig_<n>.png`: the cropped figure;
- `report.html`: a self-contained page to review the output in a browser.

For a folder, `out/all_results.json` collects every paper. No model download and no GPU are needed.

Options: `--implicit rule` (default, the `B0f` rule) or `--implicit scorer` (an older similarity-based scorer, kept for comparison); `--no-images` skips figure crops.

## 2. Run the learned models and the interactive viewer

`demo/run_local.py` takes a paper and produces `explorer.html`, an interactive page showing, figure by figure, the sentences each method selects. It accepts a PDF, a JATS XML file, a LaTeX source folder, or a `.jsonl` file of already parsed papers.

**Without the models (CPU, a few seconds):** runs FigSum+ Paragraph and our rule `B0f`.

```bash
python demo/run_local.py path/to/paper.pdf --out demo_out
```

**With the models (GPU, Apple-silicon Mac or CPU):**

1. Install the extra requirements: `pip install -r requirements-models.txt`.
2. Download the trained models (about 6 GB in total) from Hugging Face into `models/`:

   ```bash
   pip install -U huggingface_hub
   hf download iLuvU2/figrefs-models --local-dir models
   ```

   This gives `models/C0L/`, `models/SPAN/` and `models/SPAN_ES/`. Each model folder contains an `input_format.json` file (and, for the span taggers, `span_head.pt`); the script finds them anywhere under `models/`. The models can also be retrained with the scripts in `eval/figsum/run1/`.
3. Run:

```bash
python demo/run_local.py path/to/paper.pdf --only SPAN_ES --out demo_out
```

`--only` limits scoring to the listed models; without it, every model found is used. With `C0L` and `SPAN` both present, the script also builds the anchored ensemble `ENS` (the system that won the pre-registered test). Cut-offs come from `demo/thresholds.json` and were chosen on the training papers. The device is picked automatically (CUDA, then Apple MPS, then CPU) and can be forced with `DEMO_DEVICE=cpu`.

On a laptop, `SPAN_ES` takes about three to four minutes for a 21-figure, 423-sentence paper; a GPU is faster.

| Model | What it is | Needed for |
|---|---|---|
| `SPAN_ES` | ModernBERT-large span tagger with early stopping | Recommended single model |
| `C0L` | DeBERTa-v3-large cross-encoder | Ensemble `ENS` |
| `SPAN` | ModernBERT-large span tagger, 5 epochs | Ensemble `ENS` |

**Comparing against reference labels (optional).** `demo/make_label_pack.py` writes one labelling prompt per paper from the exact sentence list the methods see. Save each annotator's JSON answer as `labels/<paper>_<annotator>.json`, then:

```bash
python demo/demo_infer.py report demo_out/articles.jsonl demo_out --labels labels
python demo/assemble_viewer.py demo_out/_empty.json demo_out/explorer_gold.html demo_out/results.json
```

This prints a table of per-figure precision, recall and F1 of every method against the sentences the annotators agree on (`--gold union|claude|gemini` changes that), and builds a viewer that shows the labels too. This is how the computer-science check in Section 9 of the report was produced.

**On Kaggle.** `demo/KAGGLE_DEMO.md` explains how to run the same scoring in a Kaggle notebook with the models attached as notebook inputs.

## Repository map

### `figrefs/`: the tool

| File | What it does |
|---|---|
| `__main__.py` | Command-line entry point (`python -m figrefs`). |
| `pipeline.py` | End to end for one PDF: parse, detect figures, explicit and implicit references, write JSON, figure crops and the HTML report. |
| `pdfparse.py` | Layout-aware PDF parser (PyMuPDF): paragraphs in reading order with section titles, captions, figure regions; stops at the reference list. |
| `textutil.py` | Text normalisation and an abbreviation-aware sentence splitter for scientific prose. |
| `doc.py` | The document model shared by all stages: paragraphs, sentences, figures, index of explicit mentions. |
| `explicit.py` | The explicit-reference grammar: "Fig. 3", "Figs. 4–6", "Figures 9(a), 10(a) and 11(a)", "Fig. 5a,b". |
| `implicit_rule.py` | The `B0f` rule for implicit references (default). |
| `implicit.py`, `dense.py` | The older similarity-based implicit scorer and its optional transformer scorers (superseded; kept for comparison). |
| `report.py` | Builds the self-contained HTML review page. |

### `demo/`: running the learned models on new papers

| File | What it does |
|---|---|
| `run_local.py` | One command on a laptop: paper in (PDF, XML, LaTeX or `.jsonl`), models run, `explorer.html` out. |
| `article_from_source.py` | Converts a PDF, JATS XML file or LaTeX source folder into the article format every method uses (sentences, paragraphs, sections, captions, citations). |
| `demo_infer.py` | `score`: runs every model found and writes `scores_<model>.json`. `report`: adds Paragraph and `B0f`, builds the ensembles, applies the cut-offs, optionally scores against reference labels, and writes `results.json`. |
| `thresholds.json` | Decision rule and cut-off of every method, chosen on the 15 training papers' out-of-fold scores. |
| `assemble_viewer.py`, `viewer_template.html` | Build the single-file interactive viewer from `results.json`. |
| `build_viewer_data.py` | Precomputes every method's output on the 18 FigSum+ papers for the viewer (needs the private training data; its output is not committed). |
| `make_label_pack.py` | Writes labelling prompts for reference annotators, aligned with the method sentence lists. |
| `KAGGLE_DEMO.md` | Running the scoring on Kaggle with the models attached. |

### `eval/`: benchmarks and experiments

Explicit references and figure detection:

| File | What it does |
|---|---|
| `fetch_arxiv.py`, `refetch_pdfs.py` | Download arXiv papers with LaTeX source (the arXiv-28 set). |
| `latex_gt.py`, `sentence_gt.py` | Ground truth for explicit references from LaTeX `\ref` commands and from JATS XML. |
| `run_eval.py`, `sentence_eval.py`, `diagnose.py` | Explicit-reference benchmarks (paragraph and sentence level) and per-paper error listing. |
| `figure_recall.py` | Figure detection recall: GROBID alone, GROBID with caption recovery, own parser. |
| `repro_grobid.py` | Check of our GROBID set-up against GROBID's published numbers. |
| `implicit_proxy.py`, `proxy_ensembles.py` | Early proxy benchmark for implicit scorers (masked-mention figure identification). |
| `results*.json`, `figrec_*.json` | Saved results of the above. |

Implicit references on FigSum+ (`eval/figsum/`):

| File | What it does |
|---|---|
| `data.py` | Loads the FigSum+ gold, builds the article model from JATS or GROBID output, aligns gold sentences. |
| `systems.py`, `evaluate.py`, `reproduce.py` | Re-implementation and scoring of the five published FigSum+ methods. |
| `distance_analysis.py` | Where implicit gold sentences lie relative to citations. |
| `features.py`, `feats3.py`, `experiments.py`, `experiments2.py`, `experiments3.py` | Feature tables and three rounds of feature-model experiments (logistic regression, boosted trees, stacking). |
| `phase1.py` … `phase5.py` | Rules and filters (`B0`, `B0f`), anaphora, far-sentence selection, LLM judges, CRF and text tiling. |
| `llm_judge.py`, `figtalk.py`, `tools/run_coref.py` | Small-LLM judge, the "figure-talk" classifier, coreference features. |
| `build_m2.py`, `m2_lite.py`, `kaggle/train_m2.py` | Free-label data from PMC papers and the cross-encoder trained on it ("M2"). |
| `build_ft.py`, `build_full.py`, `kaggle/ft_m2.py`, `phase6.py`, `phase7.py` | Fine-tuning on FigSum+ pairs (`build_ft.py` data) and the first whole-paper model (`build_full.py` data, trained with the same `ft_m2.py` by setting `FT_FILE`); `phase6.py` and `phase7.py` score them. |
| `pdf_route.py`, `run_ours.py` | PDF-to-result route through GROBID; the tool scored next to the published methods. |
| `*.json`, `*.log` | Saved scores and logs of these experiments. |

Transformer runs (`eval/figsum/run1/`):

| File | What it does |
|---|---|
| `articles.py`, `pdf_articles.py` | All 18 articles as one article model, the fixed split, the `B0f` used in evaluation; parser for the PDF-only articles. |
| `textaug.py`, `backtranslate.py`, `prepare.py` | Masking and citation deletion, back-translated paraphrases, and the final training data file. |
| `run1_kaggle.py` | Run 1: base cross-encoder and the two-stage curriculum. |
| `run2a_kaggle.py`, `run2b_kaggle.py` | Run 2: second attempt at stage 1; augmentation and the large cross-encoder. |
| `run3_kaggle.py` | Run 3: large cross-encoder variants and the span tagger (its implementation lives here). |
| `run4_kaggle.py`, `crop_figures.py`, `figtext.py` | Run 4: early stopping; figure crops, OCR and image descriptions. |
| `phase8.py` | Offline scorer: out-of-fold thresholds, comparison with `B0f`, bootstrap intervals. |
| `TEST_PLAN.md`, `test_eval.py`, `TEST_RESULTS.txt` | The pre-registered test: plan, script and output. |
| `run*_results.txt` | Result logs of every run. |

### `scripts/`: data fetching

| File | What it does |
|---|---|
| `fetch_figsum.py` | Downloads the open-access FigSum+ articles (PDF and XML) from PubMed Central. |
| `fetch_biorxiv.py` | Downloads the bioRxiv-177 sample. |
| `fetch_pmc.py` | Samples open-access PMC articles for free-label training data. |
| `run_grobid.py` | Runs a GROBID server over a folder of PDFs. |

### `report/`

`main.tex`, `sections/`, `refs.bib`, `figures/` and the compiled `main.pdf`. Build with `latexmk -pdf main.tex`.

## Reproducing the experiments

- **Paper text is not in this repository.** Files that contain the full text of papers are excluded on purpose: the FigSum+ training data (five articles are not open access), the precomputed viewer data for those papers (`demo/viewer_data.json`), and the parsed computer-science papers used in the report's sanity check. Every one of them can be rebuilt with the scripts here from papers you have access to.
- **Data that is not in this repository.** The FigSum+ gold standard (figshare 858903) goes in `eval/figsum/gold/`. Five of the 18 articles are not open access; their text cannot be redistributed, so the training data file (`run1_data.jsonl`) is not included. `eval/figsum/run1/articles.py` and `prepare.py` rebuild it from the XML files and your own copies of those PDFs.
- **Paths.** The experiment scripts were written for a fixed layout: the project at `/home/charan/proj/figrefs` and data at `/home/charan/bench`. Create those paths (or symbolic links to them), or edit the constants at the top of the scripts.
- **Training.** Each `run*_kaggle.py` is a single command on a Kaggle notebook with two T4 GPUs, for example `python run4_kaggle.py launch SPAN_ES`. Scores are then evaluated offline with `phase8.py`.
- **Published baselines.** `python eval/figsum/reproduce.py GOLD_DIR ARTICLE_ROOT`.
- **GROBID** (only for the GROBID parsing route): `docker run -d --init -p 8070:8070 grobid/grobid:0.9.1-crf`.

## Limitations

The learned models were trained and tested on 18 biomedical articles. Human agreement on this task is κ = 0.68, the test set has 18 figures, and references far from any citation are still mostly missed. See Section 10 of the report.
