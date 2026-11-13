# Evaluation

## Explicit references (LaTeX-derived ground truth)
    python eval/fetch_arxiv.py --n 40          # downloads PDFs + LaTeX source into eval/arxiv/
    docker run --rm --init -p 8070:8070 grobid/grobid:0.8.2-crf   # only for the GROBID baseline
    python eval/run_eval.py --systems ours,grobid
    python eval/diagnose.py eval/arxiv/<id>    # per-paper misses / false positives

Ground truth = every \ref to a figure label in the LaTeX body (figures numbered in source order of top-level
\caption), paired with its LaTeX paragraph. Papers whose LaTeX numbering does not match the PDF
(caption-alignment < 0.6, checked on raw PDF text) and theses (>60 pages) are skipped; single GT mentions whose
figure number cannot be found near the paragraph in the raw PDF text are dropped.

## Implicit references (proxy)
    python eval/implicit_proxy.py && python eval/proxy_ensembles.py
Masked-mention figure identification: delete "Figure N" from explicit sentences and rank the paper's figures.
