"""CLI: python -m figrefs <pdf-or-directory> -o <out_dir>"""
import argparse

from .pipeline import process_dir


def main():
    ap = argparse.ArgumentParser(prog="figrefs", description="Extract, for every figure in research-paper PDFs, "
                                 "the paragraphs/sentences that discuss it.")
    ap.add_argument("input", help="a PDF file or a directory of PDFs (searched recursively)")
    ap.add_argument("-o", "--out", default="figrefs_out", help="output directory")
    ap.add_argument("--dpi", type=int, default=200, help="resolution of cropped figure images")
    ap.add_argument("--implicit", default="rule", choices=["rule", "scorer"],
                    help="implicit references: 'rule' = validated B0f rule (default); 'scorer' = legacy v0.3 scorer")
    ap.add_argument("--threshold", type=float, default=0.5, help="[scorer only] implicit-sentence selection threshold")
    ap.add_argument("--dense", default="none", help="[scorer only] bi-encoder: bge | mpnet | specter2 | none")
    ap.add_argument("--rerank", default="none", help="[scorer only] cross-encoder re-ranker: bge-reranker | msmarco | none")
    ap.add_argument("--no-images", action="store_true", help="skip writing figure crops")
    a = ap.parse_args()
    process_dir(a.input, a.out, dpi=a.dpi, threshold=a.threshold, save_images=not a.no_images,
                dense=None if a.dense == "none" else a.dense, rerank=None if a.rerank == "none" else a.rerank,
                implicit_mode=a.implicit)


if __name__ == "__main__":
    main()
