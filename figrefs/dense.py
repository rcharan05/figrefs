"""Transformer scorers for sentence <-> figure relevance.

BiEncoder   : embeds sentences and figure descriptions independently (fast; SPECTER2, BGE, MPNet)
CrossEncoder: reads (figure description, sentence) jointly (slower, more accurate; used to re-rank top-k)
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BI_MODELS = {
    "specter2": "allenai/specter2_base",
    "bge": "BAAI/bge-base-en-v1.5",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}
CROSS_MODELS = {
    "bge-reranker": "BAAI/bge-reranker-base",
    "msmarco": "cross-encoder/ms-marco-MiniLM-L-12-v2",
}
BGE_QUERY = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=4)
def _bi(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(BI_MODELS.get(name, name), device="cpu")


@lru_cache(maxsize=2)
def _cross(name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CROSS_MODELS.get(name, name), device="cpu", max_length=384)


def bi_similarity(name: str, queries: list[str], sentences: list[str], batch_size: int = 32) -> np.ndarray:
    """cosine similarity matrix [len(sentences), len(queries)]"""
    m = _bi(name)
    q = [BGE_QUERY + x for x in queries] if name == "bge" else queries
    Q = m.encode(q, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    S = m.encode(sentences, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return S @ Q.T


def cross_scores(name: str, pairs: list[tuple[str, str]], batch_size: int = 16) -> np.ndarray:
    """relevance logits for (query, passage) pairs, squashed to 0..1"""
    if not pairs:
        return np.zeros(0)
    raw = np.asarray(_cross(name).predict(pairs, batch_size=batch_size, show_progress_bar=False), dtype=float)
    if raw.min() < 0 or raw.max() > 1:
        raw = 1 / (1 + np.exp(-raw))
    return raw


_EMB_CACHE: dict[tuple[str, str], np.ndarray] = {}


def embed(name: str, texts: list[str], is_query: bool = False, batch_size: int = 32) -> np.ndarray:
    """normalised embeddings with an in-process cache (queries/sentences repeat a lot)"""
    pref = BGE_QUERY if (name == "bge" and is_query) else ""
    keys = [(name, pref + t) for t in texts]
    todo = list(dict.fromkeys(k[1] for k in keys if k not in _EMB_CACHE))
    if todo:
        E = _bi(name).encode(todo, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        for t, e in zip(todo, E):
            _EMB_CACHE[(name, t)] = e
    return np.stack([_EMB_CACHE[k] for k in keys]) if keys else np.zeros((0, 1))


_CROSS_CACHE: dict[tuple[str, str, str], float] = {}


def cross_cached(name: str, pairs: list[tuple[str, str]]) -> np.ndarray:
    todo = list(dict.fromkeys(p for p in pairs if (name,) + p not in _CROSS_CACHE))
    if todo:
        for p, v in zip(todo, cross_scores(name, todo)):
            _CROSS_CACHE[(name,) + p] = float(v)
    return np.array([_CROSS_CACHE[(name,) + p] for p in pairs])
