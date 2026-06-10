"""
src/retrieval.py — dense + BM25 + Reciprocal Rank Fusion retrieval.

Phase A (precompute): embed all candidates + JD → .npy files; build BM25 index.
Phase B (rank):       load artifacts, compute scores, return semantic_fits dict.

Also supports on-the-fly embedding of a small sample (≤200 candidates) for
dev/testing without running the full precompute pipeline.
"""

import sys
sys.path.insert(0, ".")

import os
import json
import numpy as np
from pathlib import Path

from config import (
    BIENCODER_MODEL, EMBEDDING_DIM, RRF_K,
    RERANKER_MODEL, RERANKER_SHORTLIST_K,
    RANK_TIME_LIMIT_SECONDS,
)
from src.text import build_candidate_text, build_jd_text


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def get_embedder():
    """Load and return the bi-encoder (cached after first call)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BIENCODER_MODEL)
    return model


def embed_texts(texts: list[str], model=None, batch_size: int = 64,
                normalize: bool = True, show_progress: bool = True) -> np.ndarray:
    """
    Embed a list of texts. Returns L2-normalized float32 ndarray of shape (N, D).
    """
    if model is None:
        model = get_embedder()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------

def build_bm25(corpus_texts: list[str]):
    """Build and return a bm25s index."""
    import bm25s
    tokenized = bm25s.tokenize(corpus_texts, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(tokenized)
    return retriever, tokenized


def bm25_scores_for_query(retriever, query_text: str,
                           n_docs: int | None = None) -> np.ndarray:
    """
    Return an array of BM25 scores (length = corpus size) for the given query.
    """
    import bm25s
    query_tokens = bm25s.tokenize([query_text], stopwords="en")
    n = n_docs or retriever.scores.shape[0]

    results, scores = retriever.retrieve(query_tokens, k=n)
    # results and scores are shape (1, k); expand to full-corpus array
    full = np.zeros(n, dtype=np.float32)
    for idx_in_result, corpus_idx in enumerate(results[0]):
        full[corpus_idx] = scores[0][idx_in_result]
    return full


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def rrf_fuse(rank_lists: list[np.ndarray], k: int = RRF_K) -> np.ndarray:
    """
    Fuse multiple ranked score arrays via Reciprocal Rank Fusion.
    Each element of rank_lists is a 1-D score array of length N.
    Returns an RRF score array of the same length N (higher = better).
    """
    n = len(rank_lists[0])
    rrf = np.zeros(n, dtype=np.float64)
    for scores in rank_lists:
        # Convert scores → rank positions (0 = best)
        order = np.argsort(-scores)          # descending
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        rrf += 1.0 / (k + ranks + 1)        # +1 so rank 0 → 1/(k+1)
    return rrf


# ---------------------------------------------------------------------------
# On-the-fly scoring for small samples (dev / dry-run)
# ---------------------------------------------------------------------------

def score_sample(candidates: list[dict],
                 jd_text: str | None = None,
                 show_progress: bool = True) -> dict[str, float]:
    """
    Embed a small batch of candidates on the fly and return
    {candidate_id: semantic_fit} where semantic_fit is RRF-fused, normalized 0..1.
    """
    if jd_text is None:
        jd_text = build_jd_text()

    texts = [build_candidate_text(c) for c in candidates]
    ids   = [c["candidate_id"] for c in candidates]
    n     = len(texts)

    print(f"[retrieval] Embedding {n} candidates + JD with {BIENCODER_MODEL} ...")
    model = get_embedder()
    cand_vecs = embed_texts(texts, model=model, show_progress=show_progress)
    jd_vec    = embed_texts([jd_text], model=model, show_progress=False)[0]

    # Dense cosine scores (vectors are L2-normalized → dot product = cosine)
    dense_scores = cand_vecs @ jd_vec         # shape (N,)

    # BM25 scores
    print("[retrieval] Building BM25 index ...")
    retriever, _ = build_bm25(texts)
    sparse_scores = bm25_scores_for_query(retriever, jd_text, n_docs=n)

    # RRF fusion
    rrf = rrf_fuse([dense_scores, sparse_scores])

    # Normalize to [0, 1]
    rrf_min, rrf_max = rrf.min(), rrf.max()
    if rrf_max > rrf_min:
        semantic_fits = (rrf - rrf_min) / (rrf_max - rrf_min)
    else:
        semantic_fits = np.ones(n) * 0.5

    return {cid: float(sf) for cid, sf in zip(ids, semantic_fits)}


# ---------------------------------------------------------------------------
# Artifact-based scoring for full 100k (used by rank.py)
# ---------------------------------------------------------------------------

def score_from_artifacts(candidate_ids: list[str],
                          artifacts_dir: str = "artifacts") -> dict[str, float]:
    """
    Load precomputed embeddings + BM25 artifacts and return semantic_fits.
    Expects: artifacts/candidate_embeddings.npy, artifacts/jd_embedding.npy,
             artifacts/candidate_ids.json, artifacts/bm25_index/ (bm25s format).
    """
    import bm25s
    adir = Path(artifacts_dir)

    # Load dense vectors
    cand_vecs = np.load(adir / "candidate_embeddings.npy")   # (N, D)
    jd_vec    = np.load(adir / "jd_embedding.npy")           # (D,)
    stored_ids: list[str] = json.loads((adir / "candidate_ids.json").read_text())

    assert cand_vecs.shape[0] == len(stored_ids), "Embedding count mismatch"
    assert cand_vecs.shape[1] == EMBEDDING_DIM, f"Expected dim {EMBEDDING_DIM}"

    # Dense scores
    dense_scores = cand_vecs @ jd_vec

    # BM25 scores
    bm25_path = str(adir / "bm25_index")
    retriever = bm25s.BM25.load(bm25_path, load_corpus=True)
    n = len(stored_ids)
    sparse_scores = bm25_scores_for_query(retriever, build_jd_text(), n_docs=n)

    # RRF
    rrf = rrf_fuse([dense_scores, sparse_scores])
    rrf_min, rrf_max = rrf.min(), rrf.max()
    if rrf_max > rrf_min:
        norm_rrf = (rrf - rrf_min) / (rrf_max - rrf_min)
    else:
        norm_rrf = np.ones(n) * 0.5

    return {cid: float(sf) for cid, sf in zip(stored_ids, norm_rrf)}


# ---------------------------------------------------------------------------
# Optional cross-encoder reranker (time-boxed)
# ---------------------------------------------------------------------------

def rerank_top_k(candidates_shortlist: list[dict],
                 jd_text: str,
                 semantic_fits: dict[str, float],
                 time_budget_seconds: float = 90.0) -> dict[str, float]:
    """
    Run a cross-encoder over the top-K shortlist and blend its score
    into the semantic_fit. Gracefully skips if time budget exceeded.
    """
    import time
    t0 = time.time()
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANKER_MODEL)
    except Exception as e:
        print(f"[retrieval] Cross-encoder load failed: {e}. Skipping rerank.")
        return semantic_fits

    pairs = [(jd_text, build_candidate_text(c)) for c in candidates_shortlist]
    scores = []
    for pair in pairs:
        if time.time() - t0 > time_budget_seconds:
            print(f"[retrieval] Cross-encoder time budget hit after {len(scores)} candidates. Stopping.")
            break
        scores.append(reranker.predict([pair])[0])

    if len(scores) < len(candidates_shortlist):
        # Incomplete rerank — blend only what we have
        completed_ids = [c["candidate_id"] for c in candidates_shortlist[:len(scores)]]
    else:
        completed_ids = [c["candidate_id"] for c in candidates_shortlist]

    # Normalize reranker scores to [0,1]
    if scores:
        s = np.array(scores, dtype=np.float32)
        s_min, s_max = s.min(), s.max()
        if s_max > s_min:
            s = (s - s_min) / (s_max - s_min)
        # Blend: 50% original RRF score + 50% cross-encoder
        for cid, rs in zip(completed_ids, s):
            orig = semantic_fits.get(cid, 0.5)
            semantic_fits[cid] = round(0.5 * orig + 0.5 * float(rs), 5)

    elapsed = time.time() - t0
    print(f"[retrieval] Rerank of {len(scores)} candidates in {elapsed:.1f}s")
    return semantic_fits


# ---------------------------------------------------------------------------
# __main__: dry-run on sample — print top-10 by semantic fit only
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.path.insert(0, ".")
    from src.parsing import load_candidates

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_candidates.json"
    candidates = load_candidates(path)

    fits = score_sample(candidates, show_progress=True)

    # Sort by semantic fit descending
    ranked = sorted(fits.items(), key=lambda x: x[1], reverse=True)
    by_id  = {c["candidate_id"]: c for c in candidates}

    print(f"\n{'Rank':<5} {'candidate_id':<15} {'title':<35} {'semantic_fit':>12}")
    print("-" * 70)
    for i, (cid, sf) in enumerate(ranked[:20], 1):
        title = by_id[cid]["profile"].get("current_title", "")[:34]
        print(f"{i:<5} {cid:<15} {title:<35} {sf:>12.4f}")
