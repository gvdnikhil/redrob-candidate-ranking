"""
rank.py - Phase B: load precomputed artifacts, score all candidates, write submission CSV.

Constraints (hackathon rules):
  - CPU only, no GPU
  - <= 5 min wall-clock
  - <= 16 GB RAM
  - No network calls, no model downloads

Usage:
    python rank.py --candidates data/sample_candidates.json --out submission.csv
    python rank.py --candidates data/candidates.jsonl --out submission.csv
    python rank.py --candidates data/sample_candidates.json --explain CAND_0000031
"""

import sys
sys.path.insert(0, ".")

import argparse
import csv
import json
import pathlib
import time

import numpy as np
import pandas as pd

from config import (
    RANK_TIME_LIMIT_SECONDS, RRF_K,
    RERANKER_SHORTLIST_K,
)
from src.parsing import load_candidates, stream_jsonl
from src.retrieval import rrf_fuse, bm25_scores_for_query, score_from_artifacts
from src.scoring import score_candidate, explain
from src.reasoning import generate_reasoning, validate_no_hallucination
from src.text import build_jd_text

ARTIFACTS_DIR = pathlib.Path("artifacts")
TOP_N = 100


def _load_artifacts():
    """Load all precomputed artifacts from disk."""
    cand_vecs  = np.load(ARTIFACTS_DIR / "candidate_embeddings.npy")  # (N, D)
    jd_vec     = np.load(ARTIFACTS_DIR / "jd_embedding.npy")           # (D,)
    cand_ids   = json.loads((ARTIFACTS_DIR / "candidate_ids.json").read_text())
    features_df = pd.read_parquet(ARTIFACTS_DIR / "features.parquet")
    return cand_vecs, jd_vec, cand_ids, features_df


def _compute_semantic_fits(cand_vecs, jd_vec, cand_ids, n) -> dict[str, float]:
    """Dense cosine + BM25 + RRF, return {candidate_id: semantic_fit}."""
    import bm25s

    # Dense scores (L2-normalised -> dot = cosine)
    dense_scores = cand_vecs @ jd_vec    # shape (N,)

    # BM25 scores
    bm25_path = str(ARTIFACTS_DIR / "bm25_index")
    retriever = bm25s.BM25.load(bm25_path, load_corpus=True)
    jd_text = build_jd_text()
    sparse_scores = bm25_scores_for_query(retriever, jd_text, n_docs=n)

    # RRF fusion
    rrf = rrf_fuse([dense_scores, sparse_scores])
    rrf_min, rrf_max = rrf.min(), rrf.max()
    if rrf_max > rrf_min:
        norm_rrf = (rrf - rrf_min) / (rrf_max - rrf_min)
    else:
        norm_rrf = np.ones(n, dtype=np.float32) * 0.5

    return {cid: float(sf) for cid, sf in zip(cand_ids, norm_rrf)}


def _enforce_submission_invariants(rows: list[dict]) -> list[dict]:
    """
    Ensure the submission has unique ranks, monotonically non-increasing score,
    and unique candidate_ids.
    """
    n_rows = len(rows)
    ranks  = [r["rank"] for r in rows]
    scores = [r["score"] for r in rows]
    ids    = [r["candidate_id"] for r in rows]

    assert sorted(ranks) == list(range(1, n_rows + 1)), "Ranks must be sequential unique integers"
    assert len(set(ids)) == n_rows, "Duplicate candidate_ids"

    # Monotonically non-increasing score
    for i in range(1, len(scores)):
        assert scores[i] <= scores[i-1] + 1e-9, \
            f"Score not non-increasing at rank {i+1}: {scores[i-1]:.5f} -> {scores[i]:.5f}"

    return rows


def rank_candidates(candidates_path: str, out_path: str,
                    artifacts_dir: str = "artifacts",
                    use_reranker: bool = False):
    t_start = time.time()
    global ARTIFACTS_DIR
    ARTIFACTS_DIR = pathlib.Path(artifacts_dir)

    print(f"\nRanking: {candidates_path}")

    # 1. Load candidates
    t0 = time.time()
    p = pathlib.Path(candidates_path)
    if p.suffix == ".json":
        candidates = load_candidates(candidates_path)
    else:
        from tqdm import tqdm
        candidates = list(tqdm(stream_jsonl(candidates_path), desc="Loading"))
    n = len(candidates)
    by_id = {c["candidate_id"]: c for c in candidates}
    print(f"  Loaded {n:,} candidates ({time.time()-t0:.1f}s)")

    # 2. Load precomputed artifacts
    t0 = time.time()
    cand_vecs, jd_vec, cand_ids, features_df = _load_artifacts()
    print(f"  Artifacts loaded ({time.time()-t0:.1f}s)")

    # Verify alignment
    if len(cand_ids) != n:
        print(f"  WARNING: artifacts have {len(cand_ids)} IDs but file has {n} candidates.")
        print("  Re-run precompute.py to regenerate artifacts.")
        sys.exit(1)

    # 3. Semantic fits via dense + BM25 + RRF
    t0 = time.time()
    print(f"  Computing semantic fits (dense+BM25+RRF) ...")
    semantic_fits = _compute_semantic_fits(cand_vecs, jd_vec, cand_ids, n)
    print(f"  Semantic fits done ({time.time()-t0:.1f}s)")

    # 4. Score all candidates
    t0 = time.time()
    print(f"  Scoring {n:,} candidates ...")
    scored = []
    for c in candidates:
        sem = semantic_fits.get(c["candidate_id"], 0.0)
        row = score_candidate(c, semantic_fit=sem)
        scored.append(row)
    # Primary: final_score descending; secondary: candidate_id ascending (deterministic tie-break)
    scored.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))
    print(f"  Scoring done ({time.time()-t0:.1f}s)")

    # Check time budget
    elapsed = time.time() - t_start
    if elapsed > RANK_TIME_LIMIT_SECONDS:
        print(f"  WARNING: already at {elapsed:.0f}s, approaching 5min budget!")

    # 5. Build submission rows (top 100, or all if < 100 candidates)
    actual_n = min(TOP_N, len(scored))
    top100 = scored[:actual_n]
    submission_rows = []
    for rank_idx, row in enumerate(top100, 1):
        cid = row["candidate_id"]
        candidate = by_id[cid]
        reasoning = generate_reasoning(rank_idx, candidate, row)

        # Validate no hallucination
        violations = validate_no_hallucination(reasoning, candidate)
        if violations:
            print(f"  HALLUCINATION WARNING rank {rank_idx} {cid}: {violations}")

        submission_rows.append({
            "candidate_id": cid,
            "rank"        : rank_idx,
            "score"       : round(row["final_score"], 5),
            "reasoning"   : reasoning,
        })

    # 6. Enforce invariants
    _enforce_submission_invariants(submission_rows)

    # 7. Write CSV
    out = pathlib.Path(out_path)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(submission_rows)

    total_time = time.time() - t_start
    print(f"\nSubmission written: {out}  ({total_time:.1f}s total)")
    print(f"Top 5:")
    for r in submission_rows[:5]:
        print(f"  #{r['rank']}  {r['candidate_id']}  score={r['score']}  {r['reasoning'][:80]}...")

    return submission_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank candidates and produce submission CSV.")
    parser.add_argument("--candidates", default="data/sample_candidates.json")
    parser.add_argument("--out",        default="submission.csv")
    parser.add_argument("--artifacts",  default="artifacts")
    parser.add_argument("--explain",    default=None,
                        help="Print full score breakdown for one candidate ID and exit.")
    parser.add_argument("--reranker",   action="store_true",
                        help="Enable cross-encoder reranking of top shortlist (slower).")
    args = parser.parse_args()

    if args.explain:
        # Explain mode: score one candidate and print breakdown
        candidates = load_candidates(args.candidates)
        by_id = {c["candidate_id"]: c for c in candidates}
        c = by_id.get(args.explain)
        if not c:
            print(f"Candidate {args.explain} not found in {args.candidates}")
            sys.exit(1)
        # Try to get semantic fit from artifacts if available
        try:
            semantic_fits = _compute_semantic_fits(
                *_load_artifacts()[:3], n=len(candidates)
            )
            sem = semantic_fits.get(args.explain, 0.0)
        except Exception:
            sem = 0.0
        explain(c, semantic_fit=sem)
    else:
        rank_candidates(
            args.candidates,
            args.out,
            artifacts_dir=args.artifacts,
            use_reranker=args.reranker,
        )
