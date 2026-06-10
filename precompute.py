"""
precompute.py - Phase A of the ranking pipeline.

Reads the candidates file, embeds all candidates, builds BM25 index,
extracts structured features, and saves artifacts to ./artifacts/.

Usage:
    python precompute.py --candidates data/sample_candidates.json
    python precompute.py --candidates data/candidates.jsonl   # full 100k

Artifacts produced:
    artifacts/candidate_embeddings.npy   float32 L2-normalised, shape (N, 768)
    artifacts/candidate_ids.json         list of candidate_ids in row order
    artifacts/jd_embedding.npy           float32, shape (768,)
    artifacts/bm25_index/                bm25s serialized index
    artifacts/features.parquet           structured feature columns
    artifacts/manifest.json              model names, dims, row count, content hash
"""

import sys
sys.path.insert(0, ".")

import json
import time
import hashlib
import pathlib
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

from config import BIENCODER_MODEL, EMBEDDING_DIM
from src.parsing import load_candidates, stream_jsonl
from src.text import build_candidate_text, build_jd_text
from src.features import extract_features
from src.retrieval import embed_texts, build_bm25, get_embedder


ARTIFACTS_DIR = pathlib.Path("artifacts")


def _hash_file(path: str, max_bytes: int = 10_000_000) -> str:
    """SHA256 of first max_bytes of the candidates file (for reproducibility tracking)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()[:16]


def precompute(candidates_path: str, batch_size: int = 64):
    t_total = time.time()
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("Precompute starting: " + candidates_path)
    print("="*60 + "\n")

    # 1. Load candidates
    t0 = time.time()
    print("Loading candidates ...")
    p = pathlib.Path(candidates_path)
    if p.suffix == ".json":
        candidates = load_candidates(candidates_path)
    else:
        candidates = list(tqdm(stream_jsonl(candidates_path), desc="Streaming"))
    print(f"  Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    ids   = [c["candidate_id"] for c in candidates]
    texts = [build_candidate_text(c) for c in candidates]
    n     = len(candidates)

    # 2. Embed candidates
    t0 = time.time()
    print(f"\nEmbedding {n:,} candidates with {BIENCODER_MODEL} ...")
    model = get_embedder()
    cand_vecs = embed_texts(texts, model=model, batch_size=batch_size,
                             normalize=True, show_progress=True)
    assert cand_vecs.shape == (n, EMBEDDING_DIM), f"Shape mismatch: {cand_vecs.shape}"
    print(f"  Embedded in {time.time()-t0:.1f}s")

    # 3. Embed JD
    jd_text = build_jd_text()
    jd_vec  = embed_texts([jd_text], model=model, normalize=True, show_progress=False)[0]

    # 4. Save embeddings
    emb_path = ARTIFACTS_DIR / "candidate_embeddings.npy"
    np.save(emb_path, cand_vecs)
    print(f"\nSaved embeddings: {emb_path}  ({cand_vecs.nbytes / 1e6:.1f} MB)")

    jd_path = ARTIFACTS_DIR / "jd_embedding.npy"
    np.save(jd_path, jd_vec)
    print(f"Saved JD embedding: {jd_path}")

    ids_path = ARTIFACTS_DIR / "candidate_ids.json"
    ids_path.write_text(json.dumps(ids))
    print(f"Saved candidate IDs: {ids_path}")

    # 5. Build BM25 index
    t0 = time.time()
    print(f"\nBuilding BM25 index over {n:,} texts ...")
    retriever, _ = build_bm25(texts)
    bm25_path = str(ARTIFACTS_DIR / "bm25_index")
    retriever.save(bm25_path)
    print(f"  BM25 built and saved: {bm25_path}  ({time.time()-t0:.1f}s)")

    # 6. Extract structured features -> parquet
    t0 = time.time()
    print("\nExtracting structured features ...")
    rows = []
    for c in tqdm(candidates, desc="Features"):
        f = extract_features(c)
        rows.append({k: v for k, v in f.items() if not k.startswith("_")})

    df = pd.DataFrame(rows)
    feat_path = ARTIFACTS_DIR / "features.parquet"
    df.to_parquet(feat_path, index=False)
    print(f"  Features saved: {feat_path}  ({time.time()-t0:.1f}s)")
    print(f"  Columns: {list(df.columns)}")

    # 7. Write manifest
    manifest = {
        "model"            : BIENCODER_MODEL,
        "embedding_dim"    : EMBEDDING_DIM,
        "n_candidates"     : n,
        "candidates_file"  : str(candidates_path),
        "candidates_hash"  : _hash_file(candidates_path),
        "artifacts"        : [
            "candidate_embeddings.npy",
            "candidate_ids.json",
            "jd_embedding.npy",
            "bm25_index/",
            "features.parquet",
        ],
    }
    manifest_path = ARTIFACTS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written: {manifest_path}")

    elapsed = time.time() - t_total
    print("\n" + "="*60)
    print(f"Precompute complete in {elapsed:.1f}s  ({n:,} candidates)")
    print("="*60 + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute embeddings and BM25 index.")
    parser.add_argument("--candidates", default="data/sample_candidates.json",
                        help="Path to candidates.json or candidates.jsonl")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    precompute(args.candidates, batch_size=args.batch_size)
