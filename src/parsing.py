"""
src/parsing.py — streaming candidate loader.

Supports:
  - JSON array file  (sample_candidates.json)
  - JSONL file       (candidates.jsonl)
  - Gzipped JSONL    (candidates.jsonl.gz)

Returns plain dicts; does NOT load the entire file into RAM for JSONL/gz formats.
"""

import gzip
import json
import pathlib
from typing import Generator


def _required(c: dict, field: str):
    if field not in c:
        raise ValueError(f"Candidate {c.get('candidate_id', '?')} missing required field: {field}")


def _validate(c: dict) -> dict:
    """Light validation — checks required top-level fields exist."""
    for f in ("candidate_id", "profile", "career_history", "education", "skills", "redrob_signals"):
        _required(c, f)
    return c


def load_json_array(path: str) -> list[dict]:
    """Load a JSON array file (e.g. sample_candidates.json) fully into memory."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array")
    return [_validate(c) for c in data]


def stream_jsonl(path: str) -> Generator[dict, None, None]:
    """Stream a .jsonl or .jsonl.gz file one candidate at a time."""
    p = pathlib.Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield _validate(json.loads(line))


def load_candidates(path: str) -> list[dict]:
    """
    Load candidates from any supported format.
    For the full 100k JSONL, this loads everything into a list — ~465MB of dicts.
    If memory is a concern, use stream_jsonl() directly.
    """
    p = pathlib.Path(path)
    if p.suffix == ".json":
        return load_json_array(path)
    # .jsonl or .jsonl.gz
    return list(stream_jsonl(path))


# ---------------------------------------------------------------------------
# Quick helpers used across modules
# ---------------------------------------------------------------------------

def get_all_description_text(candidate: dict) -> str:
    """Concatenate all career_history descriptions into one searchable string."""
    parts = []
    for role in candidate.get("career_history", []):
        desc = role.get("description", "").strip()
        if desc:
            parts.append(desc)
    return " ".join(parts).lower()


def get_all_skill_names(candidate: dict) -> list[str]:
    """Return lowercase skill names."""
    return [s["name"].lower() for s in candidate.get("skills", []) if s.get("name")]


# ---------------------------------------------------------------------------
# __main__: dry-run — print summary table for all candidates in the file
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_candidates.json"
    candidates = load_candidates(path)

    header = f"{'candidate_id':<15} {'current_title':<35} {'yoe':>5} {'country':<12} {'open_to_work'}"
    print(header)
    print("-" * len(header))

    for c in candidates:
        p = c["profile"]
        sig = c["redrob_signals"]
        print(
            f"{c['candidate_id']:<15} "
            f"{p.get('current_title', '')[:34]:<35} "
            f"{p.get('years_of_experience', 0):>5.1f} "
            f"{p.get('country', '')[:11]:<12} "
            f"{sig.get('open_to_work_flag', False)}"
        )

    print(f"\nTotal: {len(candidates)} candidates loaded from {path}")
