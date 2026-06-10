"""
validate_submission.py - Format validator for submission CSV.

Checks all rules from submission_spec.md Section 3:
  - Exactly 100 rows (unless --allow-short for sample testing)
  - Ranks 1..100 each appearing exactly once
  - candidate_ids all unique
  - candidate_ids all exist in the candidates file
  - score is float, monotonically non-increasing with rank
  - reasoning column present (warns if empty)

Usage:
    python validate_submission.py submission.csv --candidates data/candidates.jsonl
    python validate_submission.py submission_sample.csv --candidates data/sample_candidates.json --allow-short
"""

import sys
import csv
import json
import gzip
import pathlib
import argparse


def load_valid_ids(candidates_path: str) -> set[str]:
    """Load all candidate IDs from a JSON array or JSONL file."""
    p = pathlib.Path(candidates_path)
    valid = set()
    if p.suffix == ".json":
        with open(candidates_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for c in data:
            valid.add(c["candidate_id"])
    else:
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(candidates_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    valid.add(json.loads(line)["candidate_id"])
    return valid


def validate(submission_path: str,
             candidates_path: str | None = None,
             allow_short: bool = False) -> bool:
    """
    Run all validation checks. Returns True if passed, False if any error.
    Prints PASS / ERROR / WARN messages.
    """
    errors  = []
    warns   = []
    passed  = []

    # --- Load CSV ---
    rows = []
    try:
        with open(submission_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"ERROR: Could not read CSV: {e}")
        return False

    # --- Column check ---
    required_cols = ["candidate_id", "rank", "score", "reasoning"]
    for c in required_cols:
        if c not in cols:
            errors.append(f"Missing required column: '{c}'")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        return False
    passed.append(f"Required columns present: {required_cols}")

    # --- Row count ---
    n = len(rows)
    if n == 100:
        passed.append(f"Row count: {n} (correct)")
    elif allow_short:
        warns.append(f"Row count: {n} (expected 100, --allow-short mode)")
    else:
        errors.append(f"Row count: {n} (expected exactly 100)")

    # --- Parse rank and score ---
    ranks  = []
    scores = []
    ids    = []
    parse_errors = 0
    for i, row in enumerate(rows):
        try:
            rank  = int(row["rank"])
            score = float(row["score"])
        except ValueError as e:
            errors.append(f"Row {i+2}: cannot parse rank/score: {e}")
            parse_errors += 1
            continue
        ranks.append(rank)
        scores.append(score)
        ids.append(row["candidate_id"].strip())

    if parse_errors:
        for e in errors:
            print(f"  ERROR: {e}")
        return False

    # --- Ranks ---
    expected_ranks = list(range(1, n + 1))
    if sorted(ranks) == expected_ranks:
        passed.append(f"Ranks 1..{n} each appear exactly once")
    else:
        missing = set(expected_ranks) - set(ranks)
        dupes   = [r for r in ranks if ranks.count(r) > 1]
        errors.append(f"Rank issues — missing: {sorted(missing)[:5]}, duplicates: {list(set(dupes))[:5]}")

    # --- Unique candidate_ids ---
    if len(set(ids)) == len(ids):
        passed.append("candidate_ids are unique")
    else:
        dupes = [x for x in ids if ids.count(x) > 1]
        errors.append(f"Duplicate candidate_ids: {list(set(dupes))[:5]}")

    # --- Score monotonically non-increasing ---
    mono_ok = True
    for i in range(1, len(scores)):
        if scores[i] > scores[i-1] + 1e-9:
            errors.append(f"Score increases at rank {i+1}: {scores[i-1]:.5f} -> {scores[i]:.5f}")
            mono_ok = False
            break
    if mono_ok:
        passed.append(f"Score monotonically non-increasing (max={max(scores):.4f}, min={min(scores):.4f})")

    # --- All-equal scores ---
    if len(set(scores)) == 1:
        warns.append("All scores are identical — model may not be differentiating candidates")

    # --- candidate_ids exist in pool ---
    if candidates_path:
        try:
            valid_ids = load_valid_ids(candidates_path)
            unknown = [cid for cid in ids if cid not in valid_ids]
            if unknown:
                errors.append(f"candidate_ids not in pool: {unknown[:5]}")
            else:
                passed.append(f"All candidate_ids exist in pool ({len(valid_ids):,} IDs checked)")
        except Exception as e:
            warns.append(f"Could not verify candidate_ids against pool: {e}")

    # --- Reasoning ---
    empty_reasoning = sum(1 for r in rows if not r.get("reasoning", "").strip())
    if empty_reasoning == 0:
        passed.append("All reasoning fields populated")
    else:
        warns.append(f"{empty_reasoning} rows have empty reasoning")

    # --- Summary ---
    print(f"\n{'='*55}")
    print(f"Submission validation: {submission_path}")
    print(f"{'='*55}")
    for p in passed:
        print(f"  PASS  {p}")
    for w in warns:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  ERROR {e}")
    print(f"{'='*55}")

    if errors:
        print(f"RESULT: FAILED ({len(errors)} error(s), {len(warns)} warning(s))")
        return False
    else:
        print(f"RESULT: PASSED ({len(warns)} warning(s))")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate submission CSV format.")
    parser.add_argument("submission", help="Path to submission CSV")
    parser.add_argument("--candidates", default=None,
                        help="Path to candidates file (to verify IDs exist)")
    parser.add_argument("--allow-short", action="store_true",
                        help="Allow fewer than 100 rows (for sample testing)")
    args = parser.parse_args()

    ok = validate(args.submission, candidates_path=args.candidates,
                  allow_short=args.allow_short)
    sys.exit(0 if ok else 1)
