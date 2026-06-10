"""
scripts/audit_sentinels.py — scan the full candidate pool and report, for every
numeric redrob signal, how often it is missing, null, or carries a -1 sentinel.

The signals doc only documents -1 sentinels for github_activity_score and
offer_acceptance_rate, but src/signals.py compares several raw values against
thresholds without guarding sentinels. This script produces the evidence for
which fields actually need a guard.

Usage:
    python scripts/audit_sentinels.py [--candidates data/candidates.jsonl]
"""

import argparse
import json
import sys

NUMERIC_SIGNALS = [
    "profile_completeness_score",
    "profile_views_received_30d",
    "applications_submitted_30d",
    "recruiter_response_rate",
    "avg_response_time_hours",
    "connection_count",
    "endorsements_received",
    "notice_period_days",
    "github_activity_score",
    "search_appearance_30d",
    "saved_by_recruiters_30d",
    "interview_completion_rate",
    "offer_acceptance_rate",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/candidates.jsonl")
    args = ap.parse_args()

    stats = {
        s: {"missing": 0, "null": 0, "neg_one": 0, "negative_other": 0,
            "min": None, "max": None}
        for s in NUMERIC_SIGNALS
    }
    total = 0

    with open(args.candidates, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            sig = json.loads(line).get("redrob_signals", {})
            for s in NUMERIC_SIGNALS:
                st = stats[s]
                if s not in sig:
                    st["missing"] += 1
                    continue
                v = sig[s]
                if v is None:
                    st["null"] += 1
                    continue
                if v == -1:
                    st["neg_one"] += 1
                elif v < 0:
                    st["negative_other"] += 1
                st["min"] = v if st["min"] is None else min(st["min"], v)
                st["max"] = v if st["max"] is None else max(st["max"], v)

    print(f"scanned {total} candidates\n")
    print(f"{'signal':<32} {'missing':>8} {'null':>6} {'-1':>8} {'-1 %':>7} {'other<0':>8} {'min':>8} {'max':>9}")
    print("-" * 92)
    for s in NUMERIC_SIGNALS:
        st = stats[s]
        pct = 100.0 * st["neg_one"] / total if total else 0.0
        print(
            f"{s:<32} {st['missing']:>8} {st['null']:>6} {st['neg_one']:>8} "
            f"{pct:>6.1f}% {st['negative_other']:>8} "
            f"{str(st['min']):>8} {str(st['max']):>9}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
