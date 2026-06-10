"""
src/signals.py — availability multiplier from the 23 redrob behavioral signals.

Returns a float in [AVAIL_MIN, AVAIL_MAX] (default [0.35, 1.15]).
A multiplier < 1.0 means the candidate is less reachable/available.
A multiplier > 1.0 means strong engagement signals.
"""

import sys
sys.path.insert(0, ".")

from datetime import date, datetime
from config import (
    AVAIL_MIN, AVAIL_MAX,
    INACTIVE_SEVERE, INACTIVE_HEAVY, INACTIVE_MILD,
    RESPONSE_RATE_LOW, RESPONSE_RATE_HIGH,
    NOTICE_LONG, NOTICE_SHORT,
)

# Reference date: set once, either from pool's max last_active_date or today.
_REFERENCE_DATE: date = date(2026, 6, 2)


def set_reference_date(d: date):
    global _REFERENCE_DATE
    _REFERENCE_DATE = d


def _parse_date(s) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def compute_availability(redrob: dict) -> float:
    """
    Compute availability multiplier from redrob_signals dict.
    All adjustments are additive to a base of 1.0.
    """
    m = 1.0

    # ---- Recency of last login ----------------------------------------
    last_active = _parse_date(redrob.get("last_active_date"))
    if last_active:
        days_inactive = (_REFERENCE_DATE - last_active).days
        if days_inactive > INACTIVE_SEVERE:
            m -= 0.35
        elif days_inactive > INACTIVE_HEAVY:
            m -= 0.20
        elif days_inactive > INACTIVE_MILD:
            m -= 0.10

    # ---- Explicit availability signal ---------------------------------
    if redrob.get("open_to_work_flag", False):
        m += 0.10

    # ---- Recruiter response rate --------------------------------------
    rrr = redrob.get("recruiter_response_rate", 0.3)
    if rrr < RESPONSE_RATE_LOW:
        m -= 0.20
    elif rrr >= RESPONSE_RATE_HIGH:
        m += 0.08

    # ---- Response time (lower = more engaged) -------------------------
    avg_rt = redrob.get("avg_response_time_hours", 100)
    if avg_rt < 12:
        m += 0.05
    elif avg_rt > 200:
        m -= 0.05

    # ---- Notice period ------------------------------------------------
    notice = redrob.get("notice_period_days", 60)
    if notice > NOTICE_LONG:
        m -= 0.10
    elif notice <= NOTICE_SHORT:
        m += 0.08

    # ---- Interview completion rate ------------------------------------
    icr = redrob.get("interview_completion_rate", 0.5)
    if icr > 0.70:
        m += 0.05
    elif icr < 0.40:
        m -= 0.08

    # ---- GitHub activity (engineering signal) -------------------------
    gh = redrob.get("github_activity_score", -1)
    if gh > 20:
        m += 0.05

    # ---- Profile completeness ----------------------------------------
    pc = redrob.get("profile_completeness_score", 50)
    if pc >= 80:
        m += 0.03
    elif pc < 40:
        m -= 0.05

    # ---- Saved by recruiters (demand signal) -------------------------
    saved = redrob.get("saved_by_recruiters_30d", 0)
    if saved >= 10:
        m += 0.03

    # Clamp to allowed range
    return round(max(AVAIL_MIN, min(AVAIL_MAX, m)), 3)


# ---------------------------------------------------------------------------
# __main__: dry-run — print availability multiplier for all 50 candidates
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.parsing import load_candidates

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_candidates.json"
    candidates = load_candidates(path)

    print(f"{'candidate_id':<15} {'title':<30} {'open':>5} {'inactive_d':>10} {'resp_rate':>9} {'notice':>6} {'multiplier':>10}")
    print("-" * 100)

    for c in candidates:
        sig = c["redrob_signals"]
        last = _parse_date(sig.get("last_active_date"))
        days_in = (_REFERENCE_DATE - last).days if last else -1
        mult = compute_availability(sig)
        title = c["profile"].get("current_title", "")[:29]
        print(
            f"{c['candidate_id']:<15} {title:<30} "
            f"{str(sig.get('open_to_work_flag',''))[:5]:>5} "
            f"{days_in:>10} "
            f"{sig.get('recruiter_response_rate',0):>9.2f} "
            f"{sig.get('notice_period_days',60):>6} "
            f"{mult:>10.3f}"
        )
