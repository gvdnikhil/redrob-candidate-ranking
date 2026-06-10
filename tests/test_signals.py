"""
tests/test_signals.py — availability multiplier invariants.

The critical property: missing data is NEUTRAL. 64.6% of the pool has
github_activity_score = -1 and 59.6% has offer_acceptance_rate = -1
(no GitHub linked / no offer history). A sentinel must never move the
multiplier in either direction.
"""

import sys
sys.path.insert(0, ".")

from src.signals import compute_availability, set_reference_date
from datetime import date

set_reference_date(date(2026, 6, 2))


def _base_signals(**overrides):
    """A candidate with every signal in its neutral zone -> multiplier 1.0."""
    sig = {
        "last_active_date": "2026-06-01",
        "open_to_work_flag": False,
        "recruiter_response_rate": 0.3,
        "avg_response_time_hours": 100,
        "notice_period_days": 60,
        "interview_completion_rate": 0.5,
        "github_activity_score": 10,
        "profile_completeness_score": 50,
        "saved_by_recruiters_30d": 0,
    }
    sig.update(overrides)
    return sig


def test_neutral_baseline_is_one():
    assert compute_availability(_base_signals()) == 1.0


# ---- sentinel neutrality: -1 must equal "field absent" -------------------

def test_minus_one_sentinels_are_neutral():
    neutral = compute_availability(_base_signals())
    for field in [
        "recruiter_response_rate",
        "avg_response_time_hours",
        "notice_period_days",
        "interview_completion_rate",
        "github_activity_score",
        "profile_completeness_score",
        "saved_by_recruiters_30d",
    ]:
        with_sentinel = compute_availability(_base_signals(**{field: -1}))
        assert with_sentinel == neutral, f"{field}=-1 moved the multiplier"


def test_missing_fields_are_neutral():
    neutral = compute_availability(_base_signals())
    for field in [
        "recruiter_response_rate",
        "avg_response_time_hours",
        "notice_period_days",
        "interview_completion_rate",
        "github_activity_score",
        "profile_completeness_score",
        "saved_by_recruiters_30d",
    ]:
        sig = _base_signals()
        del sig[field]
        assert compute_availability(sig) == neutral, f"missing {field} moved the multiplier"


def test_null_fields_are_neutral():
    neutral = compute_availability(_base_signals())
    sig = _base_signals(recruiter_response_rate=None, avg_response_time_hours=None)
    assert compute_availability(sig) == neutral


# ---- real values still move the multiplier --------------------------------

def test_low_response_rate_penalized():
    assert compute_availability(_base_signals(recruiter_response_rate=0.05)) < 1.0


def test_high_response_rate_rewarded():
    assert compute_availability(_base_signals(recruiter_response_rate=0.8)) > 1.0


def test_open_to_work_rewarded():
    assert compute_availability(_base_signals(open_to_work_flag=True)) > 1.0


def test_long_inactivity_penalized():
    assert compute_availability(_base_signals(last_active_date="2025-09-01")) < 1.0


def test_zero_notice_period_rewarded():
    # 0 is a legitimate value (range 0-180), not a sentinel
    assert compute_availability(_base_signals(notice_period_days=0)) > 1.0


def test_active_github_rewarded():
    assert compute_availability(_base_signals(github_activity_score=50)) > 1.0


# ---- clamping --------------------------------------------------------------

def test_clamped_to_range():
    worst = _base_signals(
        last_active_date="2024-01-01",
        recruiter_response_rate=0.01,
        avg_response_time_hours=250,
        notice_period_days=150,
        interview_completion_rate=0.1,
        profile_completeness_score=30,
    )
    best = _base_signals(
        last_active_date="2026-06-01",
        open_to_work_flag=True,
        recruiter_response_rate=0.9,
        avg_response_time_hours=5,
        notice_period_days=0,
        interview_completion_rate=0.9,
        github_activity_score=90,
        profile_completeness_score=95,
        saved_by_recruiters_30d=20,
    )
    assert compute_availability(worst) >= 0.35
    assert compute_availability(best) <= 1.15
