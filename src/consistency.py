"""
src/consistency.py — plausibility / honeypot detection.

Checks internal consistency of a candidate record without any hardcoded ID list.
Returns a plausibility_score in [0, 1]. Score near 0 → likely honeypot.

Each check is independent; deductions accumulate.
"""

import sys
sys.path.insert(0, ".")

from datetime import datetime, date
from config import (
    CONSISTENCY_YOE_DIFF_MONTHS,
    DEDUCT_YOE_MISMATCH, DEDUCT_DATE_ORDER, DEDUCT_FUTURE_DATE,
    DEDUCT_EXPERT_ZERO, DEDUCT_MULTI_EXPERT, DEDUCT_EDUCATION_DATE,
    DEDUCT_MEGA_TENURE,
)

# Reference "today" for the dataset — will be set once at module init
# and can be overridden if the caller knows the pool's max date.
_REFERENCE_DATE: date = date(2026, 6, 2)   # today per conversation context


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


def check_consistency(c: dict) -> tuple[float, list[str]]:
    """
    Returns (plausibility_score, [list of failure reasons]).
    plausibility_score = max(0, 1.0 - total_deduction)
    """
    deduction = 0.0
    reasons   = []
    profile   = c.get("profile", {})
    career    = c.get("career_history", [])
    skills    = c.get("skills", [])
    education = c.get("education", [])

    # ------------------------------------------------------------------ #
    # Check 1: yoe vs sum of career duration_months                       #
    # ------------------------------------------------------------------ #
    yoe_years   = profile.get("years_of_experience", 0) or 0
    yoe_months  = yoe_years * 12
    career_months = sum(max(0, r.get("duration_months", 0)) for r in career)

    if career_months > 0:
        diff = abs(yoe_months - career_months)
        if diff > CONSISTENCY_YOE_DIFF_MONTHS:
            deduction += DEDUCT_YOE_MISMATCH
            reasons.append(
                f"yoe_mismatch: claimed {yoe_years:.1f}yr "
                f"but career sums to {career_months/12:.1f}yr (diff={diff/12:.1f}yr)"
            )

    # ------------------------------------------------------------------ #
    # Check 2: role date ordering (start_date > end_date)                 #
    # ------------------------------------------------------------------ #
    for role in career:
        sd = _parse_date(role.get("start_date"))
        ed = _parse_date(role.get("end_date"))
        if sd and ed and sd > ed:
            deduction += DEDUCT_DATE_ORDER
            reasons.append(
                f"date_order: role '{role.get('title','')}' at '{role.get('company','')}' "
                f"start {sd} > end {ed}"
            )
            break  # one deduction for the whole record

    # ------------------------------------------------------------------ #
    # Check 3: future dates                                               #
    # ------------------------------------------------------------------ #
    for role in career:
        for field in ("start_date",):
            d = _parse_date(role.get(field))
            if d and d > _REFERENCE_DATE:
                deduction += DEDUCT_FUTURE_DATE
                reasons.append(f"future_date: {field}={d} is in the future")
                break

    # ------------------------------------------------------------------ #
    # Check 4: expert skill with duration_months = 0                     #
    # ------------------------------------------------------------------ #
    zero_expert_count = 0
    for s in skills:
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0:
            deduction += DEDUCT_EXPERT_ZERO
            zero_expert_count += 1
            reasons.append(f"expert_zero_duration: '{s.get('name','')}' is expert but 0 months")

    # ------------------------------------------------------------------ #
    # Check 5: ≥3 expert skills with 0 endorsements AND no assessments   #
    # ------------------------------------------------------------------ #
    assessments = c.get("redrob_signals", {}).get("skill_assessment_scores", {})
    suspicious_experts = [
        s for s in skills
        if s.get("proficiency") == "expert"
        and s.get("endorsements", 0) == 0
        and s.get("name", "") not in assessments
    ]
    if len(suspicious_experts) >= 3:
        deduction += DEDUCT_MULTI_EXPERT
        reasons.append(
            f"multi_expert_no_proof: {len(suspicious_experts)} expert skills with "
            f"0 endorsements and no assessment"
        )

    # ------------------------------------------------------------------ #
    # Check 6: education date ordering                                    #
    # ------------------------------------------------------------------ #
    for edu in education:
        sy = edu.get("start_year")
        ey = edu.get("end_year")
        if sy and ey and int(ey) < int(sy):
            deduction += DEDUCT_EDUCATION_DATE
            reasons.append(
                f"edu_date_order: {edu.get('institution','')} "
                f"end_year={ey} < start_year={sy}"
            )
            break

    # ------------------------------------------------------------------ #
    # Check 7: single role tenure > 360 months (30 years — implausible)  #
    # ------------------------------------------------------------------ #
    for role in career:
        dur = role.get("duration_months", 0)
        if dur > 360:
            deduction += DEDUCT_MEGA_TENURE
            reasons.append(
                f"mega_tenure: '{role.get('title','')}' at '{role.get('company','')}' "
                f"= {dur} months ({dur/12:.0f} years)"
            )
            break

    score = max(0.0, 1.0 - deduction)
    return round(score, 3), reasons


# ---------------------------------------------------------------------------
# __main__: dry-run — print plausibility score for all 50 candidates
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.parsing import load_candidates

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_candidates.json"
    candidates = load_candidates(path)

    print(f"{'candidate_id':<15} {'title':<35} {'plausibility':>12}  reasons")
    print("-" * 100)

    flagged = 0
    for c in candidates:
        score, reasons = check_consistency(c)
        title = c["profile"].get("current_title", "")[:34]
        flag = " ⚠" if score < 0.7 else ""
        print(f"{c['candidate_id']:<15} {title:<35} {score:>12.3f}{flag}  {'; '.join(reasons) if reasons else ''}")
        if score < 0.7:
            flagged += 1

    print(f"\nFlagged (score < 0.70): {flagged}/{len(candidates)}")
