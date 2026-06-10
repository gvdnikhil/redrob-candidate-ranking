"""
src/scoring.py — composite candidate score.

Formula:
    fit          = W_SEM*semantic_fit + W_CAREER*career_quality
                 + W_MUST*musthave_evidence + W_LOC*location_fit
    final_score  = max(0, fit * availability_multiplier - penalty_total)
                 * plausibility_score

All weights and penalties are in config.py.
Pass semantic_fit=0.0 when retrieval hasn't been run yet (structured-only pass).
"""

import sys
sys.path.insert(0, ".")

from config import (
    W_SEM, W_CAREER, W_MUST, W_LOC,
    PENALTY_BAD_TITLE, PENALTY_CONSULTING_ONLY, PENALTY_RESEARCH_ONLY,
    PENALTY_CV_SPEECH, PENALTY_NON_INDIA, PENALTY_SHORT_TENURE,
)
from src.features import extract_features
from src.signals import compute_availability
from src.consistency import check_consistency


# ---------------------------------------------------------------------------
# Sub-score helpers
# ---------------------------------------------------------------------------

def _career_quality(feat: dict) -> float:
    """
    Score based on career substance: product-company ML years, title quality,
    production shipping evidence.
    Returns 0..1.
    """
    score = 0.0

    # Product-company ML months (capped benefit at 60 months = 5 years)
    pm = min(feat["product_ml_months"], 60) / 60
    score += 0.40 * pm

    # Title class
    tc = feat["title_class"]
    if tc == "good":
        score += 0.35
    elif tc == "neutral":
        score += 0.10
    # "bad" → 0

    # Shipped-to-production signal
    if feat["has_production"]:
        score += 0.15

    # YOE in target range (4–10 years)
    if feat["yoe_in_range"]:
        score += 0.10

    return min(1.0, score)


def _musthave_evidence(feat: dict) -> float:
    """
    Score based on evidence for the 4 hard must-haves.
    Each must-have contributes equally; bonus for recsys/finetune.
    Returns 0..1.
    """
    count = feat["musthave_count"]         # 0..4
    base  = count / 4.0                    # 0.0..1.0

    bonus = 0.0
    if feat["has_recsys"]:
        bonus += 0.10
    if feat["has_finetune"]:
        bonus += 0.05

    return min(1.0, base + bonus)


def _location_score(feat: dict) -> float:
    return 1.0 if feat["location_fit"] else 0.0


def _penalties(feat: dict, candidate: dict) -> float:
    total = 0.0
    reasons = []

    # Bad title (non-engineering role)
    if feat["title_class"] == "bad":
        total += PENALTY_BAD_TITLE
        reasons.append(f"bad_title({feat['current_title']}): -{PENALTY_BAD_TITLE}")

    # Entire career at consulting firms
    if feat["consulting_only"]:
        total += PENALTY_CONSULTING_ONLY
        reasons.append(f"consulting_only: -{PENALTY_CONSULTING_ONLY}")

    # Outside India and not willing to relocate
    if not feat["location_fit"]:
        total += PENALTY_NON_INDIA
        reasons.append(f"location_mismatch: -{PENALTY_NON_INDIA}")

    # Short-tenure job-hopper: avg months per role < 12
    career = feat.get("_career", [])
    if len(career) >= 3:
        durations = [r.get("duration_months", 0) for r in career]
        avg_dur = sum(durations) / len(durations)
        if avg_dur < 12:
            total += PENALTY_SHORT_TENURE
            reasons.append(f"short_tenure(avg {avg_dur:.0f}mo): -{PENALTY_SHORT_TENURE}")

    return total, reasons


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score_candidate(candidate: dict, semantic_fit: float = 0.0) -> dict:
    """
    Returns a dict with final_score and all sub-scores.
    semantic_fit should be in [0, 1]; pass 0.0 for structured-only pass.
    """
    feat        = extract_features(candidate)
    avail       = compute_availability(candidate["redrob_signals"])
    plausib, _  = check_consistency(candidate)
    penalty, p_reasons = _penalties(feat, candidate)

    cq   = _career_quality(feat)
    mh   = _musthave_evidence(feat)
    loc  = _location_score(feat)

    fit  = (W_SEM * semantic_fit
          + W_CAREER * cq
          + W_MUST   * mh
          + W_LOC    * loc)

    raw_score   = max(0.0, fit * avail - penalty)
    final_score = round(raw_score * plausib, 5)

    return {
        "candidate_id"     : candidate["candidate_id"],
        "current_title"    : feat["current_title"],
        "yoe"              : feat["yoe"],
        "final_score"      : final_score,
        # Sub-scores for explain mode
        "semantic_fit"     : round(semantic_fit, 4),
        "career_quality"   : round(cq, 4),
        "musthave_evidence": round(mh, 4),
        "location_score"   : round(loc, 4),
        "fit"              : round(fit, 4),
        "availability"     : avail,
        "penalty"          : round(penalty, 4),
        "penalty_reasons"  : p_reasons,
        "plausibility"     : plausib,
        # Feature pass-through for reasoning
        "_feat"            : feat,
    }


def score_batch(candidates: list, semantic_fits: dict | None = None) -> list[dict]:
    """
    Score a list of candidates.
    semantic_fits: optional dict of {candidate_id: float}.
    Returns list sorted by final_score descending.
    """
    sf = semantic_fits or {}
    results = []
    for c in candidates:
        sem = sf.get(c["candidate_id"], 0.0)
        results.append(score_candidate(c, semantic_fit=sem))
    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results


def explain(candidate: dict, semantic_fit: float = 0.0):
    """Print a full score breakdown for one candidate."""
    r = score_candidate(candidate, semantic_fit=semantic_fit)
    print(f"\n{'='*60}")
    print(f"EXPLAIN: {r['candidate_id']} — {r['current_title']} ({r['yoe']:.1f}yr)")
    print(f"{'='*60}")
    print(f"  final_score       : {r['final_score']:.5f}")
    print(f"  fit (pre-avail)   : {r['fit']:.4f}")
    print(f"    semantic_fit    : {r['semantic_fit']:.4f}  (weight {W_SEM})")
    print(f"    career_quality  : {r['career_quality']:.4f}  (weight {W_CAREER})")
    print(f"    musthave_evid.  : {r['musthave_evidence']:.4f}  (weight {W_MUST})")
    print(f"    location_score  : {r['location_score']:.4f}  (weight {W_LOC})")
    print(f"  availability      : {r['availability']:.3f}  (multiplier)")
    print(f"  penalty           : {r['penalty']:.4f}")
    for pr in r["penalty_reasons"]:
        print(f"    - {pr}")
    print(f"  plausibility      : {r['plausibility']:.3f}")
    feat = r["_feat"]
    print(f"\n  Feature highlights:")
    print(f"    title_class     : {feat['title_class']}")
    print(f"    services_frac   : {feat['services_fraction']:.2f}")
    print(f"    consulting_only : {feat['consulting_only']}")
    print(f"    has_embeddings  : {feat['has_embeddings']}")
    print(f"    has_vectordb    : {feat['has_vectordb']}")
    print(f"    has_eval        : {feat['has_eval']}")
    print(f"    has_python      : {feat['has_python']}")
    print(f"    has_recsys      : {feat['has_recsys']}")
    print(f"    musthave_count  : {feat['musthave_count']}")
    print(f"    location_fit    : {feat['location_fit']}")
    print(f"    notice_days     : {feat['notice_days']}")


# ---------------------------------------------------------------------------
# __main__: dry-run ranked table (semantic_fit=0) + explain two candidates
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.parsing import load_candidates

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_candidates.json"
    candidates = load_candidates(path)
    results = score_batch(candidates)

    print(f"\n{'Rank':<5} {'candidate_id':<15} {'title':<35} {'yoe':>5}  {'score':>8}  {'avail':>6}  {'pen':>6}")
    print("-" * 90)
    for i, r in enumerate(results, 1):
        print(
            f"{i:<5} {r['candidate_id']:<15} "
            f"{r['current_title'][:34]:<35} "
            f"{r['yoe']:>5.1f}  "
            f"{r['final_score']:>8.4f}  "
            f"{r['availability']:>6.3f}  "
            f"{r['penalty']:>6.3f}"
        )

    # Explain top candidate and CAND_0000031
    by_id = {c["candidate_id"]: c for c in candidates}
    explain(by_id["CAND_0000031"])
