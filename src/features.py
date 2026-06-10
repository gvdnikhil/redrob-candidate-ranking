"""
src/features.py — structured feature extraction from a candidate record.

All logic is pure Python / regex — no ML. Returns a flat dict of named features
that scoring.py and signals.py consume. Also callable as a batch to build a DataFrame.
"""

import re
import sys
sys.path.insert(0, ".")

from config import (
    BAD_TITLE_PATTERNS, GOOD_TITLE_PATTERNS,
    SERVICES_FIRMS, TARGET_LOCATIONS,
    PATTERN_EMBEDDINGS, PATTERN_VECTORDB, PATTERN_EVAL,
    PATTERN_PYTHON, PATTERN_PRODUCTION, PATTERN_RECSYS, PATTERN_FINETUNE,
)
from src.parsing import get_all_description_text, get_all_skill_names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text, re.IGNORECASE))


def _classify_title(title: str) -> str:
    t = title.lower()
    for pat in BAD_TITLE_PATTERNS:
        if re.search(pat, t):
            return "bad"
    for pat in GOOD_TITLE_PATTERNS:
        if re.search(pat, t):
            return "good"
    return "neutral"


def _services_fraction(career_history: list) -> float:
    """Fraction of total career months spent at pure-services firms."""
    total_months = 0
    services_months = 0
    for role in career_history:
        dur = max(0, role.get("duration_months", 0))
        total_months += dur
        company = role.get("company", "").lower()
        if any(sf in company for sf in SERVICES_FIRMS):
            services_months += dur
    if total_months == 0:
        return 0.0
    return services_months / total_months


def _product_ml_months(career_history: list) -> int:
    """
    Months in roles with ML/AI/data/search/engineering titles
    at NON-services companies.
    """
    ml_titles = re.compile(
        r"ml|ai|machine learning|data scientist|data engineer|"
        r"search|nlp|recommendation|ranking|retrieval|applied|"
        r"engineer|developer|architect",
        re.IGNORECASE,
    )
    months = 0
    for role in career_history:
        company = role.get("company", "").lower()
        if any(sf in company for sf in SERVICES_FIRMS):
            continue
        title = role.get("title", "")
        if ml_titles.search(title):
            months += max(0, role.get("duration_months", 0))
    return months


def _location_fit(profile: dict, redrob: dict) -> bool:
    location = profile.get("location", "").lower()
    country  = profile.get("country", "").lower()
    willing  = redrob.get("willing_to_relocate", False)

    if country == "india":
        return True
    if willing:
        return True
    # Check if any target city appears in location string
    for city in TARGET_LOCATIONS:
        if city in location:
            return True
    return False


def _has_skill(skills: list, proficiencies: tuple = ("intermediate", "advanced", "expert")) -> dict:
    """Build a lookup: skill_name_lower → proficiency."""
    return {
        s["name"].lower(): s.get("proficiency", "beginner")
        for s in skills
        if s.get("proficiency", "beginner") in proficiencies
    }


# ---------------------------------------------------------------------------
# Main feature extractor
# ---------------------------------------------------------------------------

def extract_features(c: dict) -> dict:
    """
    Returns a flat dict of features for one candidate.
    All keys match what scoring.py expects.
    """
    profile  = c.get("profile", {})
    career   = c.get("career_history", [])
    skills   = c.get("skills", [])
    redrob   = c.get("redrob_signals", {})

    # Combine all text for regex search
    desc_text  = get_all_description_text(c)           # lowercase
    skill_text = " ".join(get_all_skill_names(c))      # lowercase
    search_text = desc_text + " " + skill_text

    # ---- Title classification ----
    title_class = _classify_title(profile.get("current_title", ""))

    # ---- Career quality signals ----
    svc_frac     = _services_fraction(career)
    consulting_only = svc_frac > 0.95 and len(career) >= 2
    product_ml_m = _product_ml_months(career)

    # ---- Must-have evidence flags ----
    has_embeddings  = _match(PATTERN_EMBEDDINGS,  search_text)
    has_vectordb    = _match(PATTERN_VECTORDB,    search_text)
    has_eval        = _match(PATTERN_EVAL,        search_text)
    has_python      = _match(PATTERN_PYTHON,      search_text)
    has_production  = _match(PATTERN_PRODUCTION,  search_text)
    has_recsys      = _match(PATTERN_RECSYS,      search_text)
    has_finetune    = _match(PATTERN_FINETUNE,    search_text)

    # Count how many of the 4 hard must-haves are evidenced
    musthave_count = sum([has_embeddings, has_vectordb, has_eval, has_python])

    # ---- Location / relocation ----
    india_loc    = profile.get("country", "").lower() == "india"
    loc_fit      = _location_fit(profile, redrob)

    # ---- Experience range ----
    yoe = profile.get("years_of_experience", 0)
    yoe_in_range = 4 <= yoe <= 10  # JD says 5-9, allow ±1

    # ---- Notice period ----
    notice_days = redrob.get("notice_period_days", 60)

    # ---- Skill assessment quality ----
    assessments = redrob.get("skill_assessment_scores", {})
    avg_assessment = (sum(assessments.values()) / len(assessments)) if assessments else -1

    return {
        "candidate_id"      : c["candidate_id"],
        "current_title"     : profile.get("current_title", ""),
        "yoe"               : yoe,
        "yoe_in_range"      : yoe_in_range,
        "title_class"       : title_class,
        "services_fraction" : round(svc_frac, 2),
        "consulting_only"   : consulting_only,
        "product_ml_months" : _product_ml_months(career),
        "has_embeddings"    : has_embeddings,
        "has_vectordb"      : has_vectordb,
        "has_eval"          : has_eval,
        "has_python"        : has_python,
        "has_production"    : has_production,
        "has_recsys"        : has_recsys,
        "has_finetune"      : has_finetune,
        "musthave_count"    : musthave_count,
        "india_location"    : india_loc,
        "location_fit"      : loc_fit,
        "notice_days"       : notice_days,
        "avg_assessment"    : round(avg_assessment, 1),
        # Raw pass-through for scoring.py
        "_profile"          : profile,
        "_career"           : career,
        "_skills"           : skills,
        "_redrob"           : redrob,
    }


def extract_features_batch(candidates: list) -> list[dict]:
    return [extract_features(c) for c in candidates]


# ---------------------------------------------------------------------------
# __main__: dry-run — show key feature columns for all 50 candidates
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import pandas as pd
    sys.path.insert(0, ".")
    from src.parsing import load_candidates

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_candidates.json"
    candidates = load_candidates(path)
    rows = extract_features_batch(candidates)

    # Display columns only (drop the raw _profile/_career/_skills/_redrob)
    display_cols = [
        "candidate_id", "current_title", "yoe", "title_class",
        "services_fraction", "consulting_only",
        "has_embeddings", "has_vectordb", "has_eval", "has_python",
        "musthave_count", "has_recsys", "has_production",
        "location_fit", "notice_days",
    ]
    clean = [{k: r[k] for k in display_cols} for r in rows]
    df = pd.DataFrame(clean)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 28)
    print(df.to_string(index=False))

    # Spotlight check
    print("\n--- SPOTLIGHT: CAND_0000031 (should be gold standard) ---")
    r31 = next(r for r in rows if r["candidate_id"] == "CAND_0000031")
    for k in display_cols:
        print(f"  {k:<22}: {r31[k]}")
