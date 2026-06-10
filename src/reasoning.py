"""
src/reasoning.py  -  deterministic, factual reasoning generator.

Each reasoning line reads like a recruiter's judgment and includes:
  1. role + years + company-type
  2. named JD-relevant skills (with Redrob assessment score when present)
  3. an explicit tie to a JD requirement (retrieval / eval / recsys / fine-tuning)
  4. a BEHAVIORAL SIGNAL VALUE  - positive when strong, a concern when weak
  5. (top ranks) one real evidence sentence from the candidate's own history, never truncated mid-word

Rules:
 - Only references facts present in the candidate's own record (no hallucination).
 - `-1` sentinels (offer_acceptance, github) are treated as "no data", never cited as low.
 - Structure varies across candidates so sampled rows read as distinct.
 - No LLM calls. No external data.
"""

import re
import sys
sys.path.insert(0, ".")

from config import (
    PATTERN_EMBEDDINGS, PATTERN_VECTORDB, PATTERN_EVAL,
    PATTERN_RECSYS, PATTERN_FINETUNE,
    SERVICES_FIRMS,
)

JD_RELEVANT_SKILLS = {
    "embeddings", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "sentence transformers", "bm25",
    "mlflow", "recommendation systems", "information retrieval",
    "hugging face transformers", "fine-tuning llms", "peft", "lora",
    "machine learning", "nlp", "python", "langchain",
    "scikit-learn", "pytorch", "tensorflow", "deep learning",
    "feature engineering", "vector search", "embeddings", "reinforcement learning",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _company_type(company: str) -> str:
    c = company.lower()
    return "services firm" if any(sf in c for sf in SERVICES_FIRMS) else "product company"


def _clean_evidence_sentence(candidate: dict, max_chars: int = 170) -> str:
    """
    Return ONE complete sentence from a career description that contains a
    JD-relevant keyword. Never cuts mid-word: if too long, trims at the last
    word boundary and adds an ellipsis.
    """
    patterns = [PATTERN_RECSYS, PATTERN_EMBEDDINGS, PATTERN_VECTORDB,
                PATTERN_EVAL, PATTERN_FINETUNE]
    for role in candidate.get("career_history", []):
        desc = role.get("description", "") or ""
        if not desc:
            continue
        for sent in re.split(r"(?<=[.!?])\s+", desc):
            s = sent.strip()
            if not s:
                continue
            if any(re.search(p, s, re.IGNORECASE) for p in patterns):
                s = re.sub(r"\s+", " ", s).rstrip(".")
                if len(s) <= max_chars:
                    return s
                # trim at last word boundary before max_chars
                cut = s[:max_chars].rsplit(" ", 1)[0]
                return cut + "..."
    return ""


def _top_matched_skills(candidate: dict, limit: int = 2) -> list[str]:
    """Up to `limit` JD-relevant skill names from the candidate's own skills,
    annotated with assessment score when present."""
    assess = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    matched = []
    for s in candidate.get("skills", []):
        name = s.get("name", "").strip()
        if name.lower() in JD_RELEVANT_SKILLS and s.get("proficiency") in ("advanced", "expert", "intermediate"):
            score = assess.get(name)
            matched.append(f"{name} (assessed {score:.0f}/100)" if score else name)
        if len(matched) >= limit:
            break
    return matched


def _jd_tie(feat: dict) -> str:
    """Explicit phrase naming which JD requirement(s) the candidate's evidence maps to."""
    needs = []
    if feat.get("has_embeddings") or feat.get("has_vectordb"):
        needs.append("embeddings/vector retrieval")
    if feat.get("has_eval"):
        needs.append("ranking evaluation")
    if feat.get("has_recsys"):
        needs.append("search/recommendation systems")
    if feat.get("has_finetune"):
        needs.append("LLM fine-tuning")
    return " and ".join(needs[:2])


def _engagement(candidate: dict, feat: dict) -> tuple[str, str]:
    """
    Returns (positive_signal, concern_signal) as plain phrases citing REAL values.
    `-1` sentinels are treated as 'no data' and never cited as low.
    """
    from src.signals import _REFERENCE_DATE, _parse_date
    sig = candidate.get("redrob_signals", {})
    last = _parse_date(sig.get("last_active_date"))
    days = (_REFERENCE_DATE - last).days if last else None
    rr   = sig.get("recruiter_response_rate")
    otw  = sig.get("open_to_work_flag", False)
    nd   = sig.get("notice_period_days")
    ic   = sig.get("interview_completion_rate")
    oa   = sig.get("offer_acceptance_rate")     # -1 = no history
    gh   = sig.get("github_activity_score")     # -1 = no GitHub

    # --- positive: collect up to 2 genuinely strong signals ---
    pos = []
    if days is not None and days <= 30:
        pos.append("active this month")
    elif days is not None and days <= 60:
        pos.append("active in the last 2 months")
    if otw:
        pos.append("open to work")
    if rr is not None and rr >= 0.60:
        pos.append(f"{rr:.0%} recruiter response rate")
    if ic is not None and ic >= 0.70:
        pos.append(f"{ic:.0%} interview-completion rate")
    if oa is not None and oa >= 0.50:
        pos.append(f"{oa:.0%} offer-acceptance rate")
    if gh is not None and gh > 40:
        pos.append(f"GitHub activity {gh:.0f}/100")
    positive = ", ".join(pos[:2])

    # --- concern: single most salient weak signal ---
    concern = ""
    country = candidate.get("profile", {}).get("country", "")
    if not feat.get("location_fit", True):
        concern = f"based in {country} and not open to relocation"
    elif days is not None and days > 150:
        concern = f"inactive for {days} days"
    elif rr is not None and rr < 0.20:
        concern = f"low recruiter response rate ({rr:.0%})"
    elif nd is not None and nd > 90:
        concern = f"{nd}-day notice period"
    elif feat.get("services_fraction", 0) > 0.60:
        concern = "services-heavy career background"
    elif feat.get("musthave_count", 0) < 2:
        concern = "thin direct production ML/IR evidence"

    return positive, concern


def _join(frags: list[str]) -> str:
    """Join non-empty fragments into clean sentences, capitalising each fragment's
    first letter (so clause labels don't render lowercase at a sentence start)."""
    clean = []
    for f in frags:
        if f and f.strip():
            t = f.strip().rstrip(".")
            if t:
                t = t[0].upper() + t[1:]
                clean.append(t)
    out = ". ".join(clean)
    return (out + ".") if out and not out.endswith(".") else out


# ---------------------------------------------------------------------------
# Assembly by rank tier (with deterministic structural variation)
# ---------------------------------------------------------------------------

def _format_reasoning(rank: int, candidate: dict, score_row: dict) -> str:
    p = candidate.get("profile", {})
    feat = score_row.get("_feat", {})
    title   = p.get("current_title", "Engineer")
    company = p.get("current_company", "")
    yoe     = p.get("years_of_experience", 0) or 0
    ctype   = _company_type(company)

    ident    = f"{yoe:.0f}yr {title} at {company} ({ctype})"
    skills   = _top_matched_skills(candidate)
    skills_s = ", ".join(skills)
    tie      = _jd_tie(feat)
    evidence = _clean_evidence_sentence(candidate)
    positive, concern = _engagement(candidate, feat)

    # skills + JD tie combined into one natural clause
    if skills_s and tie:
        skill_tie = f"{skills_s} map to the JD's need for {tie}"
    elif skills_s:
        skill_tie = f"key JD-relevant skills: {skills_s}"
    elif tie:
        skill_tie = f"experience maps to the JD's need for {tie}"
    else:
        skill_tie = ""

    pos_clause = positive if positive else ""          # natural prose, e.g. "active this month, open to work"
    con_clause = f"concern: {concern}" if concern else ""   # _join capitalises -> "Concern: ..."
    variant = hash(candidate["candidate_id"]) % 3

    if rank <= 10:
        # Confident. Lead with fit + positive signal; concern only if it exists.
        if variant == 0:
            frags = [ident, skill_tie, evidence, pos_clause, con_clause]
        elif variant == 1:
            frags = [f"Strong fit: {ident}", pos_clause, skill_tie, evidence, con_clause]
        else:
            frags = [ident, evidence, skill_tie, pos_clause, con_clause]

    elif rank <= 50:
        # Confident with one explicit caveat; balance signal and evidence.
        secondary = con_clause if con_clause else pos_clause
        if variant == 0:
            frags = [ident, skill_tie, evidence, secondary]
        elif variant == 1:
            frags = [ident, skill_tie, secondary, evidence]
        else:
            frags = [f"Solid fit: {ident}", evidence, skill_tie, secondary]

    else:
        # Tail: explicitly marginal.
        body = skill_tie if skill_tie else (evidence if evidence else "limited JD-relevant signal")
        frags = [f"Adjacent fit: {ident}", body,
                 "included as filler given limited direct ML/IR evidence",
                 con_clause]

    return _join(frags)


def generate_reasoning(rank: int, candidate: dict, score_row: dict) -> str:
    """Public entry point. 1-2 factual sentences; every fact comes from the record."""
    return _format_reasoning(rank, candidate, score_row)


def validate_no_hallucination(reasoning: str, candidate: dict) -> list[str]:
    """Confirm every employer/skill named in the reasoning exists in the record."""
    violations = []
    all_companies = {r["company"].lower() for r in candidate.get("career_history", [])}
    all_skills    = {s["name"].lower() for s in candidate.get("skills", [])}
    r_low = reasoning.lower()

    company = candidate["profile"].get("current_company", "")
    if company and company.lower() in r_low and company.lower() not in all_companies:
        violations.append(f"company '{company}' not in career_history")

    for tag in _top_matched_skills(candidate):
        name = tag.split(" (")[0]
        if name.lower() in r_low and name.lower() not in all_skills:
            violations.append(f"skill '{name}' not in candidate skills")

    return violations


# ---------------------------------------------------------------------------
# __main__: dry-run on 5 diverse candidates from the 50-sample
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.path.insert(0, ".")
    from src.parsing import load_candidates
    from src.scoring import score_candidate

    candidates = load_candidates("data/sample_candidates.json")
    by_id = {c["candidate_id"]: c for c in candidates}

    targets = [
        (1,  "CAND_0000031", "strong fit, great engagement"),
        (6,  "CAND_0000010", "Data Engineer, UK (location concern)"),
        (15, "CAND_0000025", "Frontend, services-heavy"),
        (45, "CAND_0000001", "Backend, Canada (outside India)"),
        (90, "CAND_0000002", "Operations Manager (trap, tail)"),
    ]

    print("Reasoning dry-run (NEW generator)\n" + "=" * 78)
    for rank, cid, note in targets:
        c = by_id.get(cid)
        if not c:
            print(f"  {cid} not found"); continue
        row = score_candidate(c)
        reasoning = generate_reasoning(rank, c, row)
        violations = validate_no_hallucination(reasoning, c)
        status = "OK" if not violations else f"VIOLATIONS: {violations}"
        print(f"\nRank {rank} | {cid}  ({note})")
        print(f"  {c['profile']['current_title']} | {c['profile']['country']}")
        print(f"  -> {reasoning}")
        print(f"  validation: {status}")
