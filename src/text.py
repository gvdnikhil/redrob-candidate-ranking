"""
src/text.py — builds canonical text representations for embedding and BM25.

build_candidate_text(c) → one rich string per candidate
build_jd_text(path)     → the job description as a single string
"""

import pathlib


def build_candidate_text(c: dict) -> str:
    """
    Builds a single string capturing the most signal-rich parts of a candidate record.
    Optimised for retrieval: career descriptions carry the most weight, then title/skills.
    """
    parts = []
    p = c.get("profile", {})

    # Headline and summary
    if p.get("headline"):
        parts.append(p["headline"])
    if p.get("summary"):
        parts.append(p["summary"])

    # Career history — title, company, and the full description text
    for role in c.get("career_history", []):
        title   = role.get("title", "")
        company = role.get("company", "")
        desc    = role.get("description", "").strip()
        industry = role.get("industry", "")
        role_text = f"{title} at {company} ({industry}): {desc}" if desc else f"{title} at {company}"
        parts.append(role_text)

    # Skills — name + proficiency (helps BM25 match skill terms)
    skill_tokens = []
    for s in c.get("skills", []):
        name = s.get("name", "")
        prof = s.get("proficiency", "")
        if name:
            skill_tokens.append(f"{name} ({prof})" if prof else name)
    if skill_tokens:
        parts.append("Skills: " + ", ".join(skill_tokens))

    # Education — field of study (helps match "Machine Learning", "NLP", etc.)
    for edu in c.get("education", []):
        field = edu.get("field_of_study", "")
        degree = edu.get("degree", "")
        if field:
            parts.append(f"{degree} in {field}")

    # Certifications
    for cert in c.get("certifications", []):
        parts.append(cert.get("name", ""))

    return " ".join(filter(None, parts))


def build_jd_text(jd_path: str = None) -> str:
    """
    Returns the job description as a plain string.
    The JD is provided as a PDF; text is extracted and embedded here directly
    so the pipeline has no dependency on a PDF parser.
    """
    # Full text extracted from jod_description.pdf
    return """
    Senior AI Engineer Founding Team. Company: Redrob AI Series A AI-native talent intelligence platform.
    Location: Pune Noida India Hybrid. Open to relocation candidates from Tier-1 Indian cities.
    Experience Required: 5 to 9 years applied machine learning AI product companies.

    We need someone comfortable with deep technical depth in modern ML systems: embeddings retrieval
    ranking LLMs fine-tuning. Scrappy product-engineering attitude willing to ship a working ranker.

    What you would actually be doing: own the intelligence layer of Redrob product.
    Ranking retrieval matching systems. Candidate job description matching at scale.
    Ship v2 ranking system embeddings hybrid retrieval LLM-based re-ranking.
    Set up evaluation infrastructure offline benchmarks online A/B testing recruiter-feedback loops.
    Mentor engineers grow team.

    Things you absolutely need:
    Production experience embeddings-based retrieval systems sentence-transformers OpenAI embeddings
    BGE E5 deployed to real users embedding drift index refresh retrieval-quality regression production.
    Production experience vector databases hybrid search infrastructure Pinecone Weaviate Qdrant Milvus
    OpenSearch Elasticsearch FAISS operational experience.
    Strong Python code quality.
    Hands-on experience designing evaluation frameworks for ranking systems NDCG MRR MAP
    offline-to-online correlation A/B test interpretation.

    Things we would like you to have:
    LLM fine-tuning experience LoRA QLoRA PEFT.
    Experience learning-to-rank models XGBoost-based neural.
    Prior exposure HR-tech recruiting tech marketplace products.
    Background distributed systems large-scale inference optimization.
    Open-source contributions AI ML space.

    Things we explicitly do NOT want:
    Title-chasers switching companies every 1.5 years.
    Framework enthusiasts LangChain tutorials demo projects.
    People who have only worked at consulting firms TCS Infosys Wipro Accenture Cognizant Capgemini
    entire career services background.
    People whose primary expertise is computer vision speech robotics without significant NLP IR exposure.
    People whose work has been entirely on closed-source proprietary systems 5 plus years.

    Ideal candidate: 6-8 years total experience 4-5 years applied ML AI roles at product companies not pure services.
    Shipped at least one end-to-end ranking search recommendation system real users meaningful scale.
    Strong opinions about retrieval hybrid versus dense evaluation offline versus online LLM integration
    fine-tune versus prompt can defend with reference to systems actually built.
    Located in or willing to relocate to Noida or Pune.
    Active on Redrob platform clear signal being in job market.

    Notice period: sub-30-day ideal can buy out up to 30 days.
    Hybrid work Pune Noida offices Hyderabad Mumbai Delhi NCR Bangalore welcome to apply.
    """


# ---------------------------------------------------------------------------
# __main__: dry-run — print texts for two contrasting candidates
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.parsing import load_candidates

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_candidates.json"
    candidates = load_candidates(path)

    # Index by ID for easy lookup
    by_id = {c["candidate_id"]: c for c in candidates}

    jd = build_jd_text()
    print("=" * 70)
    print("JD TEXT (first 400 chars):")
    print(jd[:400].strip())

    for cid in ["CAND_0000031", "CAND_0000002"]:
        c = by_id.get(cid)
        if not c:
            continue
        text = build_candidate_text(c)
        print("\n" + "=" * 70)
        print(f"CANDIDATE TEXT: {cid} — {c['profile']['current_title']}")
        print(f"Length: {len(text)} chars")
        print("-" * 40)
        print(text[:600])
        print("...")
