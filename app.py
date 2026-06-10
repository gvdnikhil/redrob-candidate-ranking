"""
app.py - Streamlit sandbox for the Redrob candidate ranking system.

Accepts up to 100 candidates as JSON input, runs the full ranking pipeline
end-to-end (embed + BM25 + RRF + composite score + reasoning), and outputs
a ranked table with downloadable CSV.

Deploy: streamlit run app.py
"""

import sys
sys.path.insert(0, ".")

import json
import time
import csv
import io

import streamlit as st
import pandas as pd

# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🔍",
    layout="wide",
)

st.title("Redrob Intelligent Candidate Ranker")
st.caption("Hackathon sandbox — accepts up to 100 candidates, ranks against Senior AI Engineer JD")

# -------------------------------------------------------------------------
# Sidebar: data source
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("Input")
    source = st.radio(
        "Candidate source",
        ["Use pre-loaded sample (50 candidates)", "Upload your own JSON"],
    )

    uploaded = None
    if source == "Upload your own JSON":
        uploaded = st.file_uploader(
            "Upload candidates JSON (array format, max 100)",
            type=["json"],
        )
        st.caption("Format: JSON array of candidate objects matching the Redrob schema.")

    st.markdown("---")
    st.header("About")
    st.markdown("""
**Pipeline:**
1. Build candidate text (headline + summary + career + skills)
2. Dense embedding via `static-retrieval-mrl-en-v1`
3. BM25 sparse retrieval
4. Reciprocal Rank Fusion
5. Structured scoring (career quality, must-have evidence, signals)
6. Deterministic slot-based reasoning

**Weights (from config.py):**
- Semantic fit: 35%
- Career quality: 30%
- Must-have evidence: 25%
- Location fit: 10%
""")

# -------------------------------------------------------------------------
# Load candidates
# -------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_sample():
    with open("data/sample_candidates.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_uploaded(file_bytes):
    data = json.loads(file_bytes.decode("utf-8"))
    if isinstance(data, list):
        return data[:100]
    return [data]

# -------------------------------------------------------------------------
# Run ranking (cached on the candidate list hash)
# -------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_ranking(candidates_json_str: str):
    """Run full pipeline on a JSON string of candidates. Cached."""
    from src.parsing import load_candidates
    from src.retrieval import score_sample
    from src.scoring import score_candidate, score_batch
    from src.reasoning import generate_reasoning, validate_no_hallucination

    candidates = json.loads(candidates_json_str)
    n = len(candidates)

    t0 = time.time()
    semantic_fits = score_sample(candidates, show_progress=False)
    t_embed = time.time() - t0

    t0 = time.time()
    results = score_batch(candidates, semantic_fits=semantic_fits)
    t_score = time.time() - t0

    # Build output rows
    rows = []
    by_id = {c["candidate_id"]: c for c in candidates}
    actual_n = min(100, len(results))

    for rank_idx, r in enumerate(results[:actual_n], 1):
        cid = r["candidate_id"]
        c   = by_id[cid]
        reasoning = generate_reasoning(rank_idx, c, r)
        rows.append({
            "rank"          : rank_idx,
            "candidate_id"  : cid,
            "score"         : round(r["final_score"], 4),
            "title"         : r["current_title"],
            "yoe"           : r["yoe"],
            "semantic_fit"  : round(r["semantic_fit"], 3),
            "career_quality": round(r["career_quality"], 3),
            "musthave"      : round(r["musthave_evidence"], 3),
            "availability"  : r["availability"],
            "penalty"       : round(r["penalty"], 3),
            "plausibility"  : r["plausibility"],
            "reasoning"     : reasoning,
            "_feat"         : r["_feat"],
        })

    return rows, t_embed, t_score

# -------------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------------
run_btn = st.button("Run Ranking", type="primary", use_container_width=True)

if run_btn or source == "Use pre-loaded sample (50 candidates)":

    # Load candidates
    with st.spinner("Loading candidates..."):
        if source == "Use pre-loaded sample (50 candidates)" or uploaded is None:
            try:
                candidates = load_sample()
                st.info(f"Using pre-loaded sample: {len(candidates)} candidates")
            except FileNotFoundError:
                st.error("data/sample_candidates.json not found. Upload a file instead.")
                st.stop()
        else:
            candidates = load_uploaded(uploaded.read())
            st.info(f"Uploaded: {len(candidates)} candidates")

    if len(candidates) > 100:
        st.warning("Trimming to first 100 candidates (sandbox limit).")
        candidates = candidates[:100]

    # Run pipeline
    with st.spinner(f"Running pipeline on {len(candidates)} candidates..."):
        t_wall = time.time()
        try:
            rows, t_embed, t_score = run_ranking(json.dumps(candidates))
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)
            st.stop()
        t_total = time.time() - t_wall

    # Timing
    col1, col2, col3 = st.columns(3)
    col1.metric("Embed + BM25", f"{t_embed:.1f}s")
    col2.metric("Scoring", f"{t_score:.1f}s")
    col3.metric("Total", f"{t_total:.1f}s")

    st.success(f"Ranked {len(rows)} candidates in {t_total:.1f}s")

    # -------------------------------------------------------------------------
    # Results table
    # -------------------------------------------------------------------------
    st.subheader("Ranked Results")

    display_df = pd.DataFrame([{
        "Rank"      : r["rank"],
        "ID"        : r["candidate_id"],
        "Title"     : r["title"],
        "YOE"       : r["yoe"],
        "Score"     : r["score"],
        "Semantic"  : r["semantic_fit"],
        "Career"    : r["career_quality"],
        "MustHave"  : r["musthave"],
        "Avail"     : r["availability"],
        "Penalty"   : r["penalty"],
        "Plausib."  : r["plausibility"],
    } for r in rows])

    st.dataframe(
        display_df.style
            .background_gradient(subset=["Score"], cmap="Greens")
            .background_gradient(subset=["Penalty"], cmap="Reds_r"),
        use_container_width=True,
        height=500,
    )

    # -------------------------------------------------------------------------
    # Expandable reasoning per candidate
    # -------------------------------------------------------------------------
    st.subheader("Reasoning (top 20)")
    for r in rows[:20]:
        feat = r["_feat"]
        flags = []
        if feat.get("has_embeddings"): flags.append("embeddings")
        if feat.get("has_vectordb"):   flags.append("vectorDB")
        if feat.get("has_eval"):       flags.append("eval-fw")
        if feat.get("has_python"):     flags.append("python")
        if feat.get("has_recsys"):     flags.append("recsys")
        flag_str = " | ".join(flags) if flags else "none"

        with st.expander(f"#{r['rank']} {r['candidate_id']} — {r['title']} (score={r['score']})"):
            st.write(f"**Reasoning:** {r['reasoning']}")
            cols = st.columns(4)
            cols[0].metric("Semantic fit", r["semantic_fit"])
            cols[1].metric("Career quality", r["career_quality"])
            cols[2].metric("Must-have", r["musthave"])
            cols[3].metric("Availability", r["availability"])
            st.caption(f"Must-have flags: {flag_str} | "
                       f"Title class: {feat.get('title_class','?')} | "
                       f"Services fraction: {feat.get('services_fraction', 0):.0%} | "
                       f"Plausibility: {r['plausibility']}")

    # -------------------------------------------------------------------------
    # Download CSV
    # -------------------------------------------------------------------------
    st.subheader("Download submission CSV")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    for r in rows:
        writer.writerow({
            "candidate_id": r["candidate_id"],
            "rank"        : r["rank"],
            "score"       : r["score"],
            "reasoning"   : r["reasoning"],
        })

    st.download_button(
        label="Download submission.csv",
        data=output.getvalue().encode("utf-8"),
        file_name="submission.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Validate on the fly
    st.caption("Format check:")
    ids    = [r["candidate_id"] for r in rows]
    scores = [r["score"] for r in rows]
    checks = {
        "100 rows"              : len(rows) == 100 or len(candidates) < 100,
        "Unique IDs"            : len(set(ids)) == len(ids),
        "Score non-increasing"  : all(scores[i] >= scores[i+1] - 1e-9 for i in range(len(scores)-1)),
        "All reasoning filled"  : all(r["reasoning"].strip() for r in rows),
    }
    for check, ok in checks.items():
        if ok:
            st.success(f"PASS  {check}")
        else:
            st.error(f"FAIL  {check}")
