"""
config.py — single source of truth for all weights, thresholds, and regex patterns.
Nothing else in the codebase should have magic numbers or hard-coded strings.
"""

# ---------------------------------------------------------------------------
# Scoring weights (must sum considerations: these are relative, not strict sum=1)
# ---------------------------------------------------------------------------
W_SEM    = 0.35   # semantic_fit from dense+sparse retrieval
W_CAREER = 0.30   # career_quality: product-company ML years, shipped systems
W_MUST   = 0.25   # musthave_evidence: embeddings / vectorDB / eval-fw / python
W_LOC    = 0.10   # location_fit: India-based or willing to relocate

# ---------------------------------------------------------------------------
# Penalty values (subtracted from fit score before multiplying by availability)
# ---------------------------------------------------------------------------
PENALTY_BAD_TITLE      = 0.25   # current_title is clearly non-engineering (Marketing, HR, etc.)
PENALTY_CONSULTING_ONLY= 0.15   # entire career is at pure services firms
PENALTY_RESEARCH_ONLY  = 0.20   # no production deployments (pure academic/research)
PENALTY_CV_SPEECH      = 0.15   # primary expertise is CV/Speech/Robotics, not NLP/IR
PENALTY_NON_INDIA      = 0.10   # outside India + willing_to_relocate = False
PENALTY_SHORT_TENURE   = 0.05   # job-hopping pattern (avg tenure < 12 months)

# ---------------------------------------------------------------------------
# Availability multiplier bounds
# ---------------------------------------------------------------------------
AVAIL_MIN = 0.35
AVAIL_MAX = 1.15

# Recency thresholds (days inactive)
INACTIVE_SEVERE  = 180   # deduct 0.35
INACTIVE_HEAVY   = 90    # deduct 0.20
INACTIVE_MILD    = 45    # deduct 0.10

# Response rate thresholds
RESPONSE_RATE_LOW  = 0.15   # deduct 0.20
RESPONSE_RATE_HIGH = 0.60   # add 0.08

# Notice period thresholds (days)
NOTICE_LONG = 90    # deduct 0.10
NOTICE_SHORT = 30   # add 0.08

# ---------------------------------------------------------------------------
# Consistency / honeypot thresholds
# ---------------------------------------------------------------------------
CONSISTENCY_YOE_DIFF_MONTHS   = 30    # allowed gap between claimed yoe and career sum
CONSISTENCY_MIN_PLAUSIBILITY  = 0.0   # floor; profiles below 0.3 are likely honeypots

# Deductions per consistency failure
DEDUCT_YOE_MISMATCH     = 0.30
DEDUCT_DATE_ORDER       = 0.25
DEDUCT_FUTURE_DATE      = 0.20
DEDUCT_EXPERT_ZERO      = 0.15   # per expert skill with duration_months=0
DEDUCT_MULTI_EXPERT     = 0.20   # ≥3 expert skills with 0 endorsements + 0 assessments
DEDUCT_EDUCATION_DATE   = 0.20
DEDUCT_MEGA_TENURE      = 0.25   # single role > 360 months

# ---------------------------------------------------------------------------
# Company classifiers
# ---------------------------------------------------------------------------

# Pure-services firms: entire career here → consulting_only penalty
SERVICES_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "mindtree", "tech mahindra", "hexaware", "mphasis",
    "l&t infotech", "ltimindtree", "birlasoft", "niit technologies",
}

# Target locations for the role (Pune/Noida preferred; broader India acceptable)
TARGET_LOCATIONS = {
    "pune", "noida", "hyderabad", "bangalore", "bengaluru",
    "mumbai", "delhi", "gurgaon", "gurugram", "chennai",
}

# ---------------------------------------------------------------------------
# Title classifier regex patterns
# ---------------------------------------------------------------------------
# "bad" titles = clearly non-engineering roles (the trap keyword-stuffers)
BAD_TITLE_PATTERNS = [
    r"\bmarketing\b", r"\bhr\b", r"\bhuman resources\b",
    r"\bcontent writ", r"\bgraphic design", r"\bcivil eng",
    r"\bmechanical eng", r"\baccountant\b", r"\baccounting\b",
    r"\bsales exec", r"\bcustomer support\b", r"\boperations manager\b",
    r"\bproject manager\b",   # generic PM without eng context
]

# "good" titles = engineering / ML / AI roles
GOOD_TITLE_PATTERNS = [
    r"\bengineer\b", r"\bml\b", r"\bai\b", r"\bdata scientist\b",
    r"\bdata engineer\b", r"\bsearch\b", r"\bnlp\b", r"\bscientist\b",
    r"\barchitect\b", r"\bdeveloper\b", r"\bdevops\b", r"\bsre\b",
    r"\bplatform\b", r"\bbackend\b", r"\bfrontend\b", r"\bfull.?stack\b",
    r"\bjava\b", r"\b\.net\b", r"\bcloud\b",
]

# ---------------------------------------------------------------------------
# Must-have evidence patterns (searched in descriptions + skill names)
# ---------------------------------------------------------------------------

# 1. Embeddings-based retrieval
PATTERN_EMBEDDINGS = (
    r"embedding|sentence.transformer|bge\b|e5\b|bi.encoder|"
    r"openai embed|dense retrieval|semantic search|vector embed"
)

# 2. Vector DB / hybrid search
PATTERN_VECTORDB = (
    r"pinecone|weaviate|qdrant|milvus|opensearch|elasticsearch|faiss|"
    r"vector.?db|vector.?database|vector search|hybrid search|"
    r"milvus|annoy|hnsw"
)

# 3. Ranking evaluation frameworks
PATTERN_EVAL = (
    r"\bndcg\b|\bmrr\b|\bmap\b|a/b test|offline.online|"
    r"learning.to.rank|\bltr\b|eval.?framework|ranking eval|"
    r"precision@|recall@|mean average"
)

# 4. Python (any evidence — include common Python ML ecosystem libs)
PATTERN_PYTHON = (
    r"\bpython\b|scikit.learn|sklearn|pytorch|tensorflow|keras|"
    r"pandas\b|numpy\b|pyspark|hugging.?face|sentence.transformer|"
    r"mlflow|langchain|fastapi|flask\b|django\b"
)

# 5. Shipped-to-production signal
PATTERN_PRODUCTION = (
    r"production|deployed|shipped|real user|at scale|"
    r"revenue|launched|live system|serving"
)

# 6. Recommendation / search / ranking systems (career evidence)
PATTERN_RECSYS = (
    r"recommendation|search engine|ranker|ranking system|"
    r"retrieval system|information retrieval|candidate matching|"
    r"job matching|talent matching"
)

# 7. LLM fine-tuning (nice-to-have)
PATTERN_FINETUNE = r"lora|qlora|peft|fine.tun|sft\b|rlhf"

# ---------------------------------------------------------------------------
# Cross-encoder model (for optional reranking at rank time)
# ---------------------------------------------------------------------------
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RERANKER_SHORTLIST_K = 150   # rerank only top-K from RRF

# ---------------------------------------------------------------------------
# Bi-encoder model (for precompute phase)
# CPU-optimised: static-retrieval-mrl-en-v1 does token-vector averaging
# (no neural forward pass) -> 3000+ candidates/sec vs 1/sec for bge-base.
# Quality is sufficient since BM25 + structured scoring cover 65% of the score.
# ---------------------------------------------------------------------------
BIENCODER_MODEL = "sentence-transformers/static-retrieval-mrl-en-v1"
EMBEDDING_DIM   = 1024

# ---------------------------------------------------------------------------
# BM25 / RRF
# ---------------------------------------------------------------------------
RRF_K = 60   # standard RRF constant

# ---------------------------------------------------------------------------
# Rank-time compute budget
# ---------------------------------------------------------------------------
RANK_TIME_LIMIT_SECONDS = 270   # warn at this threshold (5min = 300s, leave 30s margin)
