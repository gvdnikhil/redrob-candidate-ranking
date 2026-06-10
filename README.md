# Redrob Intelligent Candidate Ranking

Hackathon submission for the Redrob "Intelligent Candidate Discovery & Ranking Challenge".  
Ranks 100,000 candidate profiles against a Senior AI Engineer JD.  
Output: top 100 best-fit candidates with factual reasoning, in under 2 minutes on CPU.

---

## Quick Start

```bash
# 1. Create venv and install
py -3 -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Place data files
#    data/sample_candidates.json   (50-candidate sample)
#    data/candidates.jsonl         (full 100k pool)

# 3. Precompute artifacts (run once — ~3.5 min on CPU)
python precompute.py --candidates data/candidates.jsonl

# 4. Rank and produce submission
python rank.py --candidates data/candidates.jsonl --out submission.csv

# Explain a single candidate score
python rank.py --candidates data/candidates.jsonl --explain CAND_0046064

# Validate format
python validate_submission.py submission.csv --candidates data/candidates.jsonl
```

---

## System Architecture

The pipeline has two phases. Phase A runs offline (no time limit, GPU allowed).
Phase B runs at submission time (CPU only, under 5 minutes, no network).

```mermaid
flowchart TD
    subgraph A["Phase A — precompute.py (offline, run once)"]
        direction TB
        RAW["candidates.jsonl\n100,000 profiles"]
        TXT["src/text.py\nbuild_candidate_text()\nheadline + summary +\ncareer descriptions + skills"]
        EMB["sentence-transformers\nstatic-retrieval-mrl-en-v1\n3,200 candidates/sec on CPU"]
        BM25["bm25s\nBM25 Index\nexact term matching"]
        FEAT["src/features.py\nextract_features()\nstructured columns per candidate"]

        RAW --> TXT
        TXT --> EMB
        TXT --> BM25
        TXT --> FEAT

        EMB -->|"390 MB"| ENPY["artifacts/\ncandidate_embeddings.npy\njd_embedding.npy"]
        BM25 --> BIDX["artifacts/\nbm25_index/"]
        FEAT -->|"1.2 MB"| PARQ["artifacts/\nfeatures.parquet"]
    end

    subgraph B["Phase B — rank.py (CPU only, &lt;2 min, no network)"]
        direction TB
        LOAD["Load artifacts\n1.2s"]
        DENSE["Dense Score\nnumpy matmul\njd_vec · all_vecs\n0.5s for 100k"]
        SPARSE["BM25 Score\nquery JD tokens\nagainst index"]
        RRF["Reciprocal Rank Fusion\nscore = Σ 1 / (60 + rank_i)\nno training needed"]
        SCORE["src/scoring.py\nComposite Score\n100s for 100k"]
        TOP["Top 100 candidates"]
        REASON["src/reasoning.py\nDeterministic reasoning\nno LLM, no hallucination"]
        CSV["submission.csv\n100 rows\ncandidate_id, rank, score, reasoning"]

        LOAD --> DENSE
        LOAD --> SPARSE
        DENSE --> RRF
        SPARSE --> RRF
        RRF -->|"semantic_fit"| SCORE
        LOAD --> SCORE
        SCORE --> TOP
        TOP --> REASON
        REASON --> CSV
    end

    A --> B
```

---

## Composite Scoring Formula

This is the heart of the system. Every number comes from `config.py`.

```mermaid
flowchart LR
    subgraph INPUTS["Inputs to scorer"]
        SF["semantic_fit\n0.0 - 1.0\nfrom RRF"]
        CQ["career_quality\n0.0 - 1.0\nproduct-co ML months\n+ title class\n+ shipped-to-prod evidence"]
        ME["musthave_evidence\n0.0 - 1.0\n4 hard must-haves:\nembeddings / vectorDB\neval-fw / python"]
        LF["location_fit\n0 or 1\nIndia OR willing to relocate"]
    end

    subgraph WEIGHTS["Weighted fit"]
        FIT["fit =\n0.35 x semantic_fit\n+ 0.30 x career_quality\n+ 0.25 x musthave_evidence\n+ 0.10 x location_fit"]
    end

    subgraph MODIFIERS["Modifiers"]
        AV["availability_multiplier\n0.35 - 1.15\nfrom 23 redrob signals"]
        PEN["penalty_total\nsubtracted from fit\nbad_title: -0.25\nconsulting_only: -0.15\nnon_india: -0.10"]
        PL["plausibility_score\n0.0 - 1.0\nhoneypot detection\n7 consistency checks"]
    end

    FINAL["final_score =\nmax(0, fit x availability - penalty)\nx plausibility"]

    SF --> FIT
    CQ --> FIT
    ME --> FIT
    LF --> FIT
    FIT --> FINAL
    AV --> FINAL
    PEN --> FINAL
    PL --> FINAL
```

---

## Availability Multiplier

How behavioral signals from the platform modify the score.

```mermaid
flowchart TD
    BASE["Start: multiplier = 1.0"]

    BASE --> R1{"last_active_date\nhow many days ago?"}
    R1 -->|"&gt; 180 days"| D1["-0.35\neffectively gone"]
    R1 -->|"90-180 days"| D2["-0.20\nprobably passive"]
    R1 -->|"45-90 days"| D3["-0.10\nmild penalty"]
    R1 -->|"&lt; 45 days"| D4["no change"]

    BASE --> R2{"open_to_work_flag?"}
    R2 -->|"true"| U1["+0.10"]
    R2 -->|"false"| U2["no change"]

    BASE --> R3{"recruiter_response_rate"}
    R3 -->|"&lt; 0.15"| D5["-0.20\nred flag"]
    R3 -->|"&gt; 0.60"| U3["+0.08"]

    BASE --> R4{"notice_period_days"}
    R4 -->|"&gt; 90 days"| D6["-0.10"]
    R4 -->|"&lt;= 30 days"| U4["+0.08\njoin quickly"]

    BASE --> R5{"github_activity_score"}
    R5 -->|"&gt; 20"| U5["+0.05\nactive coder"]
    R5 -->|"-1 = no GitHub"| U6["no change"]

    CLAMP["CLAMP to 0.35 - 1.15"]

    D1 & D2 & D3 & D4 & U1 & U2 & U3 & D5 & D6 & U4 & U5 & U6 --> CLAMP
```

---

## Honeypot / Plausibility Detection

How suspicious profiles are caught without a hardcoded ID list.

```mermaid
flowchart TD
    CAND["candidate record"]
    START["plausibility = 1.0"]

    CAND --> START

    START --> C1{"claimed yoe vs\nsum of career months\ndiff &gt; 30 months?"}
    C1 -->|yes| P1["-0.30"]

    START --> C2{"any role\nstart_date &gt; end_date?"}
    C2 -->|yes| P2["-0.25"]

    START --> C3{"any future date\nbeyond pool max?"}
    C3 -->|yes| P3["-0.20"]

    START --> C4{"skill proficiency = expert\nAND duration_months = 0?"}
    C4 -->|yes| P4["-0.15 per skill"]

    START --> C5{"3+ expert skills with\n0 endorsements AND\n0 assessment scores?"}
    C5 -->|yes| P5["-0.20"]

    START --> C6{"education end_year\n&lt; start_year?"}
    C6 -->|yes| P6["-0.20"]

    START --> C7{"single role tenure\n&gt; 360 months (30 yrs)?"}
    C7 -->|yes| P7["-0.25"]

    P1 & P2 & P3 & P4 & P5 & P6 & P7 --> FINAL["plausibility = max(0, 1.0 - total_deductions)\nhoneypots converge to ~0.0"]
```

---

## How Reasoning is Generated (No Hallucination)

```mermaid
flowchart LR
    subgraph SLOTS["Slot-based assembly — only real facts"]
        S1["Slot A\ncurrent_title at current_company\n(product co vs services firm)"]
        S2["Slot B\nbest evidence phrase\nfrom career_history description\n(regex match, real sentence)"]
        S3["Slot C\ntop 2 matched skills\nfrom candidates own skills array\nwith platform assessment score"]
        S4["Slot D\nhonest concern\nnotice period / low response rate\n/ inactive / location mismatch"]
    end

    subgraph TONE["Tone by rank tier"]
        T1["Rank 1-10\nconfident + specific\nall 4 slots"]
        T2["Rank 11-50\nconfident with caveat\nslots A + C + D"]
        T3["Rank 51-100\nexplicitly marginal\nadjacent fit language"]
    end

    subgraph GUARD["Hallucination guard"]
        V["validate_no_hallucination()\nasserts every skill name mentioned\nexists in candidates skills array\nevery employer mentioned\nexists in career_history"]
    end

    SLOTS --> TONE --> GUARD --> OUT["reasoning string\n1-2 sentences\nASCII clean, factual"]
```

---

## File Structure

```
.
├── config.py              # ALL weights, thresholds, regex patterns (single source of truth)
├── precompute.py          # Phase A: embed + BM25 + features -> artifacts/
├── rank.py                # Phase B: load artifacts -> score -> submission.csv
├── validate_submission.py # Format validator (7 checks)
├── src/
│   ├── parsing.py         # JSON array + JSONL streaming loader
│   ├── text.py            # Candidate + JD text builders
│   ├── features.py        # Structured feature extraction (regex, heuristics)
│   ├── retrieval.py       # Dense cosine + BM25 + RRF + optional cross-encoder
│   ├── signals.py         # Availability multiplier from 23 redrob signals
│   ├── scoring.py         # Composite score + --explain mode
│   ├── consistency.py     # Honeypot detection (7 plausibility checks)
│   └── reasoning.py       # Deterministic slot-based reasoning generator
├── data/                  # candidates.jsonl, sample_candidates.json, schema
├── artifacts/             # candidate_embeddings.npy, bm25_index/, features.parquet
└── .venv/                 # project-local virtualenv
```

---

## Performance

| Step | Time |
|------|------|
| Precompute 100k (Phase A) | ~3.5 min |
| Load artifacts | 1.2s |
| Dense + BM25 + RRF scoring | 1.7s |
| Composite scoring 100k | ~101s |
| Reasoning + CSV write | ~1s |
| **Total rank.py** | **~1 min 53 sec** |

Budget constraint: 5 min wall-clock. Actual: under 2 min.
