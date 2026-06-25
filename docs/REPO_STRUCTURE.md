# FitRank — Repository Structure & Navigation Guide

> Quick reference for all three team members. Every file listed, what it does, who owns it, and when it gets created.

---

## Full Tree

```
fitrank/
│
├── README.md                          ← Start here. Architecture + one-command reproduce.
├── requirements.txt                   ← All pinned dependencies. Install this first.
├── submission_metadata.yaml           ← Portal submission form (team info, GitHub URL, sandbox URL)
├── Dockerfile                         ← (optional) Clean-env reproduce for Stage 3
│
├── role_model.yaml                    ← The JD encoded as rules. Brain of the scorer.
│
├── data/
│   ├── candidates.jsonl               ← Full dataset (475 MB). Gitignored.
│   └── sample_candidates.json         ← 100-ish candidates for development. Use this daily.
│
├── artifacts/                         ← Precomputed outputs. Gitignored (too large).
│   ├── embeddings.npy                 ← All candidate embeddings. (n_candidates × 768, fp16)
│   ├── candidate_ids.npy              ← Aligned array of CAND_XXXXXXX IDs
│   ├── jd_vector.npy                  ← JD embedding vector (1 × 768, fp16)
│   └── ltr_model.txt                  ← Trained LightGBM LambdaMART model
│
├── src/
│   ├── precompute.py                  ← OFFLINE. Builds all artifacts. Run once.
│   ├── data_loader.py                 ← Streaming loader + candidate text builder
│   ├── retriever.py                   ← Cosine similarity retrieval + BM25 hybrid
│   ├── feature_builder.py             ← Turns candidate dict → feature vector dict
│   ├── scorer.py                      ← LTR scoring (LightGBM) + weighted-sum fallback
│   ├── reasoning.py                   ← Fact-grounded reasoning string composer
│   ├── train_ltr.py                   ← OFFLINE. Trains LightGBM on golden set labels.
│   └── rank.py                        ← ENTRYPOINT. Sandboxed ranking step. One command.
│
├── eval/
│   ├── golden_set.csv                 ← Hand-labeled candidates (30–50 rows). Ground truth.
│   ├── notes.txt                      ← B's raw hand-ranking notes. Seed for golden set.
│   ├── evaluate.py                    ← Computes NDCG@10/50, MAP, P@10 vs golden set
│   ├── honeypot_audit.py              ← Prints top-20 profiles for manual inspection
│   └── ablation_results.txt           ← Logged results of signal ablation experiments
│
├── app.py                             ← Streamlit demo. Hosted on HuggingFace Spaces.
└── validate_submission.py             ← Provided by organizers. Run before every submit.
```

---

## File-by-File Reference

### Root Files

---

#### `README.md`
**Owner:** A &nbsp;|&nbsp; **Created:** Day 6 (final), skeleton Day 1

The first thing judges read at Stage 3 and Stage 4. Must contain:
- Two-paragraph architecture overview
- The exact one-command reproduce sequence
- Headline NDCG@10 metric vs baseline
- Fairness statement (matching is attribute-blind)
- Dependency install instructions

```bash
# The reproduce sequence that must be in README
python src/precompute.py
python src/rank.py --candidates ./data/candidates.jsonl --out ./team_xxx.csv
```

---

#### `requirements.txt`
**Owner:** A &nbsp;|&nbsp; **Created:** Day 1

All Python dependencies with pinned versions. Judges reproduce your environment from this file. If a version isn't pinned, Stage 3 reproduction may break.

```
sentence-transformers==2.7.0
lightgbm==4.3.0
rank-bm25==0.2.2
polars==0.20.0
orjson==3.9.15
numpy==1.26.4
pyyaml==6.0.1
python-dateutil==2.9.0
scikit-learn==1.4.0
streamlit==1.33.0
```

---

#### `submission_metadata.yaml`
**Owner:** C &nbsp;|&nbsp; **Created:** Day 3

Required portal metadata. Mirrors what you fill in the submission form. Template provided by organizers. Fields include: team name, contact info, GitHub URL, sandbox/demo URL, AI tools declared, compute environment summary, methodology summary (≤200 words).

---

#### `Dockerfile`
**Owner:** A &nbsp;|&nbsp; **Created:** Day 5 (optional but recommended)

Self-contained build for Stage 3 code reproduction. If the judges can `docker pull` and `docker run` your image, Stage 3 becomes trivial. Without it, they clone your repo and hope your environment matches.

---

#### `role_model.yaml`
**Owner:** B &nbsp;|&nbsp; **Created:** Day 1, refined through Day 3

The single most important non-code file. Contains the JD encoded as structured rules. Every feature in `feature_builder.py` and every weight in `scorer.py` references values from here. When you change your understanding of the JD, you change this file — not the code.

```yaml
experience_band: [5, 9]
must_have_domains:
  - nlp
  - information retrieval
  - ranking
  - recommendation
  - search
  - embeddings
disqualifying_titles:
  - marketing manager
  - hr manager
  - content writer
  - graphic designer
  - business analyst
disqualifying_company_types:
  - tcs
  - infosys
  - wipro
  - accenture
  - cognizant
  - capgemini
ideal_signals:
  - production_deployment
  - shipped_retrieval_system
  - shipped_ranking_system
  - shipped_recommendation_system
location_preferences:
  - pune
  - noida
  - delhi
  - mumbai
  - hyderabad
notice_period_preference_days: 30
```

---

### `data/`

---

#### `data/candidates.jsonl`
**Owner:** Everyone reads, no one edits &nbsp;|&nbsp; **Gitignored**

The full 475 MB candidate pool. ~100K+ records, one JSON object per line. Structure defined by `candidate_schema`. Never load all of this into memory at once — always stream it with `data_loader.py`.

Fields per record: `candidate_id`, `profile`, `career_history`, `education`, `skills`, `certifications`, `languages`, `redrob_signals`

---

#### `data/sample_candidates.json`
**Owner:** Everyone reads, no one edits &nbsp;|&nbsp; **Committed to git**

~100 candidate records for daily development. Use this exclusively on Days 1–3. Never debug on `candidates.jsonl` — it's 475 MB and makes iteration slow.

---

### `artifacts/`

All files in this directory are generated by `precompute.py` and `train_ltr.py`. They are **gitignored** — too large for git. The `README.md` must document how to regenerate them.

---

#### `artifacts/embeddings.npy`
**Owner:** A &nbsp;|&nbsp; **Generated by:** `precompute.py`

Shape: `(n_candidates, 768)` — dtype: `float16`

One embedding vector per candidate, in the same order as `candidate_ids.npy`. Generated with BGE-base-en-v1.5, L2-normalized so dot product = cosine similarity.

Memory: ~153 MB for 100K candidates at fp16.

```python
# Load at ranking time
embeddings = np.load("artifacts/embeddings.npy").astype(np.float32)
```

---

#### `artifacts/candidate_ids.npy`
**Owner:** A &nbsp;|&nbsp; **Generated by:** `precompute.py`

Shape: `(n_candidates,)` — dtype: string array

The `CAND_XXXXXXX` IDs in the exact same order as `embeddings.npy`. Index `i` in this array corresponds to row `i` in `embeddings.npy`. Never let these get out of sync.

```python
candidate_ids = np.load("artifacts/candidate_ids.npy")
# candidate_ids[42] → "CAND_0042871"
# embeddings[42]    → embedding for CAND_0042871
```

---

#### `artifacts/jd_vector.npy`
**Owner:** A &nbsp;|&nbsp; **Generated by:** `precompute.py`

Shape: `(1, 768)` — dtype: `float16`

The job description encoded as a single embedding vector. Uses the BGE query prefix: `"Represent this sentence for searching relevant passages: "` + JD text. Generated once in precompute, loaded at ranking time.

---

#### `artifacts/ltr_model.txt`
**Owner:** B &nbsp;|&nbsp; **Generated by:** `train_ltr.py`

The trained LightGBM LambdaMART model in text format. Loaded by `scorer.py` at ranking time via `lgb.Booster(model_file=...)`. If this file doesn't exist, `scorer.py` falls back to the weighted-sum scorer automatically.

---

### `src/`

---

#### `src/precompute.py`
**Owner:** A &nbsp;|&nbsp; **Created:** Day 2 &nbsp;|&nbsp; **Phase:** Offline (no constraints)

Runs once, offline, before the competition's sandboxed ranking step. GPU and hosted LLMs allowed here. Produces all files in `artifacts/`. Can take hours — that's fine.

**What it does:**
1. Reads all candidates from `candidates.jsonl` in batches
2. Calls `build_candidate_text()` to build embeddable text per candidate
3. Encodes all texts with BGE-base-en-v1.5 (`normalize_embeddings=True`)
4. Saves `embeddings.npy` (fp16) and `candidate_ids.npy`
5. Encodes the JD text with query prefix → saves `jd_vector.npy`

```bash
python src/precompute.py
# Expected output: "Saved. Shape: (103842, 768), Size: 159.1 MB"
```

---

#### `src/data_loader.py`
**Owner:** A &nbsp;|&nbsp; **Created:** Day 1

Two core functions used by both `precompute.py` and `rank.py`.

**`stream_candidates(path, batch_size=1000)`**
Generator that reads `candidates.jsonl` line-by-line with `orjson`, yields lists of candidate dicts. Never loads the whole file. Safe to call on 475 MB.

**`build_candidate_text(candidate)`**
Takes a candidate dict, builds the text string we embed. Prioritises career history descriptions (hard to fake) over skills list (easy to fake). Skills are appended last, down-weighted by position. Caps at 2048 chars.

```python
from src.data_loader import stream_candidates, build_candidate_text

for batch in stream_candidates("data/candidates.jsonl", batch_size=500):
    for cand in batch:
        text = build_candidate_text(cand)
```

---

#### `src/retriever.py`
**Owner:** A &nbsp;|&nbsp; **Created:** Day 2

Retrieves the top-K most relevant candidates from precomputed artifacts. Two retrieval modes:

**Dense retrieval** (`retrieve_top_k`)
Matrix multiply: `embeddings @ jd_vector.T` → similarity scores for all 100K candidates in ~50ms. Returns top-K by cosine similarity.

**Hybrid retrieval** (`HybridRetriever`)
Combines dense + BM25 via Reciprocal Rank Fusion (RRF). Catches both semantic matches and exact keyword hits (specific tool names like "Qdrant", "LambdaMART").

```python
from src.retriever import retrieve_top_k

top_ids, scores = retrieve_top_k(embeddings, jd_vector, candidate_ids, k=50)
```

---

#### `src/feature_builder.py`
**Owner:** B &nbsp;|&nbsp; **Created:** Day 2 &nbsp;|&nbsp; **Most important file**

The heart of the ranking quality. Turns a raw candidate dict + role model into a flat dict of numerical features that the scorer can act on.

**Input:** `(candidate: dict, role_model: dict, cosine_sim: float)`
**Output:** `dict` with 12+ float features, all in range 0.0–1.0 (except penalties which are negative)

| Feature | What it measures |
|---------|-----------------|
| `cosine_similarity` | Semantic similarity to JD via embeddings |
| `experience_fit_score` | YOE vs the 5–9 year band in role_model |
| `is_ml_engineer` | Current/past title matches ML engineering titles |
| `title_disqualified` | Hard gate: -1.0 if non-engineer title detected |
| `production_ml_score` | Evidence of shipping real ML systems in career text |
| `domain_alignment` | NLP/IR/ranking keyword density in career descriptions |
| `consulting_penalty` | Fraction of career at consulting firms |
| `behavioral_multiplier` | Weighted: availability + response rate + open_to_work |
| `consistency_score` | Honeypot detector: checks profile internal consistency |
| `location_score` | City match + willing_to_relocate flag |
| `notice_penalty` | Penalises >30 day notice periods |
| `github_activity` | Normalised github_activity_score from redrob_signals |

```python
from src.feature_builder import build_feature_vector

feats = build_feature_vector(candidate, role_model, cosine_sim=0.82)
# → {"cosine_similarity": 0.82, "production_ml_score": 0.75, ...}
```

---

#### `src/scorer.py`
**Owner:** B &nbsp;|&nbsp; **Created:** Day 2 (weighted-sum), Day 4 (LTR)

Takes the feature vector from `feature_builder.py` and returns a single float score. Two modes:

**LTR mode (primary):** Loads `artifacts/ltr_model.txt`, calls `lgb.Booster.predict()`. Optimised directly for NDCG — the competition metric.

**Weighted-sum mode (fallback):** Used when no LTR model exists, or as an explainable baseline for Stage 5 interviews. Weights are visible and tunable in `role_model.yaml`.

```python
from src.scorer import LTRScorer, score_with_weighted_sum

scorer = LTRScorer("artifacts/ltr_model.txt")
score = scorer.score(feature_matrix)   # LTR

# or fallback:
score = score_with_weighted_sum(features)
```

---

#### `src/reasoning.py`
**Owner:** C &nbsp;|&nbsp; **Created:** Day 2

Generates the `reasoning` column for each ranked candidate. No LLM at ranking time — this is a fact-grounded template composer. Every claim it makes is pulled directly from the candidate record.

**Stage 4 checks it must pass:**
- Specific facts (title, YOE, company, named signals)
- JD connection (links facts to role requirements)
- Honest concerns (surfaces at least one gap per candidate)
- No hallucination (zero claims not in the profile)
- Variation (10 sampled rows are substantively different)
- Rank consistency (tone matches rank position)

```python
from src.reasoning import compose_reasoning

text = compose_reasoning(candidate, features, rank=3, role_model=role_model)
# → "Senior ML Engineer at Zomato (7 yrs) with evidence of production
#    retrieval system deployment and strong platform activity.
#    Notice period of 60 days may delay onboarding."
```

---

#### `src/train_ltr.py`
**Owner:** B &nbsp;|&nbsp; **Created:** Day 4 &nbsp;|&nbsp; **Phase:** Offline

Trains the LightGBM LambdaMART model on the hand-labeled golden set. Runs offline (not in the sandbox). Reads `eval/golden_set.csv`, builds feature matrix using `feature_builder.py`, trains, saves `artifacts/ltr_model.txt`.

```bash
python src/train_ltr.py
# Output: "Training LTR model on 50 labeled candidates..."
# Output: "Top features by gain: production_ml_score: 142.3, is_ml_engineer: 98.1..."
```

---

#### `src/rank.py`
**Owner:** A &nbsp;|&nbsp; **Created:** Day 3 &nbsp;|&nbsp; **The sandboxed entrypoint**

The single file that runs inside the competition sandbox. Must complete in under 5 minutes on CPU with no network access. This is what gets reproduced at Stage 3.

**What it does (in order):**
1. Load all artifacts from `artifacts/`
2. Run `retrieve_top_k()` → top-50 candidate IDs by cosine similarity
3. Stream `candidates.jsonl`, collect only the 50 full records (early-stop)
4. Call `build_feature_vector()` for each of the 50
5. Call `scorer.py` → final scores
6. Sort descending, enforce monotonic scores, break ties by `candidate_id`
7. Call `compose_reasoning()` for each of top-100
8. Write `submission.csv`

```bash
# The one command. Must work from a clean clone.
python src/rank.py --candidates ./data/candidates.jsonl --out ./team_xxx.csv
```

---

### `eval/`

---

#### `eval/golden_set.csv`
**Owner:** B creates, C maintains &nbsp;|&nbsp; **Created:** Day 1 (seed), Day 2–4 (expanded)

The ground truth for local evaluation. 30–50 candidate profiles hand-labeled against the JD. This is the only way to measure ranking quality without the hidden ground truth.

```csv
candidate_id,relevance_label,notes
CAND_0012345,3,"ML engineer at Zomato, shipped recsys, active, Pune"
CAND_0067890,0,"Marketing Manager with AI keywords — classic trap"
CAND_0099123,2,"Data scientist, some NLP, no production deployment"
CAND_0045678,1,"ML background but consulting-only career (TCS, Infosys)"
```

Labels: `0` = irrelevant · `1` = weak · `2` = good · `3` = excellent

---

#### `eval/notes.txt`
**Owner:** B &nbsp;|&nbsp; **Created:** Day 1

Raw hand-ranking notes from B's initial JD analysis. Not machine-readable — just a scratchpad. Becomes the seed for `golden_set.csv`. Example:

```
Candidate CAND_0012345 — rank 1
  Title: ML Engineer at Zomato
  Why: Built recommendation system (production), 6 years, Pune, open to work
  Concern: none

Candidate CAND_0067890 — rank 0 (trap)
  Title: Marketing Manager
  Why not: Wrong domain entirely, skills list stuffed with AI keywords
```

---

#### `eval/evaluate.py`
**Owner:** C &nbsp;|&nbsp; **Created:** Day 1

Takes a `submission.csv` and `golden_set.csv`, computes all four scoring metrics, prints results. Run this after every significant change to the scorer.

```bash
python eval/evaluate.py \
  --submission submission_sample.csv \
  --golden eval/golden_set.csv

# Output:
# NDCG@10 : 0.743  (target: beat 0.5 baseline)
# NDCG@50 : 0.681
# MAP     : 0.592
# P@10    : 0.800
```

---

#### `eval/honeypot_audit.py`
**Owner:** C &nbsp;|&nbsp; **Created:** Day 1

Takes a `submission.csv`, loads the top-20 ranked profiles from `candidates.jsonl`, prints key fields for manual inspection. Run this every time you change the scoring logic.

```bash
python eval/honeypot_audit.py \
  --submission team_xxx.csv \
  --candidates data/candidates.jsonl \
  --top-n 20

# Output: prints title, company, YOE, key skills, and any inconsistencies
# for each of the top-20 candidates. You read them. Are they ML engineers?
```

---

#### `eval/ablation_results.txt`
**Owner:** B &nbsp;|&nbsp; **Created:** Day 5

Logged results of two key ablation experiments. Used in the deck and Stage 5 defense.

```
ABLATION 1: behavioral signals OFF (behavioral_multiplier = 0)
  NDCG@10 with signals:    0.743
  NDCG@10 without signals: 0.621
  Delta: -0.122  → signals add real value

ABLATION 2: domain gate OFF (title_disqualified = 0)
  Top-10 check: Marketing Manager appears at rank 4
  → domain gate is critical for trap defense
```

---

### Root Files (Demo & Validation)

---

#### `app.py`
**Owner:** C &nbsp;|&nbsp; **Created:** Day 3 (skeleton), Day 4 (polish)

Streamlit demo deployed to HuggingFace Spaces or Streamlit Cloud. Mandatory for portal submission — the organizers use it to verify your code runs before Stage 3 reproduction.

**What it does:**
- Upload a small candidates JSON (≤100 candidates)
- Click "Run ranking"
- Shows ranked table: candidate ID, score, reasoning
- Expandable per-candidate: full feature score breakdown
- Runs the full pipeline end-to-end in the browser

```bash
# Run locally
streamlit run app.py
```

---

#### `validate_submission.py`
**Owner:** Provided by organizers — do not edit &nbsp;|&nbsp; **Run:** Before every submission

Checks your CSV against all format requirements. Must pass with zero errors before uploading to portal. Run it on every CSV you produce.

```bash
python validate_submission.py --file team_xxx.csv

# Pass: "All checks passed. Your submission is valid."
# Fail: "ERROR: Score not monotonically non-increasing at row 43"
```

---

## Ownership Summary

| File / Directory | Owner | Day Created |
|-----------------|-------|-------------|
| `README.md` | A | Day 6 |
| `requirements.txt` | A | Day 1 |
| `submission_metadata.yaml` | C | Day 3 |
| `Dockerfile` | A | Day 5 |
| `role_model.yaml` | B | Day 1 |
| `data/candidates.jsonl` | — (provided) | — |
| `data/sample_candidates.json` | — (provided) | — |
| `artifacts/embeddings.npy` | A | Day 2 (sample), Day 3 (full) |
| `artifacts/candidate_ids.npy` | A | Day 2 |
| `artifacts/jd_vector.npy` | A | Day 2 |
| `artifacts/ltr_model.txt` | B | Day 4 |
| `src/precompute.py` | A | Day 2 |
| `src/data_loader.py` | A | Day 1 |
| `src/retriever.py` | A | Day 2 |
| `src/feature_builder.py` | B | Day 2 |
| `src/scorer.py` | B | Day 2 |
| `src/reasoning.py` | C | Day 2 |
| `src/train_ltr.py` | B | Day 4 |
| `src/rank.py` | A | Day 3 |
| `eval/golden_set.csv` | B → C | Day 1 |
| `eval/notes.txt` | B | Day 1 |
| `eval/evaluate.py` | C | Day 1 |
| `eval/honeypot_audit.py` | C | Day 1 |
| `eval/ablation_results.txt` | B | Day 5 |
| `app.py` | C | Day 3 |
| `validate_submission.py` | — (provided) | — |

---

## Data Flow Between Files

```
candidates.jsonl
      │
      ▼
data_loader.py ──────────────────────► precompute.py
      │                                      │
      │                           ┌──────────┴──────────┐
      │                           ▼                     ▼
      │                   embeddings.npy          jd_vector.npy
      │                   candidate_ids.npy
      │                           │
      │                           ▼
      │                      retriever.py  ◄──── jd_vector.npy
      │                           │
      │                    top-50 IDs
      │                           │
      ▼                           ▼
  (full records)         feature_builder.py ◄──── role_model.yaml
      │                           │
      │                    feature vectors
      │                           │
      │                           ▼
      │                       scorer.py ◄───────── ltr_model.txt
      │                           │
      │                     final scores
      │                           │
      ▼                           ▼
  reasoning.py ◄──── (candidate + features + rank)
      │
  reasoning strings
      │
      ▼
   rank.py  ──────────────► team_xxx.csv
                                  │
                                  ▼
                       validate_submission.py
                                  │
                                  ▼
                            evaluate.py ◄─── golden_set.csv
```

---

## Interface Contracts

These are the exact function signatures A, B, and C must agree on before Day 3 merge.
If any of these change, all three people need to know immediately.

```python
# data_loader.py  (A provides)
def stream_candidates(path: str, batch_size: int = 1000) -> Generator[list[dict], None, None]
def build_candidate_text(candidate: dict) -> str

# retriever.py  (A provides)
def retrieve_top_k(
    embeddings: np.ndarray,       # (n, 768) float32
    jd_vector: np.ndarray,        # (1, 768) float32
    candidate_ids: np.ndarray,    # (n,) str
    k: int = 50
) -> tuple[list[str], np.ndarray]  # (ids, scores)

# feature_builder.py  (B provides)
def build_feature_vector(
    candidate: dict,
    role_model: dict,
    cosine_sim: float
) -> dict[str, float]              # flat dict, all values float

# scorer.py  (B provides)
def score_with_weighted_sum(features: dict[str, float]) -> float  # 0.0–1.0

# reasoning.py  (C provides)
def compose_reasoning(
    candidate: dict,
    features: dict[str, float],
    rank: int,
    role_model: dict
) -> str                           # 1–2 sentence string, grounded in facts
```

---

## Gitignore Reference

```gitignore
# Large data files
data/candidates.jsonl
data/*.jsonl

# Generated artifacts (rebuild with precompute.py)
artifacts/

# Python
__pycache__/
*.pyc
.venv/
*.egg-info/

# Notebooks (use scratch only, not production)
*.ipynb
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Secrets
.env
*.key
```

---

## Quick Navigation by Task

| I want to... | Go to |
|-------------|-------|
| Change how candidate text is built for embedding | `src/data_loader.py` → `build_candidate_text()` |
| Add or modify a ranking feature | `src/feature_builder.py` → add to `build_feature_vector()` |
| Change feature weights | `src/scorer.py` → `WEIGHTS` dict |
| Change what the JD requires | `role_model.yaml` |
| Fix a hallucination in reasoning | `src/reasoning.py` → `compose_reasoning()` |
| Regenerate all embeddings | `python src/precompute.py` |
| Retrain the LTR model | `python src/train_ltr.py` |
| Run the full pipeline | `python src/rank.py --candidates ./data/candidates.jsonl --out ./team_xxx.csv` |
| Check submission format | `python validate_submission.py --file team_xxx.csv` |
| Measure ranking quality | `python eval/evaluate.py --submission team_xxx.csv --golden eval/golden_set.csv` |
| Check for honeypots in top-20 | `python eval/honeypot_audit.py --submission team_xxx.csv --candidates data/candidates.jsonl` |
| Add a label to the golden set | `eval/golden_set.csv` → add a row |
| Run the demo locally | `streamlit run app.py` |
