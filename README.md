# FitRank

Candidate ranking engine for Redrob AI's Senior AI Engineer (Founding Team) role.
Two-stage retrieval → multi-signal scoring pipeline that separates genuine ML engineers
from keyword-stuffed profiles.

**NDCG@10: 0.7929 · MAP: 0.5069 · `team_xxx.csv` validates with 0 errors · LambdaMART scorer**

---

## Two-track architecture

This repo intentionally separates the competition submission from the explainability
tooling built on top of it. They run different code paths on purpose — not an oversight.

**Track 1 — Competition submission (`src/rank.py`)**
Produces `team_xxx.csv`. Pure CPU, no network calls, runs in under 5 minutes inside the
sandbox. This is the only track that's actually scored.

```bash
python -m src.rank --candidates data/candidates.jsonl --output team_xxx.csv
```

**Track 2 — Explainability layer (`eval/generate_audit.py`, `app.py`)**
Produces `eval/decision_audit.json` — hiring-tier labels, counterfactual feature-ablation
analysis, and role-fit summaries for the candidates in `team_xxx.csv`. This is what powers
the Streamlit dashboard. It is **not** part of the competition-time ranking pipeline and
does not influence `team_xxx.csv` in any way.

```bash
python eval/generate_audit.py --submission team_xxx.csv --top-n 100
```

`--top-n` defaults to 100 (the full submission) rather than a smaller sample, since
reviewers may sample any row from the submission, not just the top of it.

This is why `src/rank.py` never imports `src/hiring_recommendation.py`,
`src/counterfactual.py`, or `src/role_analyzer.py` — those modules belong entirely to
Track 2 and would add risk (LLM calls, extra dependencies, runtime) to the sandboxed
submission path for no scoring benefit.

---

## How it works

FitRank is a two-stage pipeline that separates genuine ML engineers from keyword-stuffed profiles at 100k-candidate scale. **Stage 1 (Hybrid Retrieval)** encodes every candidate and the job description with BAAI/bge-base-en-v1.5 (768-dim, L2-normalised) and retrieves the top-K by cosine similarity. A BM25Okapi index runs in parallel on the same corpus; both ranked lists are merged via Reciprocal Rank Fusion (k=60) so that candidates strong on either semantic meaning or exact keyword match survive into the scoring stage.

**Stage 2 (LambdaMART Scoring)** applies a 15-feature signal vector to every retrieved candidate and scores it with a trained LightGBM LambdaMART model (`artifacts/ltr_model.txt`) — this is the scorer that actually produces `team_xxx.csv`. LambdaMART learns its own feature weights from the golden set during training; there are no static per-feature weights at inference time. Top signals include `cosine_similarity` (semantic fit), `domain_alignment` (NLP/IR/ranking keyword density corroborated against career descriptions, not just skill lists), and `production_ml_score` (evidence of shipped retrieval/ranking systems). `behavioral_multiplier` (open-to-work, recruiter response rate, recency, interview completion) proved critical in ablation — removing it drops NDCG@10 by 57.6%. Hard gates (`title_disqualified`, `impossibility_flag`, domain cap) ensure no non-engineer or fabricated profile can outrank genuine ML candidates regardless of other scores.

If `artifacts/ltr_model.txt` is missing, `LTRScorer` falls back to `score_with_weighted_sum()` — a simpler, hand-tuned static-weight scorer (the weight table below) used only when the trained model is unavailable.

```
candidates.jsonl
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1 — Hybrid Retrieval                         │
│  Dense (BGE cosine sim) + BM25 → RRF fusion         │
│  Top-100 candidates advance to scoring              │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2 — LambdaMART Scoring (15 features)         │
│  + domain cap (hard gate for non-ML profiles)       │
│  + corroborated signals (career desc vs skill only) │
└─────────────────────────────────────────────────────┘
      │
      ▼
team_xxx.csv  (candidate_id, rank, score, reasoning)
```

### Feature vector — 15 features (LambdaMART, the scorer that produces `team_xxx.csv`)

`LTRScorer.FEATURE_ORDER` in `src/scorer.py` and `src/train_ltr.py` (kept in sync —
must match exactly). LambdaMART learns weights for these during training; there is
no static per-feature weight at inference time.

| Feature | What it measures |
|---|---|
| `cosine_similarity` | Semantic match of candidate text to JD embedding (BGE) |
| `experience_fit_score` | YoE vs 5–9 year band, soft taper outside |
| `is_ml_engineer` | Current or past ML engineering title match |
| `production_ml_score` | Evidence of shipping real systems (ranking, retrieval, eval infra, vector DBs) |
| `domain_alignment` | NLP/IR/ranking keyword density — **career description corroborated only** |
| `consulting_penalty` | Fraction of career at consulting firms (soft, career-wide only) |
| `behavioral_multiplier` | Composite of open-to-work, recruiter response rate, recency, interview completion |
| `location_score` | Preferred Indian city / willing to relocate |
| `notice_penalty` | Stepped penalty: >30d / >60d / >90d / >120d |
| `github_activity` | Normalised `github_activity_score` from redrob signals |
| `ce_score` | Cross-encoder relevance score (offline, GPU-precomputed; see step 2b) |
| `response_rate_score` | Recruiter response rate, as-is from redrob signals |
| `active_job_seeking` | Applications submitted in the last 30 days |
| `skill_depth_score` | Duration + proficiency depth of ranking/retrieval/ML-relevant skills |
| `profile_completeness` | Normalised `profile_completeness_score` from redrob signals |

**Hard gates** (not in `FEATURE_ORDER` — applied before scoring, both modes):

| Feature | Effect |
|---|---|
| `title_disqualified` | Non-engineer title + no ML career history → score forced to 0.01 |
| `impossibility_flag` | Fabricated-credential signal (e.g. expert skill with 0 months duration) → score forced to 0.01 |

**Domain cap:** candidates with `is_ml_engineer=0` AND `domain_alignment=0` are
hard-capped at 0.25, preventing high behavioral scores from lifting non-engineers.

**Computed but not in `FEATURE_ORDER`:** `build_feature_vector()` also computes
`consistency_score` (honeypot detector — expert skills with zero duration, YoE vs
career-month gap). It's excluded from the LambdaMART feature set but is still read
directly by `score_with_weighted_sum()` (the fallback scorer below), by
`fast_filter.py`'s Stage-2 honeypot gate, and by `counterfactual.py`'s
explainability tooling.

### Weighted-sum fallback scorer (used only if `artifacts/ltr_model.txt` is absent)

| Feature | Weight | What it measures |
|---|---|---|
| `cosine_similarity` | 0.20 | Semantic match of candidate text to JD embedding (BGE) |
| `domain_alignment` | 0.20 | NLP/IR/ranking keyword density — **career description corroborated only** |
| `production_ml_score` | 0.20 | Evidence of shipping real systems (ranking, retrieval, eval infra, vector DBs) |
| `experience_fit_score` | 0.10 | YoE vs 5–9 year band, soft taper outside |
| `is_ml_engineer` | 0.10 | Current or past ML engineering title match |
| `behavioral_multiplier` | 0.10 | Composite of open-to-work, recruiter response rate, recency, interview completion |
| `consistency_score` | 0.05 | Honeypot detector — expert skills with zero duration, YoE vs career-month gap |
| `location_score` | 0.03 | Preferred Indian city / willing to relocate |
| `github_activity` | 0.02 | Normalised `github_activity_score` from redrob signals |
| `title_disqualified` | –1.0 | **Hard gate** — non-engineer title + no ML career history → score 0.01 |
| `consulting_penalty` | –0.08 | Fraction of career at consulting firms (soft, career-wide only) |
| `notice_penalty` | –0.02 | Stepped penalty: >30d / >60d / >90d / >120d |

**Corroboration rule:** skill-name-only keyword hits count at 0.3× (domain) and 0.4×
(production ML) vs career-description hits. A frontend engineer claiming FAISS expertise
with no retrieval work in their history scores near zero on both signals.

---

## Reproduce in one shot

```bash
pip install -r requirements.txt
python src/precompute.py
python src/rank.py --candidates ./data/candidates.jsonl --out ./team_xxx.csv
```

`precompute.py` defaults to `data/sample_candidates.json`; for the full 100k corpus pass
`--candidates data/candidates.jsonl`. `rank.py` auto-detects available artifacts (falls back
to `sample_` prefix if full embeddings are absent). No network calls occur during `rank.py`.

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 2. Precompute embeddings

```bash
# Sample (50 candidates, fast — use for development)
python -m src.precompute --candidates data/sample_candidates.json \
    --artifacts-dir artifacts --prefix sample_

# Full corpus (100k candidates, ~10 min CPU)
python -m src.precompute --candidates data/candidates.jsonl \
    --artifacts-dir artifacts
```

### 2b. Precompute cross-encoder scores (offline, Colab GPU only)

`ce_score` (one of the LTR features) comes from `artifacts/ce_scores.json`, a
`{candidate_id: raw_logit}` map produced by a sentence-transformers
`CrossEncoder`. Cross-encoders score every (JD, candidate) pair through a full
transformer forward pass with no caching, which is too slow for 100k
candidates on CPU — this step **must run on a GPU runtime (e.g. Colab)**,
never in the CPU-only competition sandbox. `rank.py` only consumes the
resulting JSON; it does not generate it.

```bash
# Run on Colab (or any CUDA machine), then download ce_scores.json into artifacts/
python scripts/precompute_cross_encoder.py \
    --candidates data/candidates.jsonl \
    --artifacts-dir artifacts
```

### 3. Run the ranking pipeline

```bash
# On sample (50 candidates → 50 rows)
python -m src.rank --candidates data/sample_candidates.json \
    --output submission_sample.csv

# On full corpus (produces 100-row submission.csv for validator)
python -m src.rank --candidates data/candidates.jsonl \
    --output submission.csv --top-k 150 --submission-size 100
```

### 4. Evaluate

```bash
python eval/evaluate.py submission_sample.csv --golden eval/golden_set.csv
```

```
Evaluation results (full 100k corpus, team_xxx.csv)
----------------------------------------------------
  NDCG@10  0.7929
  NDCG@50  0.6871
  MAP      0.5069
  P@10     0.4000
```

### 5. Validate submission

```bash
python validate_submission.py submission.csv
# → Submission is valid.
```

### 6. Honeypot audit

```bash
python eval/honeypot_audit.py submission.csv \
    --candidates data/candidates.jsonl --top 10
```

Prints full profiles for the top-N ranked candidates for manual inspection.

### 7. Streamlit demo

```bash
streamlit run app.py
```

Upload a candidates JSON file (same format as `data/sample_candidates.json`),
click **Run ranking**, and inspect the ranked table with per-candidate feature
breakdowns and reasoning strings.

---

## Project structure

```
FitRank/
├── src/
│   ├── precompute.py      # Offline: embed candidates + JD with BGE-base-en-v1.5
│   ├── rank.py            # Entrypoint: retrieve → feature → score → reason → CSV
│   ├── retriever.py       # Dense cosine + BM25 + RRF fusion
│   ├── feature_builder.py # 15-feature vector engineering (all signal logic here)
│   ├── scorer.py          # Weighted-sum + LambdaMART LTR scorer (auto-fallback)
│   ├── reasoning.py       # Natural language reasoning string per candidate
│   ├── data_loader.py     # Streaming JSONL reader, candidate text builder
│   └── train_ltr.py       # LightGBM LambdaMART training script
├── scripts/
│   └── precompute_cross_encoder.py  # Offline (Colab GPU): generates artifacts/ce_scores.json
├── eval/
│   ├── evaluate.py        # NDCG@10, NDCG@50, MAP, P@10 against golden set
│   ├── honeypot_audit.py  # Print top-N full profiles for manual inspection
│   └── golden_set.csv     # Hand-labeled relevance scores (0–3) for sample
├── artifacts/             # Precomputed .npy files (embeddings, ids, jd_vector)
├── data/                  # candidates.jsonl (100k) + sample_candidates.json (50)
├── app.py                 # Streamlit demo
├── role_model.yaml        # All weights, thresholds, disqualifiers — edit here
├── validate_submission.py # Competition validator (100 rows, ranks 1–100, UTF-8)
├── submission.csv         # Full-run output
└── submission_metadata.yaml
```

---

## Tuning weights

All scoring parameters live in [`role_model.yaml`](role_model.yaml). Change a weight,
re-run the pipeline on the sample, and check NDCG@10:

```bash
# Edit role_model.yaml, then:
python -m src.rank --candidates data/sample_candidates.json --output submission_sample.csv
python eval/evaluate.py submission_sample.csv --golden eval/golden_set.csv
```

Key levers:
- `feature_weights.domain_alignment` — controls how hard the NLP/IR keyword signal gates
- `feature_weights.behavioral_multiplier` — how much reachability matters vs domain fit
- `behavioral_weights.*` — sub-weights inside the behavioral composite
- `location_preferences.preferred_cities` — cities that score 1.0 vs 0.5
- `notice_period_preference_days` — threshold for zero penalty

---

## Honeypot defence

The dataset contains ~80 honeypot-style profiles. Three defences:

1. **`consistency_score`** — catches expert skills with `duration_months=0`, YoE vs
   career-months gap >3 years, and non-ML titles with ML-heavy descriptions.
2. **Corroboration rule** — `domain_alignment` and `production_ml_score` heavily
   discount keywords that appear only in skill lists, not in career descriptions.
3. **Domain cap** — any candidate with no ML title and no domain signal is capped at
   0.25 regardless of behavioral or cosine scores.

---

## Tech stack

| Component | Library |
|---|---|
| Candidate + JD embedding | `sentence-transformers` · BAAI/bge-base-en-v1.5 (768-dim) |
| Sparse retrieval | `rank-bm25` · BM25Okapi |
| Fusion | Reciprocal Rank Fusion (k=60) |
| LTR scorer (optional) | `lightgbm` · LambdaMART (falls back to weighted-sum) |
| Eval metrics | `scikit-learn` · `ndcg_score` |
| Demo | `streamlit` · `pandas` |

All inference runs **CPU-only**. Peak RAM ~400 MB for the sample run; ~2 GB for the
full 100k corpus (BM25 index dominates).

---

## Compute

| Step | Time (CPU) | Output |
|---|---|---|
| Precompute sample (50 candidates) | ~20s | `artifacts/sample_*.npy` |
| Precompute full (100k candidates) | ~10 min | `artifacts/embeddings.npy` (146 MB fp16) |
| Rank sample | ~2s | `submission_sample.csv` |
| Rank full | ~5 min | `team_xxx.csv` |

---

## Fairness statement

All scoring signals are **attribute-blind**: no feature uses or proxies for candidate gender,
age, religion, caste, nationality, or any other protected characteristic. The pipeline ranks
solely on professional evidence — career descriptions, verified skill signals, production
deployment history, and job-match proximity. Location and notice-period signals reflect
operational constraints stated in the job description (Pune/Noida/Delhi preferred; sub-30-day
notice), not demographic inferences. Honeypot defences target fabricated credentials, not
any candidate demographic. The corroboration rule penalises keyword stuffing equally regardless
of candidate background.
