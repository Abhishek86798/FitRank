# FitRank

Candidate ranking engine for Redrob AI's Senior AI Engineer (Founding Team) role.
Two-stage retrieval → multi-signal scoring pipeline that separates genuine ML engineers
from keyword-stuffed profiles.

**NDCG@10: 0.6586 · MAP: 0.6759 · `submission.csv` validates with 0 errors**

---

## How it works

```
candidates.jsonl
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1 — Hybrid Retrieval                         │
│  Dense (BGE cosine sim) + BM25 → RRF fusion         │
│  Top-50 candidates advance to scoring               │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2 — 12-Feature Weighted Scoring              │
│  + domain cap (hard gate for non-ML profiles)       │
│  + corroborated signals (career desc vs skill only) │
└─────────────────────────────────────────────────────┘
      │
      ▼
submission.csv  (candidate_id, rank, score, reasoning)
```

### Feature vector (12 features)

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

**Domain cap:** candidates with `is_ml_engineer=0` AND `domain_alignment=0` are
hard-capped at 0.25, preventing high behavioral scores from lifting non-engineers.

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
Evaluation results
-------------------
  NDCG@10  0.6586
  NDCG@50  0.8065
  MAP      0.6759
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
│   ├── feature_builder.py # 12-feature vector engineering (all signal logic here)
│   ├── scorer.py          # Weighted-sum + LambdaMART LTR scorer (auto-fallback)
│   ├── reasoning.py       # Natural language reasoning string per candidate
│   ├── data_loader.py     # Streaming JSONL reader, candidate text builder
│   └── train_ltr.py       # LightGBM LambdaMART training script
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
| Precompute sample (50 candidates) | ~5s | `artifacts/sample_*.npy` |
| Precompute full (100k candidates) | ~10 min | `artifacts/embeddings.npy` (~380 MB) |
| Rank sample | ~2s | `submission_sample.csv` |
| Rank full | ~5 min | `submission.csv` |
