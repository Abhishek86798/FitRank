# Requirements — Explainable Candidate Ranking Engine (FitRank)

**Draft v1.0** — updated against real dataset, JD, submission spec, and validator. Companion to `PROJECT_CONTEXT.md`.

**Priority key:** `[M]` must-have for a valid submission · `[S]` should-have, strong differentiator · `[C]` could-have if time allows

---

## 1. Purpose

Define what the system must do and the qualities it must have, in traceable terms, so we can build to a clear target under a 40–50 hour budget and verify we've met it before submitting.

---

## 2. Functional requirements

### 2.1 Ingestion and representation

| ID | Priority | Requirement |
|----|----------|-------------|
| FR-1 | `[M]` | Stream `candidates.jsonl` record by record without loading the whole file into memory; handle the full ~475 MB pool. |
| FR-2 | `[M]` | Parse each record against `candidate_schema` into a normalized internal candidate card. |
| FR-3 | `[M]` | Run the full pipeline against `sample_candidates` first, end to end, before scaling to the full file. |
| FR-4 | `[S]` | Derive signals not present literally in the data: tenure stability, seniority level, career trajectory (is scope climbing?), recency of relevant experience, domain match, and evidence of ownership/impact. |
| FR-5 | `[S]` | Parse and incorporate the 23 platform/behavioral signals from `redrob_signals` (activity, responsiveness, assessment outcomes, etc.). Each signal must be addressable by the scorer. |
| FR-6 | `[M]` | De-identify candidate cards before any ranking or reasoning step: strip name, gender, ethnicity, photo, and other protected attributes (see NFR-7). |

### 2.2 Role understanding

| ID | Priority | Requirement |
|----|----------|-------------|
| FR-7 | `[M]` | Convert `job_description` into a structured role spec: hard requirements, soft/nice-to-haves, seniority, and domain. Committed to `role_model.yaml`. |
| FR-8 | `[S]` | Infer requirements the JD implies but does not state (e.g., "early-stage startup" → tolerance for ambiguity, breadth over depth). |
| FR-9 | `[S]` | Generate an "ideal candidate persona" used as the semantic match target, so matching is meaning-to-meaning rather than keyword-to-keyword. |

### 2.3 Retrieval — broad funnel

| ID | Priority | Requirement |
|----|----------|-------------|
| FR-10 | `[M]` | Build, once and cache to disk, a dense vector index over all candidate cards using approximate nearest neighbour (ANN) search. |
| FR-11 | `[S]` | Build a sparse/lexical index (e.g., BM25) over the same cards. |
| FR-12 | `[S]` | Combine dense and sparse retrieval via rank fusion (e.g., Reciprocal Rank Fusion) to catch both semantic matches and exact must-have terms. |
| FR-13 | `[M]` | Per role, retrieve a top-K candidate set (target ~50) efficiently from cached indexes, without re-embedding the pool. |

### 2.4 Reranking — medium funnel

| ID | Priority | Requirement |
|----|----------|-------------|
| FR-14 | `[M]` | Score the retrieved set with a transparent composite that combines semantic relevance with the derived and platform signals (FR-4, FR-5). All weights must be visible and tunable. |
| FR-15 | `[C]` | Optionally apply a cross-encoder reranker on the persona-vs-card pair for the retrieved set. |
| FR-16 | `[M]` | Narrow to a final candidate set for judgment (target ~15). |

### 2.5 Judgment and explainability — narrow funnel

| ID | Priority | Requirement |
|----|----------|-------------|
| FR-17 | `[M]` | For each final candidate, produce a per-dimension fit assessment against a fixed rubric (core skills, domain fit, seniority match, trajectory, risk). |
| FR-18 | `[M]` | Produce a short, human-readable rationale per candidate: why they fit, what's missing/risky, and a confidence level. |
| FR-19 | `[M]` | Ground every claim in a rationale to a specific span in the candidate's data; do not assert qualifications not supported by the record (anti-hallucination). |
| FR-20 | `[S]` | Reduce position/order bias by comparing candidates relative to one another (listwise/pairwise), not scoring each in isolation, and check stability across at least two runs. |
| FR-21 | `[M]` | Emit a final ranked ordering of candidates with scores. |

### 2.6 Output

| ID | Priority | Requirement |
|----|----------|-------------|
| FR-22 | `[M]` | Write the ranked result to the exact format defined by `submission_spec` and matching `sample_submission.csv` (column names, order, types, score range). Columns: `candidate_id,rank,score,reasoning`. |
| FR-23 | `[M]` | Populate `submission_metadata_template.yaml` with all required metadata. |
| FR-24 | `[M]` | The generated output must pass `validate_submission.py` with no errors before submission. This is a hard gate. |

### 2.7 Demo interface

| ID | Priority | Requirement |
|----|----------|-------------|
| FR-25 | `[S]` | Provide a simple UI (e.g., Streamlit) that shows the ranked shortlist for the role, lets the user expand any candidate to read the grounded reasoning, and surfaces the score breakdown. |
| FR-26 | `[C]` | Allow the user to nudge signal weights and re-rank live, demonstrating that the ranking responds to behavioral signals. |

---

## 3. Non-functional requirements

| ID | Priority | Requirement |
|----|----------|-------------|
| NFR-1 | `[M]` | **Scale.** Indexing must complete on the full ~475 MB pool within the hackathon's practical compute budget; per-role retrieval must return in seconds, not minutes. |
| NFR-2 | `[M]` | **Cost control.** The LLM judgment stage runs only on the final ~15 candidates, never the full pool. Expensive operations (embedding) run once and are cached. |
| NFR-3 | `[M]` | **Reproducibility.** One documented command takes raw data to a valid output file. Fixed random seeds where applicable. Pinned dependencies. |
| NFR-4 | `[S]` | **Explainability/trust.** Output is auditable: a reader can trace every ranking decision to inputs and weights. |
| NFR-5 | `[S]` | **Robustness.** Pipeline tolerates malformed/missing fields in records without crashing the run; skipped records are logged. |
| NFR-6 | `[S]` | **Configurability.** Rubric, signal weights, K values, and shortlist length live in config, not hard-coded. |
| NFR-7 | `[M]` | **Fairness.** Matching and judgment operate on de-identified cards (FR-6); protected attributes never enter scoring. Stated explicitly in the deck and README. |
| NFR-8 | `[C]` | **Cost/scale transparency.** The deck shows that per-role cost stays roughly flat as the pool grows. |

---

## 4. Data requirements

| ID | Priority | Requirement |
|----|----------|-------------|
| DR-1 | `[M]` | Treat `candidate_schema` as the source of truth for candidate fields; map each field to one of: embed-as-text, structured-signal, identifier, or drop. |
| DR-2 | `[M]` | Confirm whether platform/behavioral signals are inline in each record or joined from elsewhere (confirmed: inline in `redrob_signals` object per candidate). |
| DR-3 | `[M]` | Confirm the candidate id field used for output keying (confirmed: `candidate_id`, format `CAND_XXXXXXX`). |
| DR-4 | `[S]` | Keep raw data read-only; all derived artifacts (cards, embeddings, indexes) written to a separate `artifacts/` cache directory. |

---

## 5. Output and submission requirements

| ID | Priority | Requirement |
|----|----------|-------------|
| OR-1 | `[M]` | Output columns in order: `candidate_id,rank,score,reasoning`. UTF-8 encoding. Exactly 100 data rows + 1 header row. |
| OR-2 | `[M]` | Ranks 1–100 each appear exactly once. Score is non-increasing with rank. Ties broken by `candidate_id` ascending. |
| OR-3 | `[M]` | Required metadata YAML is complete and valid before submission. |
| OR-4 | `[M]` | `validate_submission.py` passes on the final file. This is a hard gate — no exceptions. |
| OR-5 | `[S]` | The `reasoning` field is populated with grounded, specific, JD-connected rationales for every ranked candidate. |

---

## 6. Evaluation requirements

| ID | Priority | Requirement |
|----|----------|-------------|
| ER-1 | `[S]` | Construct a golden set: hand rank-order ~30–50 candidates for the role to serve as ground truth. |
| ER-2 | `[S]` | Report at least one ranking-quality metric against the golden set (NDCG@10, NDCG@50, MAP, Kendall's tau). |
| ER-3 | `[S]` | Produce a side-by-side "missed candidate" example: a candidate ranked highly by the system that a naive keyword/BM25 filter buries, with the reason why they're actually a fit. |
| ER-4 | `[C]` | Run a small ablation showing the platform signals (FR-5) change the ranking, evidencing their value as a differentiator. |

---

## 7. Scoring model — feature groups

Each candidate receives a feature vector. The final score combines them via the LTR model, with a transparent weighted-sum as the explainable fallback.

| # | Feature group | Description |
|---|--------------|-------------|
| 1 | **Semantic fit** | `cosine(candidate_emb, jd_vector)` — one signal among many, deliberately not the decider. |
| 2 | **Domain / role gate** | Is this person actually an ML/IR engineer? Derived from current + past titles and career descriptions, not the skills list. Non-engineering titles with AI skills get strongly penalized. |
| 3 | **JD disqualifier rules** | From `role_model.yaml`: consulting-firm-only career, pure-research-only, CV/speech/robotics without NLP/IR, recent-LangChain-only without prior ML, "architect who stopped coding," title-chasing. Each applies a calibrated down-weight. |
| 4 | **Experience fit** | Total YOE vs the 5–9 band (soft, not a hard cut), and applied-ML years at product companies vs services. |
| 5 | **Production-ML evidence** | Career descriptions showing shipped retrieval/ranking/search/recommendation/evaluation systems at real scale. The single strongest positive signal. |
| 6 | **Consistency / honeypot penalty** | Internal contradictions: proficiency vs `duration_months`, YOE vs summed tenures, assessment scores vs claimed proficiency. Impossible profiles sink. |
| 7 | **Behavioral multiplier** | From `redrob_signals`: `recruiter_response_rate`, `last_active_date` recency, `open_to_work_flag`, `interview_completion_rate`, `saved_by_recruiters_30d`, `github_activity_score`. |
| 8 | **Location fit** | Noida/Pune/Hyderabad/Mumbai/Delhi NCR, or `willing_to_relocate`. |
| 9 | **Assessment validation** | `skill_assessment_scores` corroborate (or contradict) self-reported skills — a cheap lie-detector on the skills list. |

---

## 8. Out of scope

Production deployment, authentication, live ATS integration, model fine-tuning, multi-role batch tooling beyond the brief, and any UI work that doesn't serve the demo or the judged outcome.

---

## 9. Acceptance criteria

The submission is ready when **all** of the following hold:

1. Pipeline runs end to end on the full pool and writes the output file (FR-1, FR-13, FR-21, FR-22).
2. `validate_submission.py` passes with no issues (OR-4).
3. Every shortlisted candidate has a grounded rationale with fit reasons, gaps, and confidence (FR-18, FR-19).
4. Matching is attribute-blind (FR-6, NFR-7).
5. At least one quality metric is reported on the golden set (ER-2).
6. The deck covers approach, rationale, the missed-candidate demo, and the scale/cost story.
7. README enables a clean clone-and-run with one command (NFR-3).

---

## 10. Submission checklist

- [ ] CSV: exactly 100 data rows + header; columns `candidate_id,rank,score,reasoning` in order; UTF-8
- [ ] Ranks 1–100 each once; score non-increasing; ties broken by `candidate_id` ascending
- [ ] Every `candidate_id` exists in `candidates.jsonl`; format `CAND_XXXXXXX`
- [ ] `validate_submission.py` passes with no issues
- [ ] Reasoning: specific, JD-connected, honest about gaps, varied, rank-consistent, zero hallucination
- [ ] Honeypot rate in top 100 = 0 (target); no skill-stuffed non-engineers in top 10
- [ ] `rank.py` reproduces the CSV on CPU, offline, ≤5 min, ≤16 GB from a clean clone
- [ ] Repo: README with 1-command repro + headline metric; pinned deps; `submission_metadata.yaml`
- [ ] Working sandbox/demo link (HF Spaces / Streamlit Cloud)
- [ ] Portal metadata ready; AI tools declared honestly
- [ ] Real git history showing iteration

---

## 11. Open questions and dependencies

### Resolved
- Candidate id field: `candidate_id`, format `CAND_XXXXXXX`.
- Output columns: `candidate_id,rank,score,reasoning`.
- Platform/behavioral signals: inline in each candidate's `redrob_signals` object (23 signals confirmed).
- Submission is a single role ranked against the whole pool.
- Score must be non-increasing; ties broken by `candidate_id` ascending.

### Still open — decide during build
- Whether ground-truth labels exist or we self-label the golden set.
- Embedding model choice (local BGE/E5 vs API) given pool size and compute available.
- Whether to use a quantized local LLM for reasoning generation vs the templated composer.
