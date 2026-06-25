# FitRank — 6-Day Team Roadmap
**Redrob Intelligent Candidate Ranking Hackathon**

---

## Roles

| Role | Name | Owns |
|------|------|------|
| **A** | Pipeline & Infrastructure | `precompute.py`, `rank.py`, data streaming, embeddings, `.npy` artifacts, constraint testing |
| **B** | Intelligence & Scoring | `role_model.yaml`, `feature_builder.py`, `scorer.py`, LightGBM LTR, golden set labeling |
| **C** | Evaluation & Submission | `evaluate.py`, `reasoning.py`, Streamlit demo, repo, `submission_metadata.yaml`, final CSV |

> **Parallel rule:** A works on sample data → full data pipeline. B works on features independently using sample JSON. C builds eval framework against mock scores. Everyone merges at Day 3 EOD.

---

## Day 1 — Setup & Parallel Foundations

### Goal
All three have working code running on `sample_candidates.json` by EOD. No one is blocked on anyone else.

---

### A — Environment + Data Loader

- [ ] Create repo, set up `requirements.txt`, folder structure (`src/`, `artifacts/`, `data/`, `eval/`)
- [ ] Write `stream_candidates()` generator — reads `candidates.jsonl` line by line using `orjson`, yields batches of 100
- [ ] Write `build_candidate_text()` — combines headline + summary + career history descriptions (skills last, down-weighted)
- [ ] Test on `sample_candidates.json`: load all records, print count, print one built text
- [ ] Install + verify: `sentence-transformers`, `numpy`, `orjson`, `polars`, `lightgbm`, `rank-bm25`, `pyyaml`, `python-dateutil`
- [ ] Pin all versions in `requirements.txt`

**Deliverable:** `src/data_loader.py` with `stream_candidates()` and `build_candidate_text()` working on sample data.

---

### B — JD Analysis + role_model.yaml

- [ ] Read the full JD carefully — at least twice. Highlight must-haves, disqualifiers, nice-to-haves
- [ ] Use Claude/GPT (offline, this is precompute) to draft `role_model.yaml` — then manually edit and own every line
- [ ] Define in the YAML: `experience_band`, `must_have_skills`, `disqualifying_titles`, `disqualifying_company_types`, `ideal_profile_signals`, `location_preferences`, `notice_period_preference`
- [ ] Read `sample_candidates.json` manually — open 10 profiles, rank them in your head 1–10 against the JD
- [ ] Write down your reasoning for each ranking in a `notes.txt` — this becomes your golden set seed

**Deliverable:** `role_model.yaml` committed and human-reviewed. Hand rankings of 10 sample candidates in `eval/notes.txt`.

---

### C — Eval Framework + Validator

- [ ] Run `validate_submission.py` on `sample_submission.csv` — read every check it performs, understand the exact format required
- [ ] Write `eval/evaluate.py` — takes a CSV + a golden set CSV, computes NDCG@10, NDCG@50, MAP, P@10 using `sklearn.metrics.ndcg_score`
- [ ] Generate a mock `submission.csv` with random scores for 100 fake `CAND_XXXXXXX` IDs — run validator on it, fix until it passes
- [ ] Write `eval/golden_set.csv` skeleton: `candidate_id, relevance_label, notes` — fill with B's 10 hand rankings
- [ ] Write `eval/honeypot_audit.py` — takes top-100 CSV, prints profiles of top-10 ranked candidates for manual review

**Deliverable:** `eval/evaluate.py` working on mock data. Validator passing on a dummy CSV. `eval/golden_set.csv` with 10 entries from B's notes.

---

### Tips for Day 1
- A: Don't touch `candidates.jsonl` yet — work only on sample. The full file is 475 MB and will slow you down.
- B: Don't second-guess `role_model.yaml` — write it fast, you'll refine it. The key question is "what gets someone immediately rejected?"
- C: Read the validator source code fully. Every check in there is a potential submission failure.

### EOD Checkpoint — Day 1
- [ ] Repo exists with folder structure, everyone has push access
- [ ] `sample_candidates.json` loads without errors (A)
- [ ] `role_model.yaml` has at least 6 fields filled (B)
- [ ] `validate_submission.py` passes on a dummy 100-row CSV (C)
- [ ] All dependencies install cleanly from `requirements.txt`

---

## Day 2 — Core Building Blocks

### Goal
A produces embeddings on sample data. B builds 8+ features. C expands golden set to 30 candidates. Everyone has working code — rough but functional.

---

### A — Embeddings Pipeline

- [ ] Write `src/precompute.py` — embeds all candidates from `sample_candidates.json` using BGE-base-en-v1.5
  - `normalize_embeddings=True` (mandatory)
  - Save as `fp16` → `artifacts/sample_embeddings.npy` + `artifacts/sample_candidate_ids.npy`
- [ ] Embed the JD text with query prefix → `artifacts/jd_vector.npy`
- [ ] Write `src/retriever.py` — `retrieve_top_k(embeddings, jd_vector, candidate_ids, k=50)` using matrix multiply
- [ ] Test: retrieve top 10 from sample, print their headlines — do they look like ML engineers?
- [ ] Add basic BM25 index on sample data (`rank_bm25`) — retrieves top-10 by keyword

**Deliverable:** `artifacts/sample_embeddings.npy` exists. `retrieve_top_k()` returns sensible results on sample data.

---

### B — Feature Builder (Core Features)

- [ ] Write `src/feature_builder.py` — `build_feature_vector(candidate, role_model, cosine_sim)` → dict
- [ ] Implement these 8 features first (the ones that matter most):
  1. `cosine_similarity` — pass-through from A's retriever
  2. `experience_fit_score` — YOE vs band from `role_model.yaml`
  3. `is_ml_engineer` — title check against ML title list
  4. `title_disqualified` — hard gate, returns -1.0 penalty if non-engineer title
  5. `production_ml_score` — scan career descriptions for deployment + system keywords
  6. `consulting_penalty` — check company names against consulting firm list
  7. `behavioral_multiplier` — `availability`, `response_rate`, `open_to_work` from `redrob_signals`
  8. `consistency_score` — proficiency vs duration_months, YOE vs career sum
- [ ] Test on 5 sample candidates: print full feature vector for each. Sanity check values.
- [ ] Write `src/scorer.py` — weighted sum baseline, returns float 0.0–1.0

**Deliverable:** `feature_builder.py` returning 8+ features per candidate. `scorer.py` returning a score. Both tested on sample data.

---

### C — Golden Set Expansion + Reasoning Skeleton

- [ ] Read 20 more candidate profiles from `sample_candidates.json` manually
- [ ] Label each 0/1/2/3 against the JD — add to `eval/golden_set.csv` (now 30 total)
  - 0 = irrelevant (wrong domain, honeypot, non-engineer)
  - 1 = weak match
  - 2 = good match
  - 3 = excellent match (ML engineer, production systems, active, nearby)
- [ ] Write `src/reasoning.py` — `compose_reasoning(candidate, features, rank, role_model)` → string
  - Must pull only facts from the actual candidate record (no hallucination)
  - Must vary phrasing between candidates
  - Must acknowledge at least one concern per candidate
- [ ] Test reasoning on 5 candidates — check: is every claim traceable to the profile?

**Deliverable:** `eval/golden_set.csv` with 30 labeled candidates. `reasoning.py` producing grounded, non-identical reasoning strings.

---

### Tips for Day 2
- A: Test `normalize_embeddings=True` is working — print `np.linalg.norm(embeddings[0])`, it should be ~1.0.
- B: Use `.get()` with defaults everywhere in `feature_builder.py`. The dataset has missing fields.
- C: When labeling the golden set, don't look at skills list first — read career history first. That's the ground truth.

### EOD Checkpoint — Day 2
- [ ] `sample_embeddings.npy` saved, shape `(n, 768)`, dtype `float16` (A)
- [ ] `retrieve_top_k()` returns 10 results that look like ML-adjacent profiles (A)
- [ ] `build_feature_vector()` returns dict with 8+ keys, all floats (B)
- [ ] `title_disqualified` correctly flags a Marketing Manager as -1.0 (B)
- [ ] Golden set has 30 labeled candidates (C)
- [ ] `reasoning.py` produces non-identical strings for 5 different candidates (C)

---

## Day 3 — Integration + First End-to-End Run

### Goal
**The big merge day.** Wire A + B + C together. Run the full pipeline on sample data and produce a valid `submission.csv` that passes the validator. Quality doesn't need to be great — just correct format and no crashes.

---

### A — Full Pipeline Skeleton (`rank.py`) + Full Data Embeddings

- [ ] Write `src/rank.py` — the sandboxed entrypoint:
  - Load artifacts from `artifacts/`
  - Call `retrieve_top_k()` → top 50 IDs
  - Stream `candidates.jsonl`, collect only the top-50 full records
  - Call B's `build_feature_vector()` for each
  - Call `scorer.py` for final scores
  - Call C's `reasoning.py` for reasoning strings
  - Write `submission.csv` in correct column order
- [ ] Run on sample data end-to-end → produce `submission_sample.csv`
- [ ] Run `validate_submission.py` on it — fix all errors
- [ ] Start embedding full `candidates.jsonl` (this takes time — kick it off, monitor memory)

**Deliverable:** `rank.py` runs end-to-end on sample. `submission_sample.csv` passes validator.

---

### B — Add Remaining Features + Tune Weights

- [ ] Add 4 more features to `feature_builder.py`:
  - `domain_alignment` — keyword density in career text (NLP/IR/ranking/retrieval/search)
  - `location_score` — city match + willing_to_relocate
  - `notice_penalty` — penalise >30 days notice
  - `github_activity` — normalise `github_activity_score` from redrob signals
- [ ] Hand-run the weighted-sum scorer on your 30 golden set candidates — compare ranking to your hand labels
- [ ] Tune weights in `scorer.py` so top-5 scored candidates match your top-5 hand-labeled candidates roughly
- [ ] Hand-check: does a honeypot candidate (impossible profile) get a low `consistency_score`? Verify with 2 examples.

**Deliverable:** 12+ features in feature vector. Weighted-sum scorer tuned against golden set. Honeypots score low.

---

### C — Evaluate First End-to-End Output + Streamlit Skeleton

- [ ] Take `submission_sample.csv` from A — run `evaluate.py` against golden set, print NDCG@10
- [ ] Run `honeypot_audit.py` — manually read top-10 profiles. Are any non-engineers in there?
- [ ] If yes: file issue to B immediately with the candidate IDs — they need to fix the domain gate
- [ ] Start `app.py` Streamlit demo skeleton:
  - File upload for a small candidates JSON
  - Button: "Run ranking"
  - Show ranked table with candidate ID, score, reasoning
- [ ] Write `submission_metadata.yaml` using the template — fill all required fields

**Deliverable:** NDCG@10 score printed for first end-to-end run. Streamlit app loads and shows a table. `submission_metadata.yaml` complete.

---

### Tips for Day 3
- A: Don't wait for full embeddings to finish before testing `rank.py`. Run it on sample first.
- B: Tune weights empirically — change one weight, rerun on golden set, see if NDCG@10 goes up or down.
- C: If you find a non-engineer in the top-10, that's not a bug report — that's the most important signal of the day. Fix it with B now.

### EOD Checkpoint — Day 3
- [ ] `rank.py` runs end-to-end on sample without crashing (A)
- [ ] `submission_sample.csv` passes `validate_submission.py` with 0 errors (A + C)
- [ ] Feature vector has 12+ features (B)
- [ ] NDCG@10 printed — even if it's a bad score, a number exists (C)
- [ ] No non-engineer in top-5 of sample ranking (B + C)
- [ ] Full `candidates.jsonl` embedding started or completed (A)

---

## Day 4 — Quality & LightGBM

### Goal
Switch from weighted-sum to LightGBM LTR. Improve NDCG@10. Run on full dataset for the first time.

---

### A — Full Data Pipeline + Constraint Testing

- [ ] Confirm `artifacts/embeddings.npy` is complete — check shape is `(n_full_candidates, 768)`
- [ ] Run `rank.py` on the **full** `candidates.jsonl` — measure wall-clock time and peak RAM
- [ ] If runtime > 5 minutes: profile with `cProfile`, find the slow line, fix it
  - Common fix: stop loading all 100K candidate records — only load top-50 full records
- [ ] If RAM > 16 GB: convert embeddings to fp16 if not already, use Polars streaming for metadata
- [ ] Add early-stop optimization: when streaming `candidates.jsonl` for the top-50 records, stop once all 50 are found
- [ ] Run with `CUDA_VISIBLE_DEVICES=""` to confirm CPU-only

**Deliverable:** `rank.py` runs on full data in under 5 minutes, under 16 GB RAM, confirmed CPU-only.

---

### B — LightGBM LambdaMART Training

- [ ] Expand golden set with C to 50 candidates (hand-label 20 more from full data)
- [ ] Write `src/train_ltr.py`:
  - Load golden set labels
  - Build feature matrix from `build_feature_vector()` for each labeled candidate
  - Train `lgb.train()` with `objective=lambdarank`, `metric=ndcg`, `group=[n_candidates]`
  - Save to `artifacts/ltr_model.txt`
  - Print top-5 features by gain importance
- [ ] Update `scorer.py` to use `lgb.Booster.predict()` instead of weighted sum
- [ ] Keep weighted sum as a fallback if LTR model file doesn't exist
- [ ] Compare NDCG@10: weighted sum vs LTR — does LTR improve it?

**Deliverable:** `artifacts/ltr_model.txt` saved. NDCG@10 comparison logged. LTR integrated in `scorer.py`.

---

### C — Reasoning Quality Pass + Demo Polish

- [ ] Run Stage-4 reasoning check manually on 10 sampled rows from the current ranking:
  - Does each reasoning mention specific facts (title, YOE, company, named skills)?
  - Is every claim in the reasoning actually in the candidate's profile?
  - Does the tone match the rank? (Rank-5 should sound positive, rank-95 should acknowledge gaps)
  - Are the 10 strings substantively different from each other?
- [ ] Fix any issues found in `reasoning.py` — add more variation templates, add more fact-pulling
- [ ] Polish Streamlit demo: add score breakdown per candidate (show individual feature scores), add "why this ranking?" explanation panel
- [ ] Deploy to HuggingFace Spaces or Streamlit Cloud — confirm it loads and runs

**Deliverable:** 10-row reasoning audit passed. Streamlit demo live at a public URL.

---

### Tips for Day 4
- A: The most common runtime issue is reading all 100K records to find 50. Fix this with early-stop.
- B: Don't panic if LTR barely improves over weighted sum on 50 labels — 50 labels is very few. The architecture is still correct and defensible.
- C: If a reasoning string says a candidate has a skill that isn't in their profile — that's hallucination, which causes Stage 4 failure. Zero tolerance.

### EOD Checkpoint — Day 4
- [ ] Full pipeline runs under 5 min, under 16 GB, CPU-only (A)
- [ ] `ltr_model.txt` exists and is integrated (B)
- [ ] NDCG@10 logged for both weighted-sum and LTR (B + C)
- [ ] 10-row reasoning audit: 0 hallucinations found (C)
- [ ] Demo live at public URL (C)

---

## Day 5 — Hardening, Audits & Submission Prep

### Goal
Lock the submission. Run all audits. Fix the last issues. Everything is submission-ready by EOD.

---

### A — Reproduce From Scratch + Final rank.py

- [ ] Delete `artifacts/` folder (keep a backup)
- [ ] Run `precompute.py` from scratch in a clean virtual environment — confirm it rebuilds correctly
- [ ] Run `rank.py` from scratch — confirm it produces identical output
- [ ] Write the one-command reproduce instruction in `README.md`:
  ```
  python src/precompute.py
  python src/rank.py --candidates ./data/candidates.jsonl --out ./team_xxx.csv
  ```
- [ ] Confirm no network calls at ranking time: run with `--no-internet` or in Docker with no network
- [ ] Write `Dockerfile` (optional but strong) for Stage 3 reproduction

**Deliverable:** Clean-environment reproduce works. README has one-command instructions. Docker optionally works.

---

### B — Final Feature Audit + Ablation

- [ ] Run ablation: score all 50 golden candidates with behavioral signals OFF (set `behavioral_multiplier=0`) — does NDCG@10 drop?
- [ ] Run ablation: score with `domain_gate` OFF — does a non-engineer appear in top 10?
- [ ] Write results to `eval/ablation_results.txt` — this is deck material and Stage 5 defense material
- [ ] Final check on `role_model.yaml` — does it accurately capture the JD? Would you hire the people it ranks top-5?
- [ ] Add `consulting_only_disqualifier` final check: any candidate whose entire career is at consulting firms should never be in top-20

**Deliverable:** `eval/ablation_results.txt` with 2 ablation results. Final `role_model.yaml` signed off.

---

### C — Final Submission Package

- [ ] Run `validate_submission.py` on the **final** `team_xxx.csv` — must pass with zero issues
- [ ] Run `honeypot_audit.py` — manually inspect top-20 candidates, confirm no impossible profiles
- [ ] Confirm scores are strictly non-increasing (rank 1 score ≥ rank 2 ≥ ... ≥ rank 100)
- [ ] Confirm all candidate IDs are in `CAND_XXXXXXX` format and exist in the dataset
- [ ] Write the "missed candidate" demo: find one candidate who ranks highly in your system but would be buried by pure BM25/keyword search — document why they're actually a fit
- [ ] Final repo check: clean commit history (multiple real commits, not one dump), no secrets, no large files without LFS

**Deliverable:** `team_xxx.csv` passes validator. Missed-candidate demo documented. Repo is clean.

---

### Tips for Day 5
- A: The clean-env reproduce is the single most important thing you do today. If it breaks, you fail Stage 3.
- B: Ablations are not just extra credit — they're Stage 5 interview prep. You'll be asked "how do you know behavioral signals help?"
- C: Check score monotonicity programmatically: `assert all(s[i] >= s[i+1] for i in range(len(s)-1))`.

### EOD Checkpoint — Day 5
- [ ] Clean-env reproduce works end-to-end (A)
- [ ] `validate_submission.py` passes on final CSV: 0 errors (C)
- [ ] No honeypots in top-20 (C)
- [ ] Ablation results logged (B)
- [ ] Missed-candidate demo written (C)
- [ ] Repo has real commit history with multiple authors (everyone)

---

## Day 6 — Final Polish & Submit

### Goal
Deck, README polish, final submission upload. No new features. Fix only critical bugs found in Day 5 testing.

---

### A — README + Submission Metadata

- [ ] Write final `README.md`:
  - Architecture overview (2 paragraphs)
  - One-command reproduce
  - Dependencies and install
  - Headline metric (NDCG@10 vs baseline)
  - Fairness statement (matching is attribute-blind)
- [ ] Fill `submission_metadata.yaml` — all fields, including compute environment and AI tools declared
- [ ] Final git push — confirm remote is up to date
- [ ] Share GitHub link with C for portal submission

---

### B — Deck Slides

- [ ] Slide 1: Problem + thesis (why keyword search fails)
- [ ] Slide 2: Architecture diagram (two-phase: precompute + ranking)
- [ ] Slide 3: Feature groups — what we score and why
- [ ] Slide 4: Evaluation — NDCG@10, ablation table showing behavioral signals work
- [ ] Slide 5: Missed-candidate demo — show the candidate keyword search would miss
- [ ] Slide 6: Scale + cost story (why this works at 100K, not just 100)
- [ ] Export as PDF

---

### C — Portal Submission

- [ ] Final `validate_submission.py` run: zero errors confirmed
- [ ] Upload `team_xxx.csv` to portal
- [ ] Fill portal metadata form: team name, contact info, GitHub URL, sandbox URL, AI tools declared
- [ ] Submit
- [ ] Screenshot the submission confirmation

---

### Tips for Day 6
- Don't add any new features today — only fix critical validator failures.
- If the sandbox demo breaks, fix it before submitting — it's a mandatory field.
- Declare AI tools honestly in the portal. The interview will ask about your workflow.

### EOD Checkpoint — Day 6 (= Submission)
- [ ] `team_xxx.csv` uploaded to portal
- [ ] Portal metadata fully filled
- [ ] GitHub repo accessible to organizers
- [ ] Sandbox/demo link live and working
- [ ] Deck PDF attached or linked
- [ ] Screenshot of submission confirmation saved

---

## Shared Tips & Suggestions

### Communication
- Daily standup: 10 minutes max. Each person says what they finished, what they're starting, and any blockers.
- Use a shared notes doc (Google Doc / Notion) to log interface contracts — e.g., "B's `build_feature_vector()` takes `(dict, dict, float)` and returns `dict`". This prevents integration surprises on Day 3.
- When B changes a feature name, tell A and C immediately — `scorer.py` and `reasoning.py` both depend on feature keys.

### Git Discipline
- Commit every time something works, not just at EOD. Stage-4 checks for real iteration in git history.
- Use branches: `feat/pipeline-a`, `feat/features-b`, `feat/eval-c`. Merge to main at Day 3 integration.
- Write real commit messages: `"add consulting_penalty feature, tested on 5 samples"` not `"update"`.

### Testing Strategy
- Always test on `sample_candidates.json` first — never debug on 475 MB.
- Use `assert` statements liberally in data loading: `assert embeddings.shape[1] == 768`, `assert not np.any(np.isnan(scores))`.
- Keep a `scratch.ipynb` notebook for experiments — don't experiment in production files.

### The 3 Things That Cause Competition Failure
1. `validate_submission.py` failing — prevents scoring entirely
2. Honeypots in top-10 — disqualification at Stage 3
3. Hallucinated reasoning — failure at Stage 4 manual review

Fix these three before anything else.

---

## Troubleshooting Table

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `validate_submission.py` fails: wrong row count | Fewer than 100 scored candidates | Ensure at least 100 candidates are retrieved. If pool < 100, pad with lowest-scored candidates |
| `validate_submission.py` fails: scores not monotonic | Scores not sorted before writing | Sort by score descending, then enforce: `score[i] = min(score[i], score[i-1])` |
| `validate_submission.py` fails: duplicate candidate_id | Candidate appears twice in retrieval | Deduplicate by ID before writing |
| `validate_submission.py` fails: ID not in dataset | Typo or test IDs used | Load all IDs from `candidates.jsonl` first, filter retrieved IDs against that set |
| NaN similarity scores | Embeddings not normalized | Re-run with `normalize_embeddings=True`, recheck with `np.linalg.norm(emb[0])` ≈ 1.0 |
| Embedding shape mismatch | JD vector has wrong dimensions | Print both shapes before dot product: `(n, 768)` × `(768, 1)` |
| `rank.py` takes > 5 minutes | Reading all 100K records to find top-50 | Add early-stop: break once all top-50 IDs are found while streaming |
| RAM > 16 GB | Embeddings loaded as fp32 | Load as fp16, convert to fp32 only for dot product |
| Non-engineers in top 10 | Domain gate not applied | Check `title_disqualified` is in scorer weights with value -1.0 |
| Honeypots in top 10 | Consistency score not penalizing them | Print `consistency_score` for known honeypots — should be < 0.3 |
| `KeyError` in feature_builder | Missing field in candidate record | Replace all direct dict access with `.get()` and sensible defaults |
| LightGBM error: group size mismatch | `group` parameter doesn't sum to `len(X)` | `group = [len(X)]` for single-query training |
| Reasoning strings are identical | Template not varying per candidate | Add rank-tier variation and pull different facts per candidate |
| Reasoning hallucination | Claiming skills not in profile | Only pull facts directly from the candidate dict — never infer |
| BM25 tokenization issues | Punctuation breaking tokens | Use `re.findall(r'\b[a-z0-9]+\b', text.lower())` as tokenizer |
| Demo crashes on HuggingFace | Large model file in repo | Store model in `artifacts/`, not in `src/`. Use relative paths. |
| Git history looks like one dump | All committed at once on final day | Commit every working milestone — start this from Day 1 |
| Stage 3 reproduction fails | Code depends on a local path | Use `Path(__file__).parent` for relative paths. Test in a clean virtual environment |
