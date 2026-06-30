# Dataset Findings

Empirical observations about `data/candidates.jsonl` that affect feature engineering
and LTR training decisions. Logged here rather than inferred fresh each time someone
investigates a feature's gain importance or training signal quality.

---

## Career history descriptions are templated — only 44 unique snippets across 100k candidates

**Finding:** A fingerprint check (MD5 of the first 100 characters, lowercased) of every
`career_history[].description` field across all 100,000 candidates found only **44 unique
fingerprints total**. 43 of those 44 appear in more than 3 candidates each — the top 9
fingerprints each appear in **~25,000+ candidates**.

In other words, career-history narrative text is drawn from a small fixed library of
canned snippets and reused at massive scale, not written per-candidate. A filter requiring
"both career_history entries to be non-templated" applied to a 77-candidate pool of
top-ranked, unlabeled candidates returned **zero survivors** — essentially the entire
corpus exhibits this pattern, not just a handful of suspicious profiles.

**Implication:** career-description keyword/phrase matching has limited discriminative
power for ranking, because most candidates' narrative text is interchangeable boilerplate
rather than a genuine signal of what they actually did. This explains why
`production_ml_score` and `domain_alignment` — both of which lean heavily on career-text
keyword density — showed lower LambdaMART gain than expected in earlier feature-importance
audits, despite being weighted heavily in `role_model.yaml` and the README's stated feature
table.

**Practical consequence for LTR feature weighting:** structured signals that aren't subject
to this templating — `current_title`, `skills` (with `duration_months` and `proficiency`),
`redrob_signals` behavioral fields, `education.tier` — should be expected to carry more
genuine discriminative signal than career-text keyword matching alone, and gain-importance
audits should be read with this in mind rather than treating low career-text-feature gain
as purely a training-set-size artifact.

**How this was found:** during targeted golden-set label=3 expansion (see git history
around `eval/golden_set.csv`), a candidate audit round manually flagged ~8 recurring
templates by eye. A mechanical fingerprint pass to systematically filter them out for a
final expansion round instead revealed the templating covers nearly the entire corpus
(43/44 fingerprints, not 8) — there are only 2 known candidates in the labeled set with
career text outside this template pool: `CAND_0005538` and `CAND_0080766`, both labeled
`relevance_label=3`.
