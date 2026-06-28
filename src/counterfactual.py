"""
Counterfactual decision-audit engine.

For each top candidate, computes how their rank changes when each scoring
feature is individually masked to 0. Returns a structured audit dict that
explains *why* a candidate ranked where they did.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.scorer import LTRScorer


# Features that can be meaningfully masked (hard-gate flags are excluded because
# masking them would produce misleading "what-if" answers for flagged candidates).
_MASKABLE_FEATURES = [
    "cosine_similarity",
    "experience_fit_score",
    "is_ml_engineer",
    "production_ml_score",
    "domain_alignment",
    "consulting_penalty",
    "behavioral_multiplier",
    "consistency_score",
    "location_score",
    "notice_penalty",
    "github_activity",
]

# Risk-flag templates — no LLM, purely rule-based from candidate_record fields.
_NOTICE_THRESHOLD_LONG = 60   # days
_INACTIVE_THRESHOLD_DAYS = 90


def _base_rank(candidate_id: str, all_scored: list[tuple[str, float]]) -> int:
    """Return 1-based rank of candidate_id in all_scored (sorted best-first)."""
    for i, (cid, _) in enumerate(all_scored):
        if cid == candidate_id:
            return i + 1
    raise ValueError(f"candidate_id {candidate_id!r} not found in all_scored")


def _rank_after_mask(
    candidate_id: str,
    masked_features: dict[str, float],
    scorer: "LTRScorer",
    all_scored: list[tuple[str, float]],
) -> int:
    """
    Re-score a single candidate with masked_features, substitute into all_scored,
    re-sort, and return the new 1-based rank.
    """
    new_score = scorer.score(masked_features)
    new_ranked: list[tuple[str, float]] = []
    for cid, score in all_scored:
        if cid == candidate_id:
            new_ranked.append((cid, new_score))
        else:
            new_ranked.append((cid, score))
    # Sort best-first, ties broken by candidate_id ascending (validator rule)
    new_ranked.sort(key=lambda x: (-x[1], x[0]))
    for i, (cid, _) in enumerate(new_ranked):
        if cid == candidate_id:
            return i + 1
    raise RuntimeError("candidate_id disappeared after re-sort")  # should never happen


def detect_tied_bands(
    all_scored: list[tuple[str, float]],
    epsilon: float = 0.01,
) -> list[list[str]]:
    """
    Group consecutively-ranked candidates whose pairwise score gap < epsilon
    into 'contested bands' — ranks statistically indistinguishable.

    Parameters
    ----------
    all_scored : [(candidate_id, score), ...] sorted best-first
    epsilon    : max score gap between adjacent candidates to be considered tied

    Returns
    -------
    List of bands. Each band is a list of candidate_ids whose scores are
    within epsilon of their immediate neighbours. Single-member groups are
    *not* returned — only genuine multi-candidate bands.
    """
    if not all_scored:
        return []

    bands: list[list[str]] = []
    current_band: list[str] = [all_scored[0][0]]

    for i in range(1, len(all_scored)):
        prev_score = all_scored[i - 1][1]
        curr_score = all_scored[i][1]
        if abs(prev_score - curr_score) < epsilon:
            current_band.append(all_scored[i][0])
        else:
            if len(current_band) > 1:
                bands.append(current_band)
            current_band = [all_scored[i][0]]

    if len(current_band) > 1:
        bands.append(current_band)

    return bands


def _find_tied_band(candidate_id: str, bands: list[list[str]]) -> list[str] | None:
    """Return the band containing candidate_id, or None if not in any tied band."""
    for band in bands:
        if candidate_id in band:
            return band
    return None


def _build_risk_flags(candidate_record: dict, role_model: dict) -> list[str]:
    """
    Return a list of human-readable risk flags derived purely from candidate_record
    and role_model fields. Zero LLM calls.
    """
    flags: list[str] = []
    signals = candidate_record.get("redrob_signals", {})
    profile = candidate_record.get("profile", {})

    # Long notice period
    notice = signals.get("notice_period_days") or 0
    max_days = role_model.get("notice_period_max_days", 60)
    if notice > max_days:
        flags.append(
            f"Long notice period ({notice}d) exceeds preferred maximum ({max_days}d)."
        )

    # Not open to work
    if not signals.get("open_to_work_flag"):
        flags.append("Candidate is not flagged as open to work on platform.")

    # Outside preferred geography
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing = signals.get("willing_to_relocate", False)
    preferred = [c.lower() for c in role_model.get("location_preferences", {}).get("preferred_cities", [])]
    in_preferred = any(city in location for city in preferred)
    if not in_preferred:
        if country != "india":
            if not willing:
                flags.append(
                    f"Located outside India ({profile.get('location', 'unknown')}) with no relocation intent."
                )
            else:
                flags.append(
                    f"Located outside India ({profile.get('location', 'unknown')}); relocation required."
                )
        else:
            if not willing:
                flags.append(
                    f"Outside preferred hiring cities ({profile.get('location', 'unknown')}); not willing to relocate."
                )

    # Profile inactive
    last_active_str = signals.get("last_active_date")
    if last_active_str:
        from datetime import date, datetime
        try:
            last_active = datetime.strptime(last_active_str[:10], "%Y-%m-%d").date()
            days_ago = (date.today() - last_active).days
            if days_ago >= _INACTIVE_THRESHOLD_DAYS:
                flags.append(
                    f"Profile inactive for ~{days_ago // 30} month(s); outreach risk."
                )
        except ValueError:
            pass

    # Consulting-heavy career
    history = candidate_record.get("career_history", [])
    consulting_firms = {f.lower() for f in role_model.get("disqualifying_company_types", [])}
    consulting_months = sum(
        (r.get("duration_months") or 0)
        for r in history
        if any(f in r.get("company", "").lower() for f in consulting_firms)
    )
    total_months = sum(r.get("duration_months") or 0 for r in history)
    if total_months > 0 and (consulting_months / total_months) > 0.5:
        flags.append(
            "Majority of career at IT-services/consulting firms "
            "(JD flags this as a negative signal)."
        )

    # YOE outside experience band
    yoe = profile.get("years_of_experience") or 0.0
    band = role_model.get("experience_band", {})
    lo, hi = float(band.get("min", 5)), float(band.get("max", 9))
    if yoe < lo:
        flags.append(f"Under-experienced: {yoe:.1f} yrs vs. preferred {lo:.0f}-{hi:.0f} yrs.")
    elif yoe > hi + 4:
        flags.append(f"Significantly over-experienced: {yoe:.1f} yrs may signal seniority mismatch.")

    return flags


def explain_candidate(
    candidate_id: str,
    features: dict[str, float],
    scorer: "LTRScorer",
    all_scored: list[tuple[str, float]],
    candidate_record: dict,
    role_model: dict,
    tied_bands: list[list[str]] | None = None,
) -> dict:
    """
    Return a decision-audit dict for one candidate.

    Parameters
    ----------
    candidate_id    : the candidate being explained
    features        : pre-built feature dict from build_feature_vector()
    scorer          : LTRScorer (or weighted-sum fallback) instance
    all_scored      : full [(candidate_id, score), ...] list, sorted best-first.
                      Used to compute ranks before/after masking.
    candidate_record: raw candidate dict from candidates.jsonl
    role_model      : parsed role_model.yaml dict
    tied_bands      : pre-computed bands from detect_tied_bands(); if None,
                      tied_band will be null in the output.

    Returns
    -------
    dict with keys:
      candidate_id   : str
      base_rank      : int
      base_score     : float
      tied_band      : list[str] | None  — band members if in a contested band
      counterfactuals: {feature: {rank_if_removed, rank_drop, score_drop}}
      top_reasons    : [{"feature", "rank_drop", "score_drop"}, ...]  (top 3 by rank_drop)
      confidence     : float in [0, 1]
      risk_flags     : list[str]
    """
    base_score = scorer.score(features)
    base_rank = _base_rank(candidate_id, all_scored)

    tied_band = _find_tied_band(candidate_id, tied_bands) if tied_bands is not None else None

    scores_only = [s for _, s in all_scored]
    score_std = statistics.stdev(scores_only) if len(scores_only) > 1 else 1.0

    # Counterfactual: mask each feature individually
    counterfactuals: dict[str, dict] = {}
    for feat in _MASKABLE_FEATURES:
        if feat not in features:
            continue
        masked = dict(features)
        masked[feat] = 0.0
        masked_score = scorer.score(masked)
        new_rank = _rank_after_mask(candidate_id, masked, scorer, all_scored)
        counterfactuals[feat] = {
            "rank_if_removed": new_rank,
            "rank_drop": new_rank - base_rank,          # positive = fell in rank
            "score_drop": round(base_score - masked_score, 6),
        }

    # Top 3 reasons: features whose removal causes the largest rank drop
    top_reasons = sorted(
        [
            {
                "feature": feat,
                "rank_drop": cf["rank_drop"],
                "score_drop": cf["score_drop"],
            }
            for feat, cf in counterfactuals.items()
            if cf["rank_drop"] > 0
        ],
        key=lambda x: (-x["rank_drop"], -x["score_drop"]),
    )[:3]

    # Confidence: margin to next-ranked candidate, normalised to [0, 1]
    # Find the score of the candidate ranked one below (base_rank + 1)
    if base_rank < len(all_scored):
        next_score = all_scored[base_rank][1]   # all_scored is 0-indexed, base_rank is 1-indexed
    else:
        next_score = 0.0
    gap = base_score - next_score
    confidence = min(0.99, 0.5 + gap / (2 * score_std)) if score_std > 0 else 0.5
    confidence = round(max(0.0, confidence), 4)

    risk_flags = _build_risk_flags(candidate_record, role_model)

    return {
        "candidate_id": candidate_id,
        "base_rank": base_rank,
        "base_score": round(base_score, 6),
        "tied_band": tied_band,
        "counterfactuals": counterfactuals,
        "top_reasons": top_reasons,
        "confidence": confidence,
        "risk_flags": risk_flags,
    }
