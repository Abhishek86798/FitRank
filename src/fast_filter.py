"""
Stage 2 of the cascade pipeline — cheap rule-based gates applied BEFORE LTR scoring.
Drops obvious non-fits to reduce the expensive feature-building + scoring work.
"""

from __future__ import annotations

import logging

from src.feature_builder import (
    _title_disqualified,
    _consistency_score,
    _domain_alignment,
)

logger = logging.getLogger(__name__)

_SAFETY_FLOOR = 0.30  # never drop more than 70% of input candidates

_DEFAULT_GATES = {
    "title_disqualified": True,
    "consistency_score": True,
    "domain_alignment": True,
    "years_of_experience": True,
}


def fast_filter(
    candidates: list[dict],
    role_model: dict,
) -> tuple[list[dict], dict]:
    """
    Stage 2 — cheap rule-based gates, no ML. Drops obvious non-fits
    before expensive LTR scoring. Returns (survivors, drop_log).

    Gates in order (each toggleable via role_model['cascade_gates']):
      1. title_disqualified: disqualifying title AND no ML role in career_history → drop
      2. consistency_score < 0.2: honeypot signal → drop
      3. domain_alignment == 0: zero NLP/IR/ranking/search keywords in career → drop
      4. years_of_experience < 2: too junior → drop

    SAFETY FLOOR: never drop more than 70% of input. If a gate would
    breach this, stop applying gates and keep remaining candidates.

    Each gate logs: "Gate {name}: dropped {n}, {remaining} survive"
    drop_log is a dict {gate_name: count_dropped} for the deck.
    """
    gates_enabled: dict[str, bool] = {**_DEFAULT_GATES, **role_model.get("cascade_gates", {})}

    survivors = list(candidates)
    drop_log: dict[str, int] = {}
    floor_count = max(1, int(len(candidates) * _SAFETY_FLOOR))

    def _would_breach_floor(remaining_after: int) -> bool:
        return remaining_after < floor_count

    # Gate 1: title_disqualified
    if gates_enabled.get("title_disqualified", True):
        kept = []
        dropped = 0
        for cand in survivors:
            if _title_disqualified(cand, role_model) < 0:
                dropped += 1
            else:
                kept.append(cand)
        if _would_breach_floor(len(kept)):
            logger.info(
                "Gate title_disqualified: skipped (would breach safety floor), "
                "%d survive", len(survivors)
            )
        else:
            drop_log["title_disqualified"] = dropped
            survivors = kept
            logger.info(
                "Gate title_disqualified: dropped %d, %d survive", dropped, len(survivors)
            )

    # Gate 2: consistency_score < 0.2
    if gates_enabled.get("consistency_score", True):
        kept = []
        dropped = 0
        for cand in survivors:
            if _consistency_score(cand) < 0.2:
                dropped += 1
            else:
                kept.append(cand)
        if _would_breach_floor(len(kept)):
            logger.info(
                "Gate consistency_score: skipped (would breach safety floor), "
                "%d survive", len(survivors)
            )
        else:
            drop_log["consistency_score"] = dropped
            survivors = kept
            logger.info(
                "Gate consistency_score: dropped %d, %d survive", dropped, len(survivors)
            )

    # Gate 3: domain_alignment == 0
    if gates_enabled.get("domain_alignment", True):
        kept = []
        dropped = 0
        for cand in survivors:
            if _domain_alignment(cand) == 0.0:
                dropped += 1
            else:
                kept.append(cand)
        if _would_breach_floor(len(kept)):
            logger.info(
                "Gate domain_alignment: skipped (would breach safety floor), "
                "%d survive", len(survivors)
            )
        else:
            drop_log["domain_alignment"] = dropped
            survivors = kept
            logger.info(
                "Gate domain_alignment: dropped %d, %d survive", dropped, len(survivors)
            )

    # Gate 4: years_of_experience < 2
    if gates_enabled.get("years_of_experience", True):
        kept = []
        dropped = 0
        for cand in survivors:
            yoe = cand.get("profile", {}).get("years_of_experience") or 0.0
            if yoe < 2:
                dropped += 1
            else:
                kept.append(cand)
        if _would_breach_floor(len(kept)):
            logger.info(
                "Gate years_of_experience: skipped (would breach safety floor), "
                "%d survive", len(survivors)
            )
        else:
            drop_log["years_of_experience"] = dropped
            survivors = kept
            logger.info(
                "Gate years_of_experience: dropped %d, %d survive", dropped, len(survivors)
            )

    return survivors, drop_log
