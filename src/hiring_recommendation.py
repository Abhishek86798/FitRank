"""
Hiring recommendation tiers — translates scoring artifacts into a recruiter-facing action.

Every tier decision is derived purely from the pre-computed audit dict, feature vector,
and rank; no LLM is called, no new scoring happens here.
"""

from __future__ import annotations

# Feature display labels (mirror app.py FEATURE_LABELS so messages are consistent)
_FEAT_LABELS: dict[str, str] = {
    "behavioral_multiplier": "Behavioral signal",
    "is_ml_engineer":        "ML engineer title",
    "domain_alignment":      "Domain alignment",
    "production_ml_score":   "Production ML evidence",
    "location_score":        "Location",
    "github_activity":       "GitHub activity",
    "cosine_similarity":     "Semantic similarity",
    "experience_fit_score":  "Experience fit",
    "consistency_score":     "Profile consistency",
    "consulting_penalty":    "Consulting background",
    "notice_penalty":        "Notice period",
}

_TIER_ORDER = ("Strong Hire", "Borderline", "Verify", "Pass")


def _feat_label(key: str) -> str:
    return _FEAT_LABELS.get(key, key.replace("_", " ").title())


def _has_hard_blocker(audit: dict, features: dict) -> bool:
    """True when any single factor would make an immediate screen inadvisable."""
    risk_flags = audit.get("risk_flags") or []
    notice_penalty = features.get("notice_penalty", 0.0)
    open_to_work   = features.get("behavioral_multiplier", 1.0)  # <1 when signals are weak

    for flag in risk_flags:
        fl = flag.lower()
        if "notice" in fl and ("90" in fl or "120" in fl):
            return True
        if "not flagged as open" in fl or "not open" in fl:
            return True
    if notice_penalty >= 0.75:   # >90-day notice
        return True
    return False


def _build_blockers(audit: dict, features: dict) -> list[str]:
    blockers: list[str] = []

    # Direct risk flags from the audit
    for flag in audit.get("risk_flags") or []:
        blockers.append(flag)

    # Notice period
    notice_penalty = features.get("notice_penalty", 0.0)
    if notice_penalty >= 0.75:
        meta   = audit.get("candidate_meta", {})
        days   = meta.get("notice_days")
        label  = f"{days}d" if days is not None else ">90d"
        blockers.append(f"Notice period: {label} (above 60d threshold)")

    # Location
    location_score = features.get("location_score", 1.0)
    if location_score < 0.5:
        meta = audit.get("candidate_meta", {})
        loc  = meta.get("location", "unknown location")
        blockers.append(f"Outside preferred geography ({loc})")

    # Open-to-work
    meta = audit.get("candidate_meta", {})
    if not meta.get("open_to_work", True):
        blockers.append("Not currently open to work — verify interest before outreach")

    # Remove duplicates while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for b in blockers:
        key = b.lower()[:60]
        if key not in seen:
            seen.add(key)
            deduped.append(b)
    return deduped


def _build_strengths(audit: dict, features: dict) -> list[str]:
    strengths: list[str] = []
    top_reasons = audit.get("top_reasons") or []
    cf          = audit.get("counterfactuals") or {}

    for r in top_reasons[:3]:
        feat  = r["feature"]
        drop  = r["rank_drop"]
        label = _feat_label(feat)

        if feat == "behavioral_multiplier" and drop >= 5:
            strengths.append(f"Behavioral signal load-bearing (rank drops {drop} without it)")
        elif feat == "production_ml_score" and drop >= 3:
            strengths.append("Production ML evidence verified in career descriptions")
        elif feat == "is_ml_engineer" and drop >= 3:
            strengths.append("ML engineer title confirmed — not research or data-only role")
        elif feat == "domain_alignment" and drop >= 3:
            strengths.append("Domain alignment with NLP/IR/RecSys verified")
        elif feat == "location_score" and drop >= 3:
            meta = audit.get("candidate_meta", {})
            loc  = meta.get("location", "")
            notice = meta.get("notice_days")
            loc_str = f"Preferred location ({loc})" if loc else "Preferred location"
            if notice is not None and notice <= 30:
                strengths.append(f"{loc_str}, short notice period ({notice}d)")
            else:
                strengths.append(loc_str)
        elif feat == "github_activity" and drop >= 2:
            strengths.append("Active public GitHub portfolio — verifiable code evidence")
        elif drop >= 3:
            strengths.append(f"{label} load-bearing (rank drops {drop} without it)")

    # Fallback: if fewer than 2 strengths from top_reasons, pull from high features
    if len(strengths) < 2:
        prod = features.get("production_ml_score", 0.0)
        if prod > 0.5 and "Production ML evidence verified in career descriptions" not in strengths:
            strengths.append("Production ML evidence in career history")
        loc = features.get("location_score", 0.0)
        if loc >= 1.0 and not any("location" in s.lower() for s in strengths):
            strengths.append("Preferred location match")

    return strengths[:3]


def get_hiring_recommendation(
    audit: dict,
    features: dict,
    rank: int,
) -> dict:
    """
    Returns hiring recommendation based on evidence from the audit, not raw score alone.

    {
      "tier": "Strong Hire" | "Borderline" | "Verify" | "Pass",
      "action": str,
      "color": str,          # green | amber | orange | red
      "primary_reason": str,
      "blockers": list[str],
      "strengths": list[str],
    }
    """
    score      = audit.get("base_score", 0.0)
    confidence = audit.get("confidence", 0.0)
    top_reasons = audit.get("top_reasons") or []
    top_drop    = top_reasons[0]["rank_drop"] if top_reasons else 0
    hard_blocker = _has_hard_blocker(audit, features)
    blockers     = _build_blockers(audit, features)
    strengths    = _build_strengths(audit, features)

    # ── Tier logic ────────────────────────────────────────────────────────────
    if score >= 0.97 and not hard_blocker:
        tier           = "Strong Hire"
        primary_reason = "Top-tier score with no blocking constraints"
        action         = "Schedule technical screen immediately"
    elif score >= 0.95 and confidence >= 0.60 and not hard_blocker and top_drop >= 5:
        tier           = "Strong Hire"
        primary_reason = (
            f"High score ({score:.3f}), strong confidence, "
            f"and {_feat_label(top_reasons[0]['feature'])} is load-bearing"
        )
        action = "Schedule technical screen immediately"
    elif score >= 0.88 and (len(blockers) <= 2 or confidence < 0.55):
        tier = "Borderline"
        if blockers:
            top_blocker    = blockers[0].split("—")[0].split("(")[0].strip()
            action         = f"30-minute exploratory call to verify: {top_blocker}"
            primary_reason = f"Strong score ({score:.3f}) but needs one verification step"
        else:
            action         = "30-minute exploratory call — contested band, confirm fit"
            primary_reason = f"Good score ({score:.3f}) with low confidence; rank is approximate"
    elif score >= 0.75 and blockers:
        tier           = "Verify"
        top_blocker    = blockers[0].split("—")[0].split("(")[0].strip()
        action         = f"Verify {top_blocker} before scheduling"
        primary_reason = f"Score ({score:.3f}) is adequate but specific concern must resolve first"
    elif score >= 0.75:
        tier           = "Verify"
        action         = "Request work sample or code link before scheduling"
        primary_reason = f"Score ({score:.3f}) is moderate; verify depth before investing screen time"
    else:
        tier           = "Pass"
        action         = "Not a fit for this role"
        primary_reason = (
            f"Score ({score:.3f}) below threshold"
            + (f"; {len(blockers)} blocking issue(s)" if blockers else "")
        )

    color_map = {
        "Strong Hire": "green",
        "Borderline":  "amber",
        "Verify":      "orange",
        "Pass":        "red",
    }

    return {
        "tier":           tier,
        "action":         action,
        "color":          color_map[tier],
        "primary_reason": primary_reason,
        "blockers":       blockers,
        "strengths":      strengths,
    }
