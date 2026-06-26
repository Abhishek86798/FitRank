# Converts a raw candidate dict + role_model + cosine_sim into a flat dict of features used by scorer.py.
# This is the primary driver of ranking quality — all signal engineering lives here.

from __future__ import annotations

import re
from datetime import date, datetime

# ── keyword sets (compiled once at import) ────────────────────────────────────

_ML_TITLES = re.compile(
    r"\b("
    r"machine learning engineer|ml engineer|applied (ml|ai) engineer|"
    r"research (scientist|engineer)|nlp engineer|search engineer|"
    r"recommendation(s)? (systems? )?(engineer|scientist)|"
    r"ranking engineer|retrieval engineer|"
    r"data scientist|ai engineer|"
    r"(senior |staff |principal |lead )?(ml|ai|nlp|search|recsys) (engineer|scientist)"
    r")\b",
    re.IGNORECASE,
)

# Keywords in career descriptions that signal production ML delivery
_PRODUCTION_KEYWORDS = [
    # deployment / shipping
    "deployed", "shipped", "launched", "production", "real-time", "real time",
    "serving", "live", "inference",
    # retrieval / ranking systems
    "ranking model", "ranking system", "retrieval system", "search system",
    "recommendation system", "recsys", "reranking", "re-ranking",
    "learning to rank", "ltr", "lambdamart", "xgboost", "lightgbm",
    "dense retrieval", "hybrid retrieval", "faiss", "annoy", "hnsw",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    # evaluation infra — JD treats this as hard requirement
    "ndcg", "map@", "mrr", "p@", "offline eval", "online eval",
    "a/b test", "ab test", "experiment", "offline-online",
    # embeddings
    "embedding", "sentence transformer", "bge", "e5", "openai embed",
    "vector index", "vector store", "vector db", "vector database",
    "semantic search",
]
_PRODUCTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _PRODUCTION_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Domain alignment keywords (NLP/IR/ranking density in career text)
_DOMAIN_KEYWORDS = [
    "nlp", "natural language processing",
    "information retrieval", "information extraction",
    "ranking", "search", "recommendation", "retrieval",
    "embeddings", "embedding", "semantic",
    "text classification", "named entity", "transformer",
    "bert", "llm", "language model",
]
_DOMAIN_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _DOMAIN_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Consulting firm names (exact, lowercase)
_CONSULTING_FIRMS = {
    "tcs", "tata consultancy services",
    "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "hcl technologies",
    "mphasis", "hexaware", "mindtree", "l&t infotech", "ltimindtree",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _career_descriptions(candidate: dict) -> str:
    """Concatenate all career history description fields."""
    parts = []
    for role in candidate.get("career_history", []):
        desc = role.get("description", "").strip()
        if desc:
            parts.append(desc)
    return " ".join(parts)


def _today() -> date:
    return datetime.utcnow().date()


# ── feature functions ─────────────────────────────────────────────────────────

def _experience_fit_score(candidate: dict, role_model: dict) -> float:
    """
    Score YOE against the experience band in role_model.
    Returns 1.0 if inside band, gracefully tapers outside.
    """
    yoe = candidate.get("profile", {}).get("years_of_experience") or 0.0
    band = role_model.get("experience_band", {})
    lo, hi = float(band.get("min", 5)), float(band.get("max", 9))

    if lo <= yoe <= hi:
        return 1.0
    if yoe < lo:
        # under-experienced: linear decay, 0 at half the minimum
        return max(0.0, (yoe - lo / 2) / (lo - lo / 2))
    # over-experienced: slow decay, still 0.5 at 15 years
    return max(0.0, 1.0 - (yoe - hi) / (15 - hi))


def _is_ml_engineer(candidate: dict) -> float:
    """
    1.0 if current title or any career title matches ML engineering patterns.
    0.5 if only past titles match (not current).
    0.0 otherwise.
    """
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "")
    if _ML_TITLES.search(current_title):
        return 1.0

    for role in candidate.get("career_history", []):
        if _ML_TITLES.search(role.get("title", "")):
            return 0.5  # past ML role — partial credit

    return 0.0


def _title_disqualified(candidate: dict, role_model: dict) -> float:
    """
    Hard gate: -1.0 if current title is a disqualifying non-engineer title
    AND the candidate has no ML engineering roles anywhere in career history.
    Returns 0.0 (no penalty) otherwise.
    """
    disq = [t.lower() for t in role_model.get("disqualifying_titles", [])]
    current_title = candidate.get("profile", {}).get("current_title", "").lower()

    title_is_disq = any(d in current_title for d in disq)
    if not title_is_disq:
        return 0.0

    # Redemption check: has any past career title been ML engineering?
    for role in candidate.get("career_history", []):
        if _ML_TITLES.search(role.get("title", "")):
            return 0.0  # past ML role saves them

    return -1.0


def _production_ml_score(candidate: dict) -> float:
    """
    Corroborated production-ML keyword density.

    Career description hits count at full weight (1.0).
    Skill-only hits (keyword present in skill name with duration_months > 0 but
    absent from any career description) count at 0.4× — credible but unverified.
    10 corroborated equivalent hits → 1.0.
    """
    career_text = _career_descriptions(candidate)
    career_matches = set(m.group(0).lower() for m in _PRODUCTION_RE.finditer(career_text))

    credible_skills = " ".join(
        s.get("name", "") for s in candidate.get("skills", [])
        if isinstance(s, dict)
        and s.get("proficiency") in ("advanced", "expert")
        and (s.get("duration_months") or 0) > 0
    )
    skill_matches = set(m.group(0).lower() for m in _PRODUCTION_RE.finditer(credible_skills))

    skill_only = skill_matches - career_matches
    score = len(career_matches) + 0.4 * len(skill_only)
    return min(1.0, score / 10.0)


def _domain_alignment(candidate: dict) -> float:
    """
    Corroborated NLP/IR/ranking domain keyword density.

    Career description hits count at full weight (1.0).
    Skill-only hits (keyword in skill name but NOT in any career description)
    count at 0.3× — they signal awareness but not demonstrated delivery.
    Threshold: 6 corroborated equivalent hits → 1.0.
    """
    career_text = _career_descriptions(candidate)
    career_matches = set(m.group(0).lower() for m in _DOMAIN_RE.finditer(career_text))

    skill_names = " ".join(
        s.get("name", "") for s in candidate.get("skills", [])
        if isinstance(s, dict) and s.get("proficiency") in ("advanced", "expert")
    )
    skill_matches = set(m.group(0).lower() for m in _DOMAIN_RE.finditer(skill_names))

    # Keywords that appear only in skills (not backed by career description evidence)
    skill_only = skill_matches - career_matches

    score = len(career_matches) + 0.3 * len(skill_only)
    return min(1.0, score / 6.0)


def _consulting_penalty(candidate: dict) -> float:
    """
    Fraction of career_history roles at known consulting firms.
    0.0 = no consulting; 1.0 = entire career at consulting.
    Per JD: only penalise when the ENTIRE career is consulting.
    Returned as a positive float (caller negates it via feature weight).
    """
    history = candidate.get("career_history", [])
    if not history:
        return 0.0

    consulting_months = 0
    total_months = 0
    for role in history:
        company = role.get("company", "").lower().strip()
        duration = role.get("duration_months") or 0
        total_months += duration
        if any(firm in company for firm in _CONSULTING_FIRMS):
            consulting_months += duration

    if total_months == 0:
        return 0.0

    fraction = consulting_months / total_months
    # Soft penalty that ramps up: <50% consulting barely hurts,
    # 100% consulting returns 1.0
    return round(fraction ** 1.5, 4)


def _behavioral_multiplier(candidate: dict, role_model: dict) -> float:
    """
    Composite reachability/availability score from redrob_signals.
    Weights from role_model.behavioral_weights.
    Returns 0.0–1.0.
    """
    signals = candidate.get("redrob_signals", {})
    weights = role_model.get("behavioral_weights", {})

    # --- open_to_work (binary → float) ---
    open_score = 1.0 if signals.get("open_to_work_flag") else 0.0

    # --- recruiter_response_rate (already 0–1) ---
    response_score = float(signals.get("recruiter_response_rate") or 0.0)

    # --- last_active_recency: decay over 180 days ---
    recency_score = 0.0
    last_active_str = signals.get("last_active_date")
    if last_active_str:
        try:
            last_active = date.fromisoformat(last_active_str)
            days_ago = (_today() - last_active).days
            recency_score = max(0.0, 1.0 - days_ago / 180.0)
        except ValueError:
            pass

    # --- interview_completion_rate (already 0–1) ---
    interview_score = float(signals.get("interview_completion_rate") or 0.0)

    # --- profile_completeness_score (0–100 → 0–1) ---
    completeness_score = float(signals.get("profile_completeness_score") or 0.0) / 100.0

    w_open = weights.get("open_to_work_flag", 0.25)
    w_resp = weights.get("recruiter_response_rate", 0.30)
    w_rec  = weights.get("last_active_recency_days", 0.20)
    w_int  = weights.get("interview_completion_rate", 0.15)
    w_comp = weights.get("profile_completeness_score", 0.10)

    total = (
        w_open * open_score
        + w_resp * response_score
        + w_rec  * recency_score
        + w_int  * interview_score
        + w_comp * completeness_score
    )
    # normalise by sum of weights (should already be ~1.0 but be safe)
    weight_sum = w_open + w_resp + w_rec + w_int + w_comp
    return round(total / weight_sum, 4)


def _consistency_score(candidate: dict) -> float:
    """
    Honeypot detector. Returns 0.0 (suspicious) to 1.0 (clean).
    Checks:
      1. expert/advanced skill with duration_months == 0 → red flag per skill
      2. career_history total months vs years_of_experience × 12 (large gap = suspicious)
      3. title vs description mismatch: disqualifying title but career desc full of ML terms
    """
    red_flags = 0
    checks = 0

    # Check 1: expert/advanced skills with zero duration
    skills = candidate.get("skills", [])
    for s in skills:
        if not isinstance(s, dict):
            continue
        if s.get("proficiency") in ("advanced", "expert"):
            checks += 1
            duration = s.get("duration_months")
            if duration is not None and duration == 0:
                red_flags += 1

    # Check 2: career duration vs stated YOE
    history = candidate.get("career_history", [])
    total_career_months = sum(r.get("duration_months") or 0 for r in history)
    yoe = candidate.get("profile", {}).get("years_of_experience") or 0.0
    yoe_months = yoe * 12
    if yoe_months > 0:
        checks += 1
        # Gap larger than 36 months (3 years) in either direction is suspicious
        gap = abs(total_career_months - yoe_months)
        if gap > 36:
            red_flags += 0.5  # softer penalty — gaps happen (gaps, overlaps)

    # Check 3: current title is disqualifying but career descriptions are ML-heavy
    # (ML in descriptions but non-ML title = possible honeypot trying to game embeddings)
    current_title = candidate.get("profile", {}).get("current_title", "").lower()
    NON_ML_TITLES = {
        "marketing manager", "hr manager", "content writer", "graphic designer",
        "business analyst", "operations manager", "civil engineer",
        "mechanical engineer", "accountant", "project manager",
        "customer support", "sales",
    }
    title_is_non_ml = any(t in current_title for t in NON_ML_TITLES)
    if title_is_non_ml:
        checks += 1
        career_text = _career_descriptions(candidate)
        domain_matches = len(set(m.group(0).lower() for m in _DOMAIN_RE.finditer(career_text)))
        if domain_matches >= 5:
            # Non-ML title but descriptions stuffed with ML keywords → suspicious
            red_flags += 1

    if checks == 0:
        return 1.0

    flag_rate = red_flags / checks
    return round(max(0.0, 1.0 - flag_rate), 4)


def _location_score(candidate: dict, role_model: dict) -> float:
    """
    1.0 = preferred Indian city
    0.7 = other Indian city + willing to relocate
    0.5 = other Indian city, not relocating
    0.3 = outside India + willing to relocate
    0.0 = outside India, not willing to relocate
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    loc_prefs = role_model.get("location_preferences", {})

    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing = signals.get("willing_to_relocate", False)

    preferred = [c.lower() for c in loc_prefs.get("preferred_cities", [])]

    if any(city in location for city in preferred):
        return 1.0
    if country == "india":
        return 0.7 if willing else 0.5
    return 0.3 if willing else 0.0


def _notice_penalty(candidate: dict, role_model: dict) -> float:
    """
    Returns a positive penalty value 0.0–1.0 (caller negates via feature weight).
    0.0  = ≤30 days (ideal)
    0.25 = 31–60 days
    0.5  = 61–90 days
    0.75 = 91–120 days
    1.0  = >120 days
    """
    signals = candidate.get("redrob_signals", {})
    notice = signals.get("notice_period_days") or 0

    if notice <= 30:
        return 0.0
    if notice <= 60:
        return 0.25
    if notice <= 90:
        return 0.5
    if notice <= 120:
        return 0.75
    return 1.0


def _github_activity(candidate: dict) -> float:
    """
    Normalise github_activity_score (0–100, or -1 if no GitHub linked) → 0–1.
    -1 (no GitHub) maps to 0.0.
    """
    score = candidate.get("redrob_signals", {}).get("github_activity_score", -1)
    if score is None or score < 0:
        return 0.0
    return min(1.0, float(score) / 100.0)


# ── public API ────────────────────────────────────────────────────────────────

def build_feature_vector(
    candidate: dict,
    role_model: dict,
    cosine_sim: float,
) -> dict[str, float]:
    """
    Build a flat feature dict for a candidate.

    Parameters
    ----------
    candidate   : raw candidate dict from candidates.jsonl
    role_model  : parsed role_model.yaml dict
    cosine_sim  : pre-computed cosine similarity from retriever.py (0–1)

    Returns
    -------
    dict with all feature keys as floats.
    title_disqualified is -1.0 or 0.0 (used as a hard gate by scorer.py).
    All other features are in 0.0–1.0 unless noted.
    """
    return {
        "cosine_similarity":    round(float(cosine_sim), 6),
        "experience_fit_score": _experience_fit_score(candidate, role_model),
        "is_ml_engineer":       _is_ml_engineer(candidate),
        "title_disqualified":   _title_disqualified(candidate, role_model),
        "production_ml_score":  _production_ml_score(candidate),
        "domain_alignment":     _domain_alignment(candidate),
        "consulting_penalty":   _consulting_penalty(candidate),
        "behavioral_multiplier":_behavioral_multiplier(candidate, role_model),
        "consistency_score":    _consistency_score(candidate),
        "location_score":       _location_score(candidate, role_model),
        "notice_penalty":       _notice_penalty(candidate, role_model),
        "github_activity":      _github_activity(candidate),
    }
