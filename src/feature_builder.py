# Converts a raw candidate dict + role_model + cosine_sim into a flat dict of features used by scorer.py.
# This is the primary driver of ranking quality — all signal engineering lives here.

from __future__ import annotations

import re
from datetime import date, datetime

from src.ownership_classifier import ownership_score as _ownership_score

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


_RECENT_MONTHS = 36          # roles ending within this window get the recency boost
_RECENCY_WEIGHT = 3.0        # multiplier for recent roles


def _role_recency_weight(role: dict) -> float:
    """
    Returns _RECENCY_WEIGHT (3.0) if the role ended within _RECENT_MONTHS of
    today, 1.0 otherwise.  Current roles (is_current=True or end_date=None)
    are always treated as recent.
    """
    if role.get("is_current") or not role.get("end_date"):
        return _RECENCY_WEIGHT
    try:
        end = datetime.strptime(role["end_date"][:10], "%Y-%m-%d").date()
        months_ago = (_today().year - end.year) * 12 + (_today().month - end.month)
        return _RECENCY_WEIGHT if months_ago <= _RECENT_MONTHS else 1.0
    except (ValueError, KeyError):
        return 1.0


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
    Recency-weighted production-ML keyword density, modulated by ownership intent.

    Per role:
      keyword_hits  = count of unique production-ML keyword matches
      ownership     = ownership_score(description, DOMAIN_KEYWORDS)  [0–1]
      contribution  = recency_weight × keyword_hits × (0.5 + 0.5 × ownership)

    Effect: same keywords but peripheral framing ("tested the ranking API") yield
    half the score of genuine delivery framing ("built the ranking system").

    Skill-only hits (advanced/expert skill with duration > 0, keyword absent from
    all career descriptions) retain their flat 0.4 bonus — no per-role timestamps
    to anchor ownership context, so we cannot penalise them.

    Normalisation: 10 equivalent career-hit units → 1.0.
    """
    all_career_keywords: set[str] = set()
    weighted_career_score = 0.0

    for role in candidate.get("career_history", []):
        desc = role.get("description", "").strip()
        if not desc:
            continue
        w       = _role_recency_weight(role)
        matches = set(m.group(0).lower() for m in _PRODUCTION_RE.finditer(desc))
        if not matches:
            continue

        # Ownership multiplier: 0.5 (passive) … 1.0 (full owner)
        intent   = _ownership_score(desc, _DOMAIN_KEYWORDS)
        own_mult = 0.5 + 0.5 * intent

        weighted_career_score += w * len(matches) * own_mult
        all_career_keywords   |= matches

    # Skill-only bonus unchanged — no description context available
    credible_skills = " ".join(
        s.get("name", "") for s in candidate.get("skills", [])
        if isinstance(s, dict)
        and s.get("proficiency") in ("advanced", "expert")
        and (s.get("duration_months") or 0) > 0
    )
    skill_matches = set(m.group(0).lower() for m in _PRODUCTION_RE.finditer(credible_skills))
    skill_only    = skill_matches - all_career_keywords

    score = weighted_career_score + 0.4 * len(skill_only)
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


def _recency_score(candidate: dict) -> float:
    """
    Step-function availability score from last_active_date.
    >180 days inactive = effectively unavailable (per redrob_signals_doc.md).
    """
    signals = candidate.get("redrob_signals", {})
    last_active_str = signals.get("last_active_date")
    if not last_active_str:
        return 0.5  # unknown → neutral
    try:
        last_active = date.fromisoformat(last_active_str)
        days = (_today() - last_active).days
    except ValueError:
        return 0.5
    if days < 30:
        return 1.0
    if days < 90:
        return 0.75
    if days < 180:
        return 0.4
    return 0.1


def _response_speed_score(candidate: dict) -> float:
    """
    avg_response_time_hours → responsiveness score.
    Absent or -1 → neutral 0.5.
    """
    signals = candidate.get("redrob_signals", {})
    hours = signals.get("avg_response_time_hours")
    if hours is None or hours < 0:
        return 0.5
    hours = float(hours)
    if hours <= 4:
        return 1.0
    if hours <= 24:
        return 0.75
    if hours <= 72:
        return 0.4
    return 0.1


def _active_job_seeking(candidate: dict) -> float:
    """
    applications_submitted_30d → active search intensity.
    0 applications = passive candidate.
    """
    signals = candidate.get("redrob_signals", {})
    apps = signals.get("applications_submitted_30d")
    if apps is None or apps < 0:
        return 0.3  # unknown → below neutral
    apps = int(apps)
    if apps >= 5:
        return 1.0
    if apps >= 2:
        return 0.7
    if apps >= 1:
        return 0.5
    return 0.2


def _market_validation(candidate: dict) -> float:
    """
    saved_by_recruiters_30d → external recruiter interest signal.
    Normalised to [0, 1], capped at 10 saves.
    """
    signals = candidate.get("redrob_signals", {})
    saved = signals.get("saved_by_recruiters_30d")
    if saved is None or saved < 0:
        return 0.0
    return min(1.0, float(saved) / 10.0)


def _behavioral_multiplier(candidate: dict, role_model: dict) -> float:
    """
    Composite reachability/availability score from redrob_signals.
    Incorporates all six behavioral signals from redrob_signals_doc.md.
    Returns 0.0–1.0.
    """
    signals = candidate.get("redrob_signals", {})

    open_score       = 1.0 if signals.get("open_to_work_flag") else 0.0
    recency          = _recency_score(candidate)
    response_rate    = float(signals.get("recruiter_response_rate") or 0.0)
    response_speed   = _response_speed_score(candidate)
    interview_rel    = float(signals.get("interview_completion_rate") or 0.0)
    active_seeking   = _active_job_seeking(candidate)
    mkt_validation   = _market_validation(candidate)

    total = (
        0.25 * open_score
        + 0.20 * recency
        + 0.20 * response_rate
        + 0.15 * response_speed
        + 0.10 * interview_rel
        + 0.05 * active_seeking
        + 0.05 * mkt_validation
    )
    return round(total, 4)


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


def _impossibility_flag(candidate: dict) -> float:
    """
    Hard honeypot gate.  Returns -1.0 when any logically impossible claim is
    detected; 0.0 otherwise.  Two checks:

    1. Expert/advanced skill with duration_months == 0
       A candidate who claims expert proficiency but zero months of practice
       is statistically impossible.  A single such skill triggers the flag.

    2. Role duration exceeds company age
       If duration_months > months elapsed since the role's start_date (i.e.
       the candidate claims to have worked there longer than the company could
       have existed by the time they joined), the record is fabricated.
    """
    # Check 1 — expert/advanced skill with zero duration
    for s in candidate.get("skills", []):
        if not isinstance(s, dict):
            continue
        if s.get("proficiency") in ("expert", "advanced"):
            duration = s.get("duration_months")
            if duration is not None and duration == 0:
                return -1.0

    # Check 2 — role tenure exceeds company age implied by start_date
    today = _today()
    for role in candidate.get("career_history", []):
        duration = role.get("duration_months")
        start_raw = role.get("start_date")
        if not duration or not start_raw:
            continue
        try:
            start = datetime.strptime(start_raw[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        # Max plausible duration for this role: months from start_date to today
        max_months = max(0, (today.year - start.year) * 12 + (today.month - start.month))
        if duration > max_months:
            return -1.0

    return 0.0


def _github_activity(candidate: dict) -> float:
    """
    Normalise github_activity_score (0–100, or -1 if no GitHub linked) → 0–1.
    -1 (no GitHub) maps to 0.0.
    """
    score = candidate.get("redrob_signals", {}).get("github_activity_score", -1)
    if score is None or score < 0:
        return 0.0
    return min(1.0, float(score) / 100.0)


def _open_to_work_score(candidate: dict) -> float:
    """1.0 if open_to_work_flag is True, 0.0 otherwise."""
    return 1.0 if candidate.get("redrob_signals", {}).get("open_to_work_flag") else 0.0


def _response_rate_score(candidate: dict) -> float:
    """recruiter_response_rate is already 0–1; return as-is (missing → 0.0)."""
    val = candidate.get("redrob_signals", {}).get("recruiter_response_rate")
    if val is None:
        return 0.0
    return float(min(1.0, max(0.0, val)))


def _profile_completeness(candidate: dict) -> float:
    """
    Low completeness = less data to evaluate = penalize slightly.
    Direct field from redrob_signals.profile_completeness_score (0-100).
    Normalize to [0,1].
    """
    score = candidate.get('redrob_signals', {}).get(
        'profile_completeness_score', 50.0
    )
    return round(float(score) / 100.0, 4)


def _education_tier_score(candidate: dict) -> float:
    """
    Soft signal from education institution tier.
    tier_1 = IIT/IIM/top NIT = 1.0
    tier_2 = good college = 0.6
    tier_3 = average = 0.3
    tier_4/unknown = 0.1

    Take the best tier across all education entries.
    This is a weak signal — weight it low in LambdaMART.
    """
    TIER_SCORES = {
        'tier_1': 1.0,
        'tier_2': 0.6,
        'tier_3': 0.3,
        'tier_4': 0.1,
        'unknown': 0.2,
    }

    education = candidate.get('education', [])
    if not education:
        return 0.2

    best = max(
        TIER_SCORES.get(e.get('tier', 'unknown'), 0.2)
        for e in education
    )
    return best


def _skill_depth_score(candidate: dict) -> float:
    """
    Measures genuine skill depth from duration_months.
    Skills with advanced/expert proficiency AND high duration
    score higher than skills just listed without time investment.

    Focus only on ranking/retrieval/ML relevant skills.
    """
    RELEVANT_SKILLS = {
        'faiss', 'pinecone', 'qdrant', 'milvus', 'weaviate', 'elasticsearch',
        'opensearch', 'pgvector', 'bm25', 'sentence transformers', 'bge',
        'learning to rank', 'lambdamart', 'xgboost', 'lightgbm',
        'embeddings', 'vector search', 'semantic search', 'information retrieval',
        'nlp', 'pytorch', 'hugging face transformers', 'fine-tuning llms',
        'rag', 'ranking systems', 'recommendation systems'
    }

    skills = candidate.get('skills', [])
    total_depth = 0.0
    relevant_count = 0

    for s in skills:
        name = s.get('name', '').lower()
        proficiency = s.get('proficiency', '')
        duration = s.get('duration_months', 0) or 0

        if name not in RELEVANT_SKILLS:
            continue
        if proficiency not in ('advanced', 'expert'):
            continue

        duration_score = min(1.0, duration / 36.0)
        prof_weight = 1.0 if proficiency == 'expert' else 0.7

        total_depth += duration_score * prof_weight
        relevant_count += 1

    if relevant_count == 0:
        return 0.0

    return min(1.0, total_depth / max(3, relevant_count))


# ── public API ────────────────────────────────────────────────────────────────

def _cross_encoder_score(candidate: dict, ce_scores: dict) -> float:
    import math
    cid = candidate.get('candidate_id', '')
    raw = ce_scores.get(cid, 0.0)
    return round(1.0 / (1.0 + math.exp(-raw)), 6)


def build_feature_vector(
    candidate: dict,
    role_model: dict,
    cosine_sim: float,
    ce_scores: dict | None = None,
) -> dict[str, float]:
    """
    Build a flat feature dict for a candidate.

    Parameters
    ----------
    candidate   : raw candidate dict from candidates.jsonl
    role_model  : parsed role_model.yaml dict
    cosine_sim  : pre-computed cosine similarity from retriever.py (0–1)
    ce_scores   : optional {candidate_id: raw_logit} from cross-encoder

    Returns
    -------
    dict with all feature keys as floats.
    title_disqualified is -1.0 or 0.0 (used as a hard gate by scorer.py).
    All other features are in 0.0–1.0 unless noted.
    """
    _ce = ce_scores if ce_scores is not None else {}
    return {
        "cosine_similarity":    round(float(cosine_sim), 6),
        "experience_fit_score": _experience_fit_score(candidate, role_model),
        "is_ml_engineer":       _is_ml_engineer(candidate),
        "title_disqualified":   _title_disqualified(candidate, role_model),
        "impossibility_flag":   _impossibility_flag(candidate),
        "production_ml_score":  _production_ml_score(candidate),
        "domain_alignment":     _domain_alignment(candidate),
        "consulting_penalty":   _consulting_penalty(candidate),
        "behavioral_multiplier":_behavioral_multiplier(candidate, role_model),
        "consistency_score":    _consistency_score(candidate),
        "location_score":       _location_score(candidate, role_model),
        "notice_penalty":       _notice_penalty(candidate, role_model),
        "github_activity":      _github_activity(candidate),
        "ce_score":             _cross_encoder_score(candidate, _ce),
        "open_to_work_score":   _open_to_work_score(candidate),
        "response_rate_score":  _response_rate_score(candidate),
        "recency_score":        _recency_score(candidate),
        "response_speed_score": _response_speed_score(candidate),
        "interview_reliability":float(candidate.get("redrob_signals", {}).get("interview_completion_rate") or 0.0),
        "active_job_seeking":   _active_job_seeking(candidate),
        "market_validation":    _market_validation(candidate),
        "skill_depth_score":    _skill_depth_score(candidate),
        "education_tier_score": _education_tier_score(candidate),
        "profile_completeness": _profile_completeness(candidate),
    }
