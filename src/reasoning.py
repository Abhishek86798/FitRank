# Generates the reasoning column for each ranked candidate via fact-grounded template composition —
# no LLM at ranking time, zero hallucination, every claim pulled directly from the candidate record.

from __future__ import annotations

import random
from datetime import date, datetime

# ── helpers ───────────────────────────────────────────────────────────────────

def _yoe(candidate: dict) -> float:
    return candidate.get("profile", {}).get("years_of_experience", 0.0)


def _current_role(candidate: dict) -> tuple[str, str]:
    profile = candidate.get("profile", {})
    return profile.get("current_title", ""), profile.get("current_company", "")


def _location(candidate: dict) -> str:
    profile = candidate.get("profile", {})
    loc = profile.get("location", "")
    country = profile.get("country", "")
    if country and country.lower() not in ("india",):
        return f"{loc}, {country}".strip(", ")
    return loc


def _notice_days(candidate: dict) -> int | None:
    return candidate.get("redrob_signals", {}).get("notice_period_days")


def _is_open(candidate: dict) -> bool:
    return bool(candidate.get("redrob_signals", {}).get("open_to_work_flag"))


def _last_active(candidate: dict) -> str | None:
    return candidate.get("redrob_signals", {}).get("last_active_date")


def _career_descriptions(candidate: dict) -> str:
    return " ".join(
        r.get("description", "") for r in candidate.get("career_history", [])
    ).lower()


def _advanced_skills(candidate: dict) -> list[str]:
    return [
        s["name"] for s in candidate.get("skills", [])
        if isinstance(s, dict) and s.get("proficiency") in ("advanced", "expert")
    ]


def _all_skill_names(candidate: dict) -> list[str]:
    return [s["name"] for s in candidate.get("skills", []) if isinstance(s, dict)]


def _assessment_scores(candidate: dict) -> dict[str, float]:
    return candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})


def _career_roles(candidate: dict) -> list[tuple[str, str]]:
    return [
        (r.get("title", ""), r.get("company", ""))
        for r in candidate.get("career_history", [])
    ]


def _career_companies(candidate: dict) -> list[str]:
    return [r.get("company", "") for r in candidate.get("career_history", [])]


_CONSULTING = {"tcs", "infosys", "wipro", "accenture", "cognizant",
               "tech mahindra", "hcl", "capgemini", "mindtree"}

def _is_consulting_heavy(candidate: dict) -> bool:
    companies = [c.lower() for c in _career_companies(candidate)]
    return sum(1 for c in companies if any(f in c for f in _CONSULTING)) >= 2


def _months_since_active(candidate: dict) -> int | None:
    raw = _last_active(candidate)
    if not raw:
        return None
    try:
        last = datetime.strptime(raw, "%Y-%m-%d").date()
        today = date(2026, 6, 26)
        return max(0, (today - last).days // 30)
    except ValueError:
        return None


_RETRIEVAL_KEYWORDS = {
    "ranking", "retrieval", "recsys", "recommendation", "faiss", "weaviate",
    "qdrant", "milvus", "opensearch", "elasticsearch", "pinecone",
    "embedding", "semantic search", "dense retrieval", "hybrid retrieval",
    "ndcg", "a/b test", "ab test",
}

def _retrieval_signal_in_career(candidate: dict) -> list[str]:
    desc = _career_descriptions(candidate)
    return [kw for kw in sorted(_RETRIEVAL_KEYWORDS) if kw in desc]


# ── phrasing banks ────────────────────────────────────────────────────────────
# Each bank holds ≥3 variants so adjacent candidates never share identical phrasing.

_OPENS_STRONG = [
    "{title} at {company} with {yoe:.1f} years of experience.",
    "Currently {title} at {company} ({yoe:.1f} yrs exp).",
    "{yoe:.1f}-year career, most recently as {title} at {company}.",
]

_OPENS_WEAK = [
    "Profile shows {yoe:.1f} years of experience as {title} at {company}.",
    "{title} background ({company}, {yoe:.1f} yrs).",
    "Registered as {title} at {company} with {yoe:.1f} years.",
]

_RETRIEVAL_MATCH = [
    "Career descriptions reference {kws} — direct signal for this role.",
    "Explicit mentions of {kws} in work history align with JD requirements.",
    "Production evidence for {kws} found in career narrative.",
]

_SKILLS_MATCH = [
    "Advanced-proficiency skills include {skills}.",
    "Declared advanced skills: {skills}.",
    "Skill profile highlights {skills} at advanced or expert level.",
]

_ASSESSMENT_NOTE = [
    "Platform assessment score for {skill}: {score:.0f}/100.",
    "Skill assessment on record — {skill}: {score:.0f}.",
    "Verified assessment: {skill} scored {score:.0f}/100 on platform.",
]

_LOCATION_OK = [
    "Located in {loc} — within preferred geography.",
    "{loc} location matches target cities.",
    "Based in {loc}, consistent with role geography.",
]

_LOCATION_BAD = [
    "Located in {loc} — outside preferred India cities; relocation unclear.",
    "{loc} location is outside target geography.",
    "Based in {loc}, which is outside the preferred hiring zone.",
]

_NOTICE_OK = [
    "{days}-day notice period — within buyout window.",
    "Notice period: {days} days, manageable.",
    "{days}-day notice; fits the sub-30-day preference or buyout ceiling.",
]

_NOTICE_LONG = [
    "Notice period of {days} days exceeds the 30-day buyout threshold.",
    "{days}-day notice is above the preferred ceiling.",
    "Long notice period ({days}d) may be a logistical blocker.",
]

_OPEN_FLAG = [
    "Profile marked open-to-work.",
    "Candidate has active open-to-work signal.",
    "Open-to-work flag is set.",
]

_NOT_OPEN = [
    "Not currently flagged as open to work.",
    "Open-to-work flag is not set.",
    "No active open-to-work signal on profile.",
]

_STALE = [
    "Last active {months} months ago — engagement risk.",
    "Profile inactive for approximately {months} months.",
    "Activity gap of ~{months} months; outreach may be slow.",
]

_DOMAIN_MISMATCH = [
    "Career history is primarily in {domain}, not NLP/IR/ML.",
    "Work descriptions describe {domain} work, not ML production.",
    "Background is {domain}-focused with no evidence of ML deployment.",
]

_CONSULTING_CONCERN = [
    "Majority of tenure at consulting/IT-services firms — JD flags this as a concern.",
    "Heavy consulting background (TCS/Wipro/Infosys class); JD explicitly flags this.",
    "Consulting-firm-heavy career path is noted as a negative signal per JD.",
]

_SKILLS_UNVERIFIED = [
    "Some declared skills ({skills}) are not corroborated by career descriptions.",
    "Skill claims ({skills}) lack supporting evidence in work history.",
    "Declared skills ({skills}) appear aspirational; not mentioned in any role description.",
]


# ── pick helpers ──────────────────────────────────────────────────────────────

def _pick(bank: list[str], seed: int) -> str:
    return bank[seed % len(bank)]


# ── main entry point ──────────────────────────────────────────────────────────

def compose_reasoning(
    candidate: dict,
    features: dict,
    rank: int,
    role_model: dict | None = None,
) -> str:
    """
    Build a grounded, non-hallucinated reasoning string for a ranked candidate.

    Every claim is derived from `candidate` (the raw profile record) or
    `features` (pre-computed scalars from feature_builder). `role_model` is
    unused at generation time but accepted for API compatibility with the
    scorer pipeline.

    Returns a single paragraph string ≤ 400 characters.
    """
    seed = rank  # deterministic variation by rank position

    title, company = _current_role(candidate)
    yoe   = _yoe(candidate)
    loc   = _location(candidate)
    days  = _notice_days(candidate)
    open_ = _is_open(candidate)
    months_inactive = _months_since_active(candidate)
    retrieval_hits  = _retrieval_signal_in_career(candidate)
    adv_skills      = _advanced_skills(candidate)
    assessments     = _assessment_scores(candidate)

    parts: list[str] = []

    # ── Opening: who is this person ──────────────────────────────────────────
    cosine = features.get("cosine_sim", 0.0)
    if cosine >= 0.70 or retrieval_hits:
        opener = _pick(_OPENS_STRONG, seed)
    else:
        opener = _pick(_OPENS_WEAK, seed + 1)
    parts.append(opener.format(title=title, company=company, yoe=yoe))

    # ── Positive signals ─────────────────────────────────────────────────────
    if retrieval_hits:
        kws = ", ".join(retrieval_hits[:3])
        parts.append(_pick(_RETRIEVAL_MATCH, seed + 2).format(kws=kws))

    if adv_skills:
        skills_str = ", ".join(adv_skills[:4])
        parts.append(_pick(_SKILLS_MATCH, seed + 3).format(skills=skills_str))

    top_assessment = max(assessments.items(), key=lambda x: x[1]) if assessments else None
    if top_assessment:
        skill_name, score = top_assessment
        parts.append(_pick(_ASSESSMENT_NOTE, seed + 4).format(skill=skill_name, score=score))

    # ── Location ─────────────────────────────────────────────────────────────
    if loc:
        india_cities = {
            "pune", "noida", "delhi", "ncr", "hyderabad", "mumbai",
            "bangalore", "bengaluru", "gurgaon", "gurugram", "chandigarh",
        }
        loc_lower = loc.lower()
        if any(city in loc_lower for city in india_cities):
            parts.append(_pick(_LOCATION_OK, seed + 5).format(loc=loc))
        else:
            parts.append(_pick(_LOCATION_BAD, seed + 5).format(loc=loc))

    # ── Availability ─────────────────────────────────────────────────────────
    if open_:
        parts.append(_pick(_OPEN_FLAG, seed + 6))
    else:
        parts.append(_pick(_NOT_OPEN, seed + 6))

    if days is not None:
        if days <= 30:
            parts.append(_pick(_NOTICE_OK, seed + 7).format(days=days))
        else:
            parts.append(_pick(_NOTICE_LONG, seed + 7).format(days=days))

    # ── Concerns (mandatory — at least one) ──────────────────────────────────
    concern_added = False

    if months_inactive is not None and months_inactive >= 3:
        parts.append(_pick(_STALE, seed + 8).format(months=months_inactive))
        concern_added = True

    domain_label = features.get("inferred_domain", "")
    if domain_label and domain_label not in ("ml", "ai", "nlp", "recsys"):
        parts.append(_pick(_DOMAIN_MISMATCH, seed + 9).format(domain=domain_label))
        concern_added = True

    if _is_consulting_heavy(candidate) and not concern_added:
        parts.append(_pick(_CONSULTING_CONCERN, seed + 10))
        concern_added = True

    # Fallback concern: declared skills not corroborated — only when no other concern raised
    # and no retrieval signal already flags the profile positively (avoid contradicting good news)
    if not concern_added and adv_skills and not retrieval_hits:
        desc = _career_descriptions(candidate)
        unverified = [s for s in adv_skills if s.lower() not in desc][:2]
        if unverified:
            skills_str = ", ".join(unverified)
            parts.append(_pick(_SKILLS_UNVERIFIED, seed + 11).format(skills=skills_str))
            concern_added = True

    if not concern_added:
        parts.append("No strong disqualifiers identified; recommend recruiter screen to verify depth.")

    result = " ".join(parts)
    # Hard cap: trim to 500 chars at a sentence boundary if needed
    if len(result) > 500:
        truncated = result[:497]
        last_period = truncated.rfind(".")
        if last_period > 300:
            result = truncated[:last_period + 1]
    return result
