"""
Convert role_model.yaml into a recruiter-readable summary.
No LLM calls — purely derives human-readable text from structured YAML fields.
"""

from __future__ import annotations


def summarize_role_model(role_model: dict) -> dict:
    """
    Convert role_model.yaml into a recruiter-readable summary dict.

    Returns:
    {
      "role_title": str,
      "experience_band": str,
      "must_haves": list[str],
      "hard_disqualifiers": list[str],
      "ideal_signals": list[str],
      "behavioral_expectations": list[str],
      "location_preferences": list[str],
      "key_traps": list[str],
    }
    """
    # ── experience band ───────────────────────────────────────────────────────
    eb = role_model.get("experience_band", {})
    lo, hi = eb.get("min", 5), eb.get("max", 9)
    experience_band = f"{lo}–{hi} years total; sweet spot 6–8 yrs with 4–5 in applied ML"

    # ── must-haves — synthesise from must_have_skills + must_have_domains ─────
    skills  = role_model.get("must_have_skills", [])
    domains = role_model.get("must_have_domains", [])

    # Build a short, sentence-form list rather than dumping raw keywords
    must_haves: list[str] = [
        "Production ML at product companies — shipped systems with real users",
        "End-to-end ownership of ranking, search, or recommendation systems",
        "Embeddings-based retrieval and vector search experience",
        "Evaluation framework built (NDCG/MAP/offline+online eval infra)",
    ]
    # Add any domains not already implied above
    covered = {"retrieval", "ranking", "embeddings", "vector search", "evaluation framework"}
    extra_domains = [d for d in domains if d not in covered and d not in {"search"}]
    if extra_domains:
        must_haves.append(f"Domain depth in: {', '.join(extra_domains)}")

    # ── hard disqualifiers ────────────────────────────────────────────────────
    dis_titles = role_model.get("disqualifying_titles", [])
    dis_cos    = role_model.get("disqualifying_company_types", [])
    dis_pats   = role_model.get("disqualifying_patterns", {})

    # Format consulting firm list compactly
    cos_str = "/".join(t.title() for t in dis_cos) if dis_cos else "pure consulting firms"

    hard_disqualifiers: list[str] = [
        f"Consulting-only career ({cos_str}) — product ownership never demonstrated",
        "Research-only background — published papers but nothing shipped to production",
        '"AI experience" = LangChain wrapper calling OpenAI API built in < 12 months',
        "Non-engineer career path (PM, QA, Marketing, Sales, HR, Ops) with no ML code",
        "No code written in the last 18 months — moved to pure architecture / tech-lead",
    ]
    if dis_titles:
        # Surface a few representative disqualifying titles
        sample = [t.title() for t in dis_titles[:4]]
        hard_disqualifiers.append(
            f"Current title is a non-engineering role: {', '.join(sample)}, etc."
        )

    # ── ideal signals — translate profile signal keys to plain English ─────────
    _signal_map: dict[str, str] = {
        "production_retrieval_system":       "Deployed embedding-based retrieval to real users",
        "production_ranking_system":         "LTR / reranking running in production",
        "production_recommendation_system":  "RecSys shipped end-to-end at product scale",
        "shipped_search_system":             "Search system with measurable recall/precision metrics",
        "evaluation_framework_built":        "Built offline eval infra (NDCG, MAP) + A/B testing",
        "ab_testing_experience":             "A/B testing experience — knows how to validate ranking quality",
        "vector_db_experience":              "Hands-on with vector DBs (Pinecone/Qdrant/Milvus/FAISS)",
    }
    raw_signals = role_model.get("ideal_profile_signals", [])
    ideal_signals = [_signal_map.get(s, s.replace("_", " ").title()) for s in raw_signals]

    # ── behavioral expectations ───────────────────────────────────────────────
    bw = role_model.get("behavioral_weights", {})
    behavioral_expectations: list[str] = []

    weight_map = {
        "open_to_work_flag":         ("Actively open to work on platform", bw.get("open_to_work_flag", 0)),
        "recruiter_response_rate":   ("High recruiter response rate — actually reachable", bw.get("recruiter_response_rate", 0)),
        "last_active_recency_days":  ("Logged in within last 90 days — not a ghost profile", bw.get("last_active_recency_days", 0)),
        "interview_completion_rate": ("Shows up when scheduled — low ghosting risk", bw.get("interview_completion_rate", 0)),
        "profile_completeness_score":("Complete profile — signals platform engagement", bw.get("profile_completeness_score", 0)),
    }
    for key, (label, weight) in sorted(weight_map.items(), key=lambda x: -x[1][1]):
        pct = int(round(weight * 100))
        behavioral_expectations.append(f"{label} ({pct}% weight)")

    # ── location preferences ──────────────────────────────────────────────────
    lp = role_model.get("location_preferences", {})
    cities   = lp.get("preferred_cities", [])
    country  = lp.get("preferred_country", "India")
    notice   = role_model.get("notice_period_preference_days", 30)
    max_notice = role_model.get("notice_period_max_days", 60)

    location_preferences: list[str] = []
    if cities:
        city_str = ", ".join(c.title() for c in cities)
        location_preferences.append(f"Preferred cities ({country}): {city_str}")
    location_preferences.append(f"Notice period: ≤ {notice}d ideal; ≤ {max_notice}d in scope; >90d penalised heavily")
    if lp.get("willing_to_relocate_bonus"):
        location_preferences.append("Relocation bonus applied for candidates willing to move to preferred cities")
    location_preferences.append("Outside India: case-by-case; no work-visa sponsorship")

    # ── key traps ─────────────────────────────────────────────────────────────
    key_traps: list[str] = [
        "Profiles with 'AI/ML' in skills section but career is non-engineering (PM, QA, Sales)",
        "Consulting careers (TCS/Infosys/Wipro) with 'ML projects' that are PoC / billable work, not shipped products",
        "Honeypot profiles with impossible tenure (e.g., 48 months at a company founded 12 months ago)",
        '"GenAI experience" = prompting or wrapping an LLM API for < 1 year, no retrieval depth',
        "Research scientists with strong papers but zero production deployments",
        "Tech leads who last wrote production code 2+ years ago — titled Senior but no longer hands-on",
    ]

    return {
        "role_title":               "Senior AI Engineer — Founding Team (Redrob)",
        "experience_band":          experience_band,
        "must_haves":               must_haves,
        "hard_disqualifiers":       hard_disqualifiers,
        "ideal_signals":            ideal_signals,
        "behavioral_expectations":  behavioral_expectations,
        "location_preferences":     location_preferences,
        "key_traps":                key_traps,
    }
