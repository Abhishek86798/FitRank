# Provides stream_candidates() (memory-safe line-by-line JSONL generator) and build_candidate_text() (builds embeddable text per candidate, prioritising career history over skills list).

from __future__ import annotations

from pathlib import Path
from typing import Generator

import orjson


def stream_candidates(
    path: str | Path, batch_size: int = 100
) -> Generator[list[dict], None, None]:
    """Yield batches of candidate dicts from a .jsonl file without loading it all into memory."""
    batch: list[dict] = []
    with open(path, "rb") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            batch.append(orjson.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def build_candidate_text(candidate: dict) -> str:
    """
    Build a single embeddable text string from a candidate record.

    Priority order (highest signal first):
      1. headline + current title
      2. career history descriptions — appended TWICE to up-weight them.
         Day 2 observation: 4-point cosine compression across 50 candidates means
         generic summaries ≈ genuine ML career text. Doubling career descriptions
         shifts the embedding toward what the person actually did, not how they
         headline themselves. Skills appear once at the end, naturally down-weighted.
      3. summary
      4. skills (advanced/expert only to reduce keyword-stuffing noise)

    Capped at 3000 characters.
    """
    parts: list[str] = []

    profile = candidate.get("profile", {})

    headline = profile.get("headline", "").strip()
    if headline:
        parts.append(headline)

    current_title = profile.get("current_title", "").strip()
    current_company = profile.get("current_company", "").strip()
    yoe = profile.get("years_of_experience")
    if current_title or current_company:
        title_line = " at ".join(filter(None, [current_title, current_company]))
        if yoe is not None:
            title_line += f" ({yoe} yrs exp)"
        parts.append(title_line)

    for role in candidate.get("career_history", [])[:5]:
        title = role.get("title", "").strip()
        company = role.get("company", "").strip()
        description = role.get("description", "").strip()

        role_header = " | ".join(filter(None, [title, company]))
        if role_header:
            parts.append(role_header)
        if description:
            entry = f"{title} at {company}: {description}" if (title or company) else description
            parts.append(entry)
            parts.append(entry)  # doubled intentionally — career descriptions carry the real signal

    summary = profile.get("summary", "").strip()
    if summary:
        parts.append(summary)

    # skills: advanced/expert only to avoid noise from aspirational keyword lists
    skills = candidate.get("skills", [])
    strong_skills = [
        s["name"] for s in skills
        if isinstance(s, dict) and s.get("proficiency") in ("advanced", "expert")
    ]
    all_skills = [s["name"] for s in skills if isinstance(s, dict)]
    skill_list = strong_skills if strong_skills else all_skills
    if skill_list:
        parts.append("Technical skills: " + ", ".join(skill_list[:15]))

    text = " ".join(parts)
    return text[:3000]
