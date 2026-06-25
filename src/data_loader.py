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
      2. career history descriptions (hard to fake, most signal-dense)
      3. summary
      4. skills (easy to inflate — appended last, naturally down-weighted by position)
         Only advanced/expert skills included to reduce noise.

    Capped at 2048 characters.
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

    for role in candidate.get("career_history", []):
        title = role.get("title", "").strip()
        company = role.get("company", "").strip()
        duration = role.get("duration_months")
        description = role.get("description", "").strip()

        role_header = " | ".join(filter(None, [title, company]))
        if duration:
            role_header += f" ({duration // 12}y {duration % 12}m)"
        if role_header:
            parts.append(role_header)
        if description:
            parts.append(description)

    summary = profile.get("summary", "").strip()
    if summary:
        parts.append(summary)

    # skills is list of {name, proficiency, endorsements, duration_months}
    # only include advanced/expert to avoid noise from keyword-stuffing
    skills = candidate.get("skills", [])
    strong_skills = [
        s["name"] for s in skills
        if isinstance(s, dict) and s.get("proficiency") in ("advanced", "expert")
    ]
    all_skills = [s["name"] for s in skills if isinstance(s, dict)]
    skill_list = strong_skills if strong_skills else all_skills
    if skill_list:
        parts.append("Skills: " + ", ".join(skill_list))

    text = " ".join(parts)
    return text[:2048]
