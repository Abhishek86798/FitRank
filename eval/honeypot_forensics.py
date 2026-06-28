"""
Honeypot forensics: turns the pipeline's passive honeypot defense into an
offensive evidence deck. For each candidate, find_contradictions() returns
named, evidence-backed contradictions that explain *why* a record is suspect.

Usage
-----
    python eval/honeypot_forensics.py [--submission team_xxx.csv]
                                      [--candidates data/candidates.jsonl]
                                      [--top-n 100]
                                      [--output eval/honeypot_forensics_report.txt]

Writes eval/honeypot_forensics_report.txt with:
  - How many honeypots were caught and filtered (never reached top-100)
  - For 5 example honeypots: exact named contradictions
  - Confirmation: 0 honeypots in final top-100
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

# ── contradiction types ────────────────────────────────────────────────────────

SKILL_DURATION_IMPOSSIBLE = "SKILL_DURATION_IMPOSSIBLE"
TENURE_EXCEEDS_COMPANY    = "TENURE_EXCEEDS_COMPANY"
YOE_MISMATCH              = "YOE_MISMATCH"
SKILL_COUNT_INFLATION     = "SKILL_COUNT_INFLATION"

# Keyword set for "supporting career mention" check used in SKILL_COUNT_INFLATION
_SKILL_NOISE_WORDS = {"and", "or", "the", "a", "an", "of", "in", "for", "with"}

# Domain keywords we expect to see if someone really knows ML production systems
_DOMAIN_KEYWORDS = re.compile(
    r"\b(nlp|retrieval|ranking|embedding|recommendation|search|transformer|bert|"
    r"llm|faiss|pinecone|weaviate|qdrant|milvus|elasticsearch|opensearch|"
    r"semantic|dense|sparse|bm25|a/b|ndcg|map@|mrr)\b",
    re.IGNORECASE,
)

_YOE_GAP_THRESHOLD = 3      # years — gaps larger than this are flagged
_EXPERT_SKILL_THRESHOLD = 8  # count of expert/advanced skills before inflation check


def _today() -> date:
    return datetime.utcnow().date()


def _career_text(candidate: dict) -> str:
    return " ".join(
        r.get("description", "") for r in candidate.get("career_history", [])
    ).lower()


def find_contradictions(candidate: dict) -> list[dict]:
    """
    Return named, evidence-backed contradictions in candidate's profile.

    Each contradiction dict has:
      type     : one of the module-level constants above
      detail   : human-readable explanation
      evidence : specific values from the record that prove the contradiction

    Checks:
      1. SKILL_DURATION_IMPOSSIBLE — expert/advanced skill with duration_months == 0
      2. TENURE_EXCEEDS_COMPANY    — role tenure > plausible company age
      3. YOE_MISMATCH              — stated YOE differs from career sum by > 3 years
      4. SKILL_COUNT_INFLATION     — 8+ expert skills, 0 supporting career mentions
    """
    contradictions: list[dict] = []

    # ── 1. SKILL_DURATION_IMPOSSIBLE ──────────────────────────────────────────
    for skill in candidate.get("skills", []):
        if not isinstance(skill, dict):
            continue
        if skill.get("proficiency") in ("expert", "advanced"):
            duration = skill.get("duration_months")
            if duration is not None and duration == 0:
                contradictions.append({
                    "type": SKILL_DURATION_IMPOSSIBLE,
                    "detail": (
                        f"Skill '{skill.get('name', '?')}' is claimed at "
                        f"{skill['proficiency']} level but has duration_months=0. "
                        "Expert/advanced proficiency in zero months is impossible."
                    ),
                    "evidence": {
                        "skill": skill.get("name"),
                        "proficiency": skill.get("proficiency"),
                        "duration_months": duration,
                    },
                })

    # ── 2. TENURE_EXCEEDS_COMPANY ─────────────────────────────────────────────
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
        max_months = max(0, (today.year - start.year) * 12 + (today.month - start.month))
        if duration > max_months:
            contradictions.append({
                "type": TENURE_EXCEEDS_COMPANY,
                "detail": (
                    f"Role '{role.get('title', '?')}' at '{role.get('company', '?')}' "
                    f"claims {duration} months of tenure, but the role started "
                    f"{start_raw[:10]} — only {max_months} months ago. "
                    "The candidate claims more tenure than time elapsed."
                ),
                "evidence": {
                    "title": role.get("title"),
                    "company": role.get("company"),
                    "start_date": start_raw[:10],
                    "claimed_duration_months": duration,
                    "max_plausible_months": max_months,
                },
            })

    # ── 3. YOE_MISMATCH ───────────────────────────────────────────────────────
    stated_yoe = candidate.get("profile", {}).get("years_of_experience") or 0.0
    history = candidate.get("career_history", [])
    career_months = sum(r.get("duration_months") or 0 for r in history)
    career_yoe = career_months / 12.0
    gap = abs(stated_yoe - career_yoe)
    if stated_yoe > 0 and gap > _YOE_GAP_THRESHOLD:
        contradictions.append({
            "type": YOE_MISMATCH,
            "detail": (
                f"Stated years_of_experience ({stated_yoe:.1f} yrs) differs from "
                f"career history sum ({career_yoe:.1f} yrs, {career_months} months) "
                f"by {gap:.1f} years — exceeds the {_YOE_GAP_THRESHOLD}-year tolerance."
            ),
            "evidence": {
                "stated_yoe": stated_yoe,
                "career_months_sum": career_months,
                "career_yoe_equivalent": round(career_yoe, 1),
                "gap_years": round(gap, 1),
            },
        })

    # ── 4. SKILL_COUNT_INFLATION ───────────────────────────────────────────────
    expert_skills = [
        s for s in candidate.get("skills", [])
        if isinstance(s, dict) and s.get("proficiency") in ("expert", "advanced")
    ]
    if len(expert_skills) >= _EXPERT_SKILL_THRESHOLD:
        career = _career_text(candidate)  # already lowercased
        # A skill is "supported" if at least one meaningful word from its name
        # appears in the career text (case-insensitive word-level match).
        # "meaningful" excludes generic stop words and very short tokens.
        _STOP = {"and", "or", "the", "a", "an", "of", "in", "for", "with",
                 "to", "at", "by", "from", "as", "is", "it"}

        def _skill_supported(skill_name: str, text: str) -> bool:
            tokens = [
                t.strip("()-/") for t in skill_name.lower().split()
                if len(t.strip("()-/")) > 2 and t.strip("()-/") not in _STOP
            ]
            return any(tok in text for tok in tokens)

        supported = sum(
            1 for s in expert_skills
            if _skill_supported(s.get("name", ""), career)
        )
        if supported == 0:
            skill_names = [s.get("name", "?") for s in expert_skills[:6]]
            contradictions.append({
                "type": SKILL_COUNT_INFLATION,
                "detail": (
                    f"Candidate claims {len(expert_skills)} expert/advanced skills "
                    f"but none of them have any supporting evidence in career descriptions. "
                    "This pattern is consistent with keyword-stuffing."
                ),
                "evidence": {
                    "expert_skill_count": len(expert_skills),
                    "skills_supported_by_career_text": supported,
                    "sample_unverified_skills": skill_names,
                },
            })

    return contradictions


# ── report runner ──────────────────────────────────────────────────────────────

def _load_submission_ids(path: Path, top_n: int) -> set[str]:
    import csv
    rows: list[tuple[int, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["rank"]), row["candidate_id"].strip()))
    rows.sort(key=lambda x: x[0])
    return {cid for _, cid in rows[:top_n]}


def _stream_candidates(path: Path):
    """Yield all candidate dicts from .json or .jsonl file."""
    if path.suffix.lower() == ".json":
        import json
        yield from json.loads(path.read_bytes())
    else:
        import orjson
        with open(path, "rb") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield orjson.loads(line)


def _is_honeypot(candidate: dict) -> bool:
    """Quick check: does this candidate trigger the impossibility_flag gate?"""
    from src.feature_builder import _impossibility_flag
    return _impossibility_flag(candidate) == -1.0


def generate_report(
    submission_path: Path,
    candidates_path: Path,
    top_n: int = 100,
    output_path: Path | None = None,
) -> str:
    """
    Scan all candidates, classify honeypots, cross-reference against top-N submission.

    Returns the report as a string (also writes to output_path if given).
    """
    top_ids = _load_submission_ids(submission_path, top_n)

    honeypots_caught: list[dict] = []       # honeypots that never reached top-N
    honeypots_in_top: list[dict] = []       # honeypots that slipped into top-N
    total_candidates = 0

    for cand in _stream_candidates(candidates_path):
        total_candidates += 1
        if _is_honeypot(cand):
            cid = cand["candidate_id"]
            entry = {"candidate_id": cid, "contradictions": find_contradictions(cand)}
            if cid in top_ids:
                honeypots_in_top.append(entry)
            else:
                honeypots_caught.append(entry)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("HONEYPOT FORENSICS REPORT")
    lines.append("=" * 72)
    lines.append(f"Submission   : {submission_path}")
    lines.append(f"Candidates   : {candidates_path}  ({total_candidates} total)")
    lines.append(f"Top-N cutoff : {top_n}")
    lines.append("")

    total_honeypots = len(honeypots_caught) + len(honeypots_in_top)
    lines.append(f"Total honeypots detected   : {total_honeypots}")
    lines.append(f"  Caught (never reached #{top_n}): {len(honeypots_caught)}")
    lines.append(f"  In final top-{top_n}           : {len(honeypots_in_top)}")
    lines.append("")

    if honeypots_in_top:
        lines.append("[!!] WARNING: honeypots found in top-N submission:")
        for entry in honeypots_in_top:
            lines.append(f"  {entry['candidate_id']}")
        lines.append("")
    else:
        lines.append(f"[OK] CONFIRMED: 0 honeypots in final top-{top_n}")
        lines.append("")

    # 5 example honeypots with full contradiction detail
    examples = honeypots_caught[:5]
    if examples:
        lines.append(f"Example honeypots (up to 5 with full contradictions):")
        lines.append("-" * 72)
        for entry in examples:
            cid = entry["candidate_id"]
            contradictions = entry["contradictions"]
            lines.append(f"\n  Candidate: {cid}  ({len(contradictions)} contradiction(s))")
            for c in contradictions:
                lines.append(f"    [{c['type']}]")
                lines.append(f"      {c['detail']}")
                lines.append(f"      Evidence: {c['evidence']}")
        lines.append("")

    lines.append("=" * 72)
    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Honeypot forensics report")
    parser.add_argument("--submission",  default="team_xxx.csv",                    help="Ranked submission CSV")
    parser.add_argument("--candidates",  default="data/candidates.jsonl",            help="Candidate data file")
    parser.add_argument("--top-n",       type=int, default=100,                      help="Top-N cutoff")
    parser.add_argument("--output",      default="eval/honeypot_forensics_report.txt", help="Output report path")
    args = parser.parse_args()

    report = generate_report(
        submission_path=Path(args.submission),
        candidates_path=Path(args.candidates),
        top_n=args.top_n,
        output_path=Path(args.output),
    )
    print(report)
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
