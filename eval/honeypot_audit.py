"""
Print full profiles of the top-N ranked candidates from a submission CSV
for manual honeypot inspection.

Usage
-----
    python eval/honeypot_audit.py <submission.csv> [--candidates data/candidates.jsonl]
                                                   [--top 10]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _load_submission_top(path: Path, top: int) -> list[tuple[int, str, float]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["rank"]), row["candidate_id"].strip(), float(row["score"])))
    rows.sort(key=lambda x: x[0])
    return rows[:top]


def _load_candidates(path: Path) -> dict[str, dict]:
    candidates: dict[str, dict] = {}
    suffix = path.suffix.lower()
    if suffix == ".json":
        for c in json.loads(path.read_bytes()):
            candidates[c["candidate_id"]] = c
    elif suffix == ".jsonl":
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    import orjson
                    c = orjson.loads(line)
                    candidates[c["candidate_id"]] = c
    return candidates


def _print_profile(rank: int, score: float, cand: dict | None, cid: str) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  RANK #{rank}  |  {cid}  |  score={score:.4f}")
    print(sep)

    if cand is None:
        print("  [profile not found in candidates file]")
        return

    profile = cand.get("profile", {})
    print(f"  Headline    : {profile.get('headline', 'N/A')}")
    print(f"  Title       : {profile.get('current_title', 'N/A')} @ {profile.get('current_company', 'N/A')}")
    print(f"  YoE         : {profile.get('years_of_experience', 'N/A')}")
    print(f"  Location    : {profile.get('location', 'N/A')}")
    print(f"  Notice days : {profile.get('notice_period_days', 'N/A')}")
    print(f"  Open to work: {profile.get('open_to_work', 'N/A')}")
    print(f"  Last active : {profile.get('last_active', 'N/A')}")

    summary = profile.get("summary", "").strip()
    if summary:
        print(f"\n  Summary:\n    {summary[:300]}")

    history = cand.get("career_history", [])
    if history:
        print("\n  Career history:")
        for role in history[:4]:
            title   = role.get("title", "?")
            company = role.get("company", "?")
            dur     = role.get("duration_months")
            dur_str = f"{dur // 12}y {dur % 12}m" if dur else "?"
            desc    = role.get("description", "").strip()[:120]
            print(f"    • {title} @ {company} ({dur_str})")
            if desc:
                print(f"      {desc}")

    skills = cand.get("skills", [])
    if skills:
        skill_names = [s.get("name", "") for s in skills[:12] if isinstance(s, dict)]
        print(f"\n  Skills: {', '.join(skill_names)}")

    signals = cand.get("redrob_signals", {})
    if signals:
        print(f"\n  Redrob signals: {signals}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Honeypot audit — inspect top-N ranked candidates.")
    parser.add_argument("submission", help="Submission CSV path")
    parser.add_argument(
        "--candidates",
        default="data/candidates.jsonl",
        help="Candidate data file (.json or .jsonl). Falls back to data/sample_candidates.json.",
    )
    parser.add_argument("--top", type=int, default=10, help="Number of top candidates to show")
    args = parser.parse_args()

    cand_path = Path(args.candidates)
    if not cand_path.exists():
        cand_path = Path("data/sample_candidates.json")
        if not cand_path.exists():
            print(f"[warn] No candidates file found; profiles will show as missing.")
            cand_path = None

    candidates = _load_candidates(cand_path) if cand_path else {}
    top_rows = _load_submission_top(Path(args.submission), args.top)

    print(f"\nHoneypot audit — top {args.top} from {args.submission}")
    for rank, cid, score in top_rows:
        _print_profile(rank, score, candidates.get(cid), cid)
    print(f"\n{'=' * 72}\n")


if __name__ == "__main__":
    main()
