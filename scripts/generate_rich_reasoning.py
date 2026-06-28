"""
Offline precompute: generate natural recruiter reasoning for top-20 candidates
using Gemini 2.5 Flash API, saving results to eval/rich_reasoning.json.

Usage (PowerShell):
    $env:GEMINI_API_KEY = "<key>"
    python scripts/generate_rich_reasoning.py

Get a free key at: https://aistudio.google.com
Install:  pip install google-genai
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

SUBMISSION_CSV   = ROOT / "team_xxx.csv"
AUDIT_JSON       = ROOT / "eval" / "decision_audit.json"
CANDIDATES_JSONL = ROOT / "data" / "candidates.jsonl"
OUTPUT_JSON      = ROOT / "eval" / "rich_reasoning.json"

TOP_N = 20

SYSTEM_PROMPT = (
    "You are a senior technical recruiter writing hiring recommendations. "
    "Be specific — cite real companies, years, and systems from the profile. "
    "Never invent facts not in the provided data. 3-4 sentences max. "
    "End with one honest concern or caveat."
)


def load_top20_from_csv() -> list[dict]:
    if not SUBMISSION_CSV.exists():
        sys.exit(f"ERROR: {SUBMISSION_CSV} not found.")
    rows = []
    with open(SUBMISSION_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "rank":         int(row["rank"]),
                "candidate_id": row["candidate_id"].strip(),
                "reasoning":    row.get("reasoning", ""),
            })
    rows.sort(key=lambda r: r["rank"])
    return [r for r in rows if r["rank"] <= TOP_N]


def load_audit_index() -> dict[str, dict]:
    if not AUDIT_JSON.exists():
        sys.exit(f"ERROR: {AUDIT_JSON} not found. Run eval/generate_audit.py first.")
    audits = json.loads(AUDIT_JSON.read_bytes())
    return {a["candidate_id"]: a for a in audits}


def load_profiles(needed_ids: set[str]) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    if not CANDIDATES_JSONL.exists():
        print(f"WARNING: {CANDIDATES_JSONL} not found — career history will be empty.", file=sys.stderr)
        return profiles
    try:
        import orjson
        loader = orjson.loads
    except ImportError:
        loader = json.loads

    with open(CANDIDATES_JSONL, "rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = loader(line)
            cid = c["candidate_id"]
            if cid in needed_ids:
                profiles[cid] = c
            if len(profiles) == len(needed_ids):
                break
    return profiles


def build_user_prompt(
    meta: dict,
    career_history: list[dict],
    top_reasons: list[dict],
) -> str:
    title      = meta.get("title", "Unknown Title")
    company    = meta.get("company", "Unknown Company")
    yoe        = meta.get("yoe", "?")
    location   = meta.get("location", "Unknown")
    notice     = meta.get("notice_days", "?")
    open_work  = meta.get("open_to_work", False)

    career_lines = []
    for entry in career_history[:2]:
        t   = entry.get("title", "")
        co  = entry.get("company", "")
        desc = (entry.get("description") or "")[:300]
        career_lines.append(f"- {t} at {co}: {desc}")
    career_block = "\n".join(career_lines) if career_lines else "- No career history available."

    reasons_lines = []
    for r in top_reasons:
        feat      = r.get("feature", "?").replace("_", " ")
        rank_drop = r.get("rank_drop", 0)
        reasons_lines.append(f"  Remove {feat} -> drops {rank_drop} positions")
    reasons_block = "\n".join(reasons_lines) if reasons_lines else "  No counterfactual data."

    return (
        f"Candidate: {title} at {company}, {yoe} years experience.\n"
        f"Location: {location}. Notice: {notice} days. Open to work: {open_work}.\n"
        "\n"
        "Career evidence:\n"
        f"{career_block}\n"
        "\n"
        "Why this rank (counterfactual analysis):\n"
        f"{reasons_block}\n"
        "\n"
        "Write a 3-4 sentence recruiter recommendation for this candidate."
    )


def call_gemini(client, user_prompt: str) -> str:
    from google.genai import types
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.4,
            max_output_tokens=300,
        ),
    )
    return response.text.strip()


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "ERROR: GEMINI_API_KEY environment variable not set.\n"
            "Get a free key at https://aistudio.google.com and set:\n"
            "  PowerShell: $env:GEMINI_API_KEY = '<your-key>'"
        )

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
    except ImportError:
        sys.exit(
            "ERROR: google-genai not installed.\n"
            "Run: pip install google-genai"
        )

    print(f"Loading top-{TOP_N} candidates from {SUBMISSION_CSV.name}...")
    top20 = load_top20_from_csv()
    print(f"  Found {len(top20)} candidates.")

    print(f"Loading audit data from {AUDIT_JSON.name}...")
    audit_index = load_audit_index()

    needed_ids = {r["candidate_id"] for r in top20}
    print(f"Loading candidate profiles from {CANDIDATES_JSONL.name}...")
    profiles = load_profiles(needed_ids)
    print(f"  Loaded profiles for {len(profiles)}/{len(needed_ids)} candidates.")

    # Load existing output if any (for resume on partial run)
    rich_reasoning: dict[str, str] = {}
    if OUTPUT_JSON.exists():
        rich_reasoning = json.loads(OUTPUT_JSON.read_bytes())
        print(f"  Resuming — {len(rich_reasoning)} already generated.")

    print(f"\nGenerating rich reasoning for {len(top20)} candidates via Gemini 2.5 Flash...\n")
    print("=" * 70)

    for row in top20:
        cid  = row["candidate_id"]
        rank = row["rank"]

        if cid in rich_reasoning:
            print(f"[#{rank:02d}] {cid} — SKIPPED (already generated)")
            continue

        audit = audit_index.get(cid, {})
        meta  = audit.get("candidate_meta", {})
        top_reasons = audit.get("top_reasons", [])

        profile      = profiles.get(cid, {})
        raw_career   = profile.get("career_history", [])

        user_prompt = build_user_prompt(meta, raw_career, top_reasons)

        try:
            reasoning = call_gemini(client, user_prompt)

            rich_reasoning[cid] = reasoning

            label = f"{meta.get('title','?')} @ {meta.get('company','?')}"
            print(f"[#{rank:02d}] {cid} — {label}")
            print(f"  {reasoning}")
            print()

            # Save after each candidate so partial results are not lost
            OUTPUT_JSON.write_text(
                json.dumps(rich_reasoning, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Respect free-tier rate limits (~15 RPM)
            time.sleep(4)

        except Exception as exc:
            print(f"[#{rank:02d}] {cid} — ERROR: {exc}", file=sys.stderr)
            # Save what we have so far and continue
            OUTPUT_JSON.write_text(
                json.dumps(rich_reasoning, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    print("=" * 70)
    print(f"\nDone. {len(rich_reasoning)} entries saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
