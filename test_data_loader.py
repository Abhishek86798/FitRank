# Smoke test for data_loader.py against the real sample_candidates.json.
# Run: python test_data_loader.py

import json
from pathlib import Path
from src.data_loader import build_candidate_text, stream_candidates

SAMPLE_JSON = Path("data/sample_candidates.json")
FULL_JSONL = Path("data/candidates.jsonl")

# --- Test 1: sample_candidates.json (array) ---
candidates = json.loads(SAMPLE_JSON.read_bytes())
print(f"sample_candidates.json: {len(candidates)} records loaded\n")

first = candidates[0]
text = build_candidate_text(first)
print(f"--- Built text for {first['candidate_id']} ({len(text)} chars) ---")
print(text)
print()

print("--- All sample candidate IDs and text lengths ---")
for c in candidates:
    t = build_candidate_text(c)
    print(f"  {c['candidate_id']}  {len(t):4d} chars  {c['profile'].get('headline','')[:55]}")

print()

# --- Test 2: stream from full candidates.jsonl if present ---
if FULL_JSONL.exists():
    total = 0
    for batch in stream_candidates(FULL_JSONL, batch_size=100):
        total += len(batch)
    print(f"candidates.jsonl: streamed {total} records in batches of 100")
else:
    print("candidates.jsonl not found in data/ — skipping stream test")
    print("Copy it from the organizer folder to data/candidates.jsonl to enable full-dataset tests.")
