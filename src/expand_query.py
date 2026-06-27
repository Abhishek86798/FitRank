"""
Persona-based query expansion for dense retrieval.

Calls Claude to generate N ideal-candidate profiles from the JD text, embeds
each with BGE, and returns their L2-normalised average as the query vector.
Averaging over multiple "shapes" of the ideal hire widens the retrieval cone
while staying in the correct semantic neighbourhood.

This is an OFFLINE step — run once via precompute.py, output saved to
artifacts/jd_vector.npy.  rank.py is unchanged.

Environment variable required:
    ANTHROPIC_API_KEY — set before running precompute.py
"""

from __future__ import annotations

import os
import textwrap
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

_MODEL = "claude-haiku-4-5-20251001"   # fast + cheap for offline precompute
_N_PROFILES = 5


_SYSTEM = textwrap.dedent("""\
    You are an expert technical recruiter writing ideal-candidate profiles for a
    semantic search index.  Each profile is a dense paragraph (100-150 words)
    written in first-person ("I have…", "I've shipped…") that describes a single
    plausible shape of the ideal hire.  Profiles must be distinct — vary
    seniority emphasis, tech stack specifics, and company type (startup vs
    product vs FAANG).  Do not repeat bullet lists from the JD; synthesise
    believable career narratives that would embed close to the best real
    candidates in the corpus.
""")

_USER_TMPL = textwrap.dedent("""\
    Job description:
    ---
    {jd}
    ---

    Write exactly {n} ideal-candidate profiles, one per numbered section.
    Format:
    1. <profile text>
    2. <profile text>
    …
    Output only the numbered profiles, nothing else.
""")


def _call_claude(jd_text: str, n: int, api_key: str) -> list[str]:
    """Return n profile strings from Claude."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        system=_SYSTEM,
        messages=[{
            "role": "user",
            "content": _USER_TMPL.format(jd=jd_text, n=n),
        }],
    )
    raw = message.content[0].text.strip()
    return _parse_profiles(raw, n)


def _parse_profiles(raw: str, n: int) -> list[str]:
    """Extract the numbered profile bodies from Claude's response."""
    import re
    # Match "1. text" ... "2. text" ... splitting on the next number
    parts = re.split(r"\n?\s*\d+\.\s+", raw)
    profiles = [p.strip() for p in parts if p.strip()]
    if len(profiles) < n:
        # Fallback: return whatever we got rather than failing hard
        pass
    return profiles[:n] if len(profiles) >= n else profiles


def expand_query(
    jd_text: str,
    embed_fn,                   # callable(list[str]) -> np.ndarray (n, 768) float32, L2-normed
    *,
    n_profiles: int = _N_PROFILES,
    api_key: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Generate n_profiles ideal-candidate profiles via Claude, embed each with
    embed_fn, and return their L2-normalised average as a (1, 768) float32 array.

    Parameters
    ----------
    jd_text     : raw JD string (no BGE prefix — embed_fn handles that)
    embed_fn    : function that accepts List[str] and returns (n, 768) float32
    n_profiles  : number of profiles to generate and average (default 5)
    api_key     : ANTHROPIC_API_KEY; falls back to env var if None
    verbose     : print progress lines
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Export it before running precompute.py."
        )

    if verbose:
        print(f"Expanding query: calling Claude ({_MODEL}) for {n_profiles} profiles …")

    profiles = _call_claude(jd_text, n_profiles, key)

    if verbose:
        print(f"  Got {len(profiles)} profiles:")
        for i, p in enumerate(profiles, 1):
            print(f"  [{i}] {p[:80].replace(chr(10), ' ')}…")

    if not profiles:
        raise ValueError("Claude returned zero profiles — cannot build expanded query vector.")

    if verbose:
        print("Embedding profiles …")

    vecs = embed_fn(profiles)          # (k, 768) float32, L2-normalised
    avg  = vecs.mean(axis=0)           # (768,)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm               # re-normalise the average
    return avg.reshape(1, -1).astype(np.float32)
