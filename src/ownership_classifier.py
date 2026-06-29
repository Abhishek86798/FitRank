"""
Ownership intent classifier — measures HOW a candidate engaged with their work,
not just WHAT keywords appear.

"I tested the API for the ranking system" → low score
"I built the ranking system"              → high score

This is intentionally a deterministic regex scorer (no LLM, zero latency cost)
designed to be composed inside _production_ml_score().
"""

from __future__ import annotations

import re

# ── verb patterns ─────────────────────────────────────────────────────────────

# Strong ownership: first-person delivery verbs that imply full accountability
_STRONG_VERBS = [
    "built", "shipped", "designed", "owned", "led", "architected",
    "created", "deployed end-to-end", "implemented from scratch",
    "took it from", "migrated", "overhauled", "rewrote", "launched",
    "delivered", "spearheaded", "drove end-to-end", "sole engineer",
    "independently built", "independently designed", "independently implemented",
]

# Moderate ownership: delivery verbs with shared or supervised scope
_MODERATE_VERBS = [
    "developed", "implemented", "wrote", "trained and shipped",
    "contributed", "improved", "optimized", "extended",
    "refactored", "maintained", "scaled", "deployed",
    "productionized", "fine-tuned", "retrained", "re-ranked",
]

# Peripheral: usage/observation verbs that dilute ownership signal
_PERIPHERAL_VERBS = [
    "tested", "used", "attended", "familiar with", "knowledge of",
    "exposure to", "assisted", "supported", "helped build",
    "worked with", "integrated with", "evaluated",
    "explored", "studied", "read about", "aware of",
    "collaborated on", "reviewed", "shadowed",
]

# Pre-compile — case-insensitive, word-boundary aware
def _compile(verbs: list[str]) -> re.Pattern[str]:
    # Sort longest first to avoid shorter patterns shadowing longer ones
    sorted_verbs = sorted(verbs, key=len, reverse=True)
    pattern = "|".join(re.escape(v) for v in sorted_verbs)
    return re.compile(r"(?<!\w)(" + pattern + r")(?!\w)", re.IGNORECASE)


_STRONG_RE     = _compile(_STRONG_VERBS)
_MODERATE_RE   = _compile(_MODERATE_VERBS)
_PERIPHERAL_RE = _compile(_PERIPHERAL_VERBS)

# ── sentence splitter ─────────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    """Split text into sentences, keeping each reasonably sized."""
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


# ── main scorer ───────────────────────────────────────────────────────────────

def ownership_score(description: str, domain_keywords: list[str]) -> float:
    """
    Returns 0.0–1.0 measuring genuine ownership of relevant work.

    Does NOT require domain keywords to be present — measures HOW the
    candidate engaged with their work, not just WHAT keywords appear.

    Algorithm
    ---------
    1. Split description into sentences.
    2. For each sentence, count strong/moderate/peripheral verb hits.
       Weight: strong=1.0, moderate=0.5, peripheral=-0.5.
       Sentences containing a domain keyword get 1.5× weight on their score.
    3. Add a global ownership pass: strong/moderate verbs anywhere in the
       text contribute a base signal even without domain keywords.
    4. Normalise to [0, 1] and clamp.

    Returns 0.0 for empty or whitespace-only descriptions.
    """
    if not description or not description.strip():
        return 0.0

    # Build domain keyword regex (dynamic per call, but these sets are small)
    if domain_keywords:
        kw_pattern = "|".join(re.escape(k) for k in domain_keywords)
        domain_re  = re.compile(r"(?<!\w)(" + kw_pattern + r")(?!\w)", re.IGNORECASE)
    else:
        domain_re = None

    sentences = _sentences(description)

    sentence_scores: list[float] = []
    for sent in sentences:
        strong     = len(_STRONG_RE.findall(sent))
        moderate   = len(_MODERATE_RE.findall(sent))
        peripheral = len(_PERIPHERAL_RE.findall(sent))

        raw = strong * 1.0 + moderate * 0.5 - peripheral * 0.5

        # Domain-relevant sentences carry more weight
        domain_hit = bool(domain_re and domain_re.search(sent))
        weight = 1.5 if domain_hit else 1.0

        sentence_scores.append(raw * weight)

    # Sentence-level aggregate (mean of non-zero sentences to avoid dilution
    # from neutral sentences that have no verb hits at all)
    non_zero = [s for s in sentence_scores if s != 0.0]
    sentence_agg = sum(non_zero) / len(non_zero) if non_zero else 0.0

    # Global ownership pass — strong verbs anywhere give a floor signal
    total_strong     = len(_STRONG_RE.findall(description))
    total_moderate   = len(_MODERATE_RE.findall(description))
    total_peripheral = len(_PERIPHERAL_RE.findall(description))
    global_raw = total_strong * 1.0 + total_moderate * 0.5 - total_peripheral * 0.5

    # Blend: 60% sentence-level context, 40% global signal
    # Normalise global by assuming 3 strong hits = saturated ownership
    global_norm = global_raw / 3.0
    blended = 0.6 * sentence_agg + 0.4 * global_norm

    # Scale so that a typical "built X" description lands ~0.7+ and
    # "tested X" lands ~0.1–0.2.  Divide by 1.3 to give lean single-verb
    # sentences the headroom they deserve.
    scaled = blended / 1.3

    return round(min(1.0, max(0.0, scaled)), 4)
