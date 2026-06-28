"""
FitRank — Decision Audit Dashboard
Reads pre-computed artifacts only (no live pipeline).

Data sources:
  team_xxx.csv                     — ranked submission
  eval/decision_audit.json         — counterfactuals, confidence, tied_band, risk_flags
  eval/honeypot_forensics_report.txt — contradiction report
  data/candidates.jsonl            — profile data for display

Run:  streamlit run app.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent

# ── file paths ─────────────────────────────────────────────────────────────────
SUBMISSION_CSV   = ROOT / "team_xxx.csv"
AUDIT_JSON       = ROOT / "eval" / "decision_audit.json"
FORENSICS_TXT    = ROOT / "eval" / "honeypot_forensics_report.txt"
CANDIDATES_JSONL = ROOT / "data" / "candidates.jsonl"
METADATA_YAML    = ROOT / "submission_metadata.yaml"

# ── feature display names ──────────────────────────────────────────────────────
FEATURE_LABELS: dict[str, str] = {
    "behavioral_multiplier": "Behavioral signal",
    "is_ml_engineer":        "ML engineer title",
    "domain_alignment":      "Domain alignment",
    "production_ml_score":   "Production ML",
    "location_score":        "Location",
    "github_activity":       "GitHub activity",
    "cosine_similarity":     "Cosine similarity",
    "experience_fit_score":  "Experience fit",
    "consistency_score":     "Consistency",
    "consulting_penalty":    "Consulting (penalty)",
    "notice_penalty":        "Notice (penalty)",
}

# Interview focus templates keyed by weak feature
_INTERVIEW_FOCUS: dict[str, str] = {
    "production_ml_score":   "Ask for a specific shipped retrieval or ranking system: scale, latency, eval metrics.",
    "domain_alignment":      "Probe NLP/IR depth: have them walk through a dense-retrieval architecture they own.",
    "is_ml_engineer":        "Clarify engineering scope: do they write production code or stay in research/notebooks?",
    "cosine_similarity":     "Low semantic match — verify understanding of embedding-based retrieval hands-on.",
    "experience_fit_score":  "Experience band mismatch — calibrate actual scope and independence of past work.",
    "behavioral_multiplier": "Low engagement signal — confirm availability and interest before investing screen time.",
    "location_score":        "Outside preferred geography — confirm relocation intent and timeline explicitly.",
    "github_activity":       "No public portfolio — ask for internal examples or code samples from past work.",
    "consistency_score":     "Profile inconsistencies detected — verify tenure dates and skill claims during screen.",
    "consulting_penalty":    "Consulting-heavy background — probe for product ownership vs staff-aug delivery.",
    "notice_penalty":        "Long notice period — confirm buyout feasibility and start-date flexibility.",
}


# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitRank — Decision Audit",
    page_icon="=",
    layout="wide",
)


# ── data loaders (cached) ──────────────────────────────────────────────────────

@st.cache_data
def load_submission() -> list[dict]:
    if not SUBMISSION_CSV.exists():
        return []
    rows = []
    with open(SUBMISSION_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "rank":         int(row["rank"]),
                "candidate_id": row["candidate_id"].strip(),
                "score":        float(row["score"]),
                "reasoning":    row.get("reasoning", ""),
            })
    rows.sort(key=lambda r: r["rank"])
    return rows


@st.cache_data
def load_audit() -> dict[str, dict]:
    """Returns {candidate_id: audit_dict}."""
    if not AUDIT_JSON.exists():
        return {}
    audits = json.loads(AUDIT_JSON.read_bytes())
    return {a["candidate_id"]: a for a in audits}


@st.cache_data
def load_forensics_text() -> str:
    if not FORENSICS_TXT.exists():
        return ""
    return FORENSICS_TXT.read_text(encoding="utf-8")


@st.cache_data
def load_profiles(needed_ids: frozenset) -> dict[str, dict]:
    """Stream candidates.jsonl and return {id: record} for needed_ids only."""
    profiles: dict[str, dict] = {}
    if not CANDIDATES_JSONL.exists():
        # Fall back to sample JSON
        sample = ROOT / "data" / "sample_candidates.json"
        if sample.exists():
            for c in json.loads(sample.read_bytes()):
                if c["candidate_id"] in needed_ids:
                    profiles[c["candidate_id"]] = c
        return profiles
    try:
        import orjson
        with open(CANDIDATES_JSONL, "rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                c = orjson.loads(line)
                cid = c["candidate_id"]
                if cid in needed_ids:
                    profiles[cid] = c
                if len(profiles) == len(needed_ids):
                    break
    except Exception:
        pass
    return profiles


@st.cache_data
def load_metadata() -> dict:
    if not METADATA_YAML.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(METADATA_YAML.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


# ── helpers ────────────────────────────────────────────────────────────────────

def _confidence_color(conf: float) -> str:
    if conf > 0.8:
        return "green"
    if conf >= 0.5:
        return "orange"
    return "red"


def _confidence_label(conf: float) -> str:
    if conf > 0.8:
        return f":green[{conf:.2f}]"
    if conf >= 0.5:
        return f":orange[{conf:.2f}]"
    return f":red[{conf:.2f}]"


def _feat_label(key: str) -> str:
    return FEATURE_LABELS.get(key, key.replace("_", " ").title())


def _parse_honeypots_caught(text: str) -> int:
    """Extract 'Caught' count from forensics report text."""
    m = re.search(r"Caught \(never reached #\d+\):\s*(\d+)", text)
    return int(m.group(1)) if m else 0


def _parse_top_n(text: str) -> int:
    """Extract top-N cutoff from forensics report."""
    m = re.search(r"Top-N cutoff\s*:\s*(\d+)", text)
    return int(m.group(1)) if m else 100


def _interview_prompts(audit: dict) -> list[str]:
    """Return interview focus lines keyed on the 3 weakest scoring features."""
    cf = audit.get("counterfactuals", {})
    # Weakest = lowest score_drop (feature contributes least / or is a penalty)
    # Show prompts for: bottom positive contributors + any active penalties
    prompts: list[str] = []
    # Top reasons already know the strongest — focus on WEAKEST non-penalty features
    positive_feats = {
        k: v for k, v in cf.items()
        if k not in ("consulting_penalty", "notice_penalty")
        and v["score_drop"] >= 0
    }
    # Sort by score_drop ascending = weakest first
    weakest = sorted(positive_feats, key=lambda k: positive_feats[k]["score_drop"])[:3]
    for feat in weakest:
        tip = _INTERVIEW_FOCUS.get(feat)
        if tip:
            prompts.append(tip)
    # Add penalty prompts if active
    for penalty in ("consulting_penalty", "notice_penalty"):
        if cf.get(penalty, {}).get("score_drop", 0) < -0.005:  # masking it helped → penalty was hurting
            tip = _INTERVIEW_FOCUS.get(penalty)
            if tip and tip not in prompts:
                prompts.append(tip)
    return prompts[:4]


# ── load all data ──────────────────────────────────────────────────────────────

submission  = load_submission()
audit_index = load_audit()
forensics   = load_forensics_text()
metadata    = load_metadata()

all_cids    = frozenset(r["candidate_id"] for r in submission)
profiles    = load_profiles(all_cids)

scorer_mode = metadata.get("scorer", "LambdaMART")
ndcg_10     = metadata.get("eval_metrics", {}).get("NDCG_at_10", 0.0)
hp_caught   = _parse_honeypots_caught(forensics)
top_n       = _parse_top_n(forensics)


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("FitRank")
    st.caption("Decision Audit Dashboard")
    st.divider()

    st.metric("Scorer", scorer_mode.split()[0] if scorer_mode else "—")
    st.metric("Candidates ranked", len(submission))
    st.metric("NDCG@10", f"{ndcg_10:.4f}" if ndcg_10 else "—")
    st.metric("Honeypots caught", hp_caught)

    st.divider()
    if not AUDIT_JSON.exists():
        st.error("decision_audit.json missing — run eval/generate_audit.py first.")
    else:
        n_audited = len(audit_index)
        st.success(f"Audit data loaded ({n_audited} candidates)")

    if not FORENSICS_TXT.exists():
        st.warning("Forensics report missing — run eval/honeypot_forensics.py.")

    st.divider()
    st.caption("Data is read-only. No live recompute.")


# ── missing audit guard ────────────────────────────────────────────────────────
if not AUDIT_JSON.exists():
    st.error(
        "**decision_audit.json not found.** "
        "Run `python eval/generate_audit.py` to generate it, then refresh."
    )
    st.stop()

if not submission:
    st.error("**team_xxx.csv not found.** Place the ranked submission CSV in the project root.")
    st.stop()


# ── main header ───────────────────────────────────────────────────────────────
st.title("Decision Audit — Senior AI Engineer")
st.caption("Redrob · Founding Team · FitRank pipeline")


# ── ranked table ──────────────────────────────────────────────────────────────
st.subheader("Ranked candidates")

table_rows = []
for row in submission:
    cid     = row["candidate_id"]
    profile = profiles.get(cid, {}).get("profile", {})
    audit   = audit_index.get(cid, {})
    conf    = audit.get("confidence", None)
    in_band = bool(audit.get("tied_band"))

    conf_display = (
        f"{conf:.2f}" if conf is not None else "—"
    )
    badge = "contested" if in_band else ""

    table_rows.append({
        "Rank":         row["rank"],
        "ID":           cid,
        "Title":        profile.get("current_title", "—"),
        "Company":      profile.get("current_company", "—"),
        "Score":        row["score"],
        "Confidence":   conf if conf is not None else 0.0,
        "Conf label":   conf_display,
        "Band":         badge,
        "YoE":          profile.get("years_of_experience", "—"),
        "Notice (d)":   profiles.get(cid, {}).get("redrob_signals", {}).get("notice_period_days", "—"),
    })

df = pd.DataFrame(table_rows)

# Color-code confidence column using pandas styler
def _style_conf(val: float) -> str:
    if val > 0.8:
        return "color: #2e7d32; font-weight: bold"
    if val >= 0.5:
        return "color: #e65100; font-weight: bold"
    return "color: #c62828; font-weight: bold"

styled = (
    df[["Rank", "ID", "Title", "Company", "Score", "Confidence", "Band", "YoE", "Notice (d)"]]
    .style
    .format({"Score": "{:.4f}", "Confidence": "{:.2f}"})
    .map(_style_conf, subset=["Confidence"])
)

st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rank":       st.column_config.NumberColumn(width="small"),
        "Score":      st.column_config.NumberColumn(format="%.4f"),
        "Confidence": st.column_config.NumberColumn(format="%.2f"),
        "Band":       st.column_config.TextColumn("Band", width="small",
                          help="'contested' = score gap to neighbours < 0.01"),
        "Notice (d)": st.column_config.NumberColumn(width="small"),
    },
)

st.caption(
    "**Confidence**: green >0.80  |  amber 0.50-0.80  |  red <0.50  "
    "| **'contested'** = tied band (ranks statistically indistinguishable within epsilon=0.01)"
)

st.divider()


# ── candidate selector ────────────────────────────────────────────────────────
st.subheader("Decision Audit Panel")

audited_cids = [r["candidate_id"] for r in submission if r["candidate_id"] in audit_index]
if not audited_cids:
    st.warning("No audit data available for any ranked candidate.")
    st.stop()

# Session-state driven selection — clicking via selectbox
if "selected_cid" not in st.session_state:
    st.session_state.selected_cid = audited_cids[0]

# Build display labels for the selectbox
cid_labels = {
    cid: f"#{audit_index[cid]['base_rank']}  {cid}"
    for cid in audited_cids
}

selected_cid = st.selectbox(
    "Select a candidate to audit:",
    options=audited_cids,
    format_func=lambda c: cid_labels.get(c, c),
    index=audited_cids.index(st.session_state.selected_cid)
    if st.session_state.selected_cid in audited_cids else 0,
    key="selectbox_cid",
)
st.session_state.selected_cid = selected_cid

audit   = audit_index[selected_cid]
profile = profiles.get(selected_cid, {}).get("profile", {})
signals = profiles.get(selected_cid, {}).get("redrob_signals", {})

# ── audit panel ───────────────────────────────────────────────────────────────

base_rank  = audit["base_rank"]
base_score = audit["base_score"]
confidence = audit["confidence"]
tied_band  = audit.get("tied_band") or []
risk_flags = audit.get("risk_flags") or []
cf         = audit.get("counterfactuals", {})
top3       = audit.get("top_reasons", [])

title_line = (
    f"**#{base_rank}  {selected_cid}** — "
    f"{profile.get('current_title', '—')} @ {profile.get('current_company', '—')}"
)
st.markdown(title_line)

# Profile quick-stats
pc1, pc2, pc3, pc4, pc5 = st.columns(5)
pc1.metric("Score",      f"{base_score:.4f}")
pc2.metric("Confidence", f"{confidence:.2f}")
pc3.metric("YoE",        profile.get("years_of_experience", "—"))
pc4.metric("Notice",     f"{signals.get('notice_period_days', '—')}d")
pc5.metric("Open to work", "Yes" if signals.get("open_to_work_flag") else "No")

# ── a. Top 3 load-bearing reasons — horizontal bars ──────────────────────────
st.markdown("#### a. Top load-bearing features")

if top3:
    bar_data = pd.DataFrame([
        {"Feature": _feat_label(r["feature"]), "Rank drop if removed": r["rank_drop"]}
        for r in top3
    ])
    import altair as alt
    bar_chart = (
        alt.Chart(bar_data)
        .mark_bar(color="#1565c0")
        .encode(
            x=alt.X("Rank drop if removed:Q", title="Rank positions lost if feature removed"),
            y=alt.Y("Feature:N", sort="-x", title=None),
            tooltip=["Feature", "Rank drop if removed"],
        )
        .properties(height=120)
    )
    st.altair_chart(bar_chart, use_container_width=True)
    st.caption("Each bar shows how far this candidate would fall in ranking if that feature were zeroed out.")
else:
    st.info("No load-bearing features identified (candidate holds rank regardless of feature masking).")


# ── b. Full counterfactual table ──────────────────────────────────────────────
st.markdown("#### b. Full counterfactual table")

if cf:
    cf_rows = []
    for feat, data in cf.items():
        cf_rows.append({
            "Feature":            _feat_label(feat),
            "Base rank":          base_rank,
            "Rank if removed":    data["rank_if_removed"],
            "Rank drop":          data["rank_drop"],
            "Score drop":         round(data["score_drop"], 4),
        })
    cf_df = pd.DataFrame(cf_rows).sort_values("Rank drop", ascending=False)

    def _style_rank_drop(val: int) -> str:
        if val > 5:   return "color: #c62828; font-weight: bold"
        if val > 0:   return "color: #e65100"
        if val < 0:   return "color: #2e7d32"
        return ""

    styled_cf = (
        cf_df.style
        .format({"Score drop": "{:+.4f}", "Rank drop": "{:+d}"})
        .map(_style_rank_drop, subset=["Rank drop"])
    )
    st.dataframe(styled_cf, use_container_width=True, hide_index=True,
                 column_config={
                     "Rank drop":      st.column_config.NumberColumn(help="Positive = falls in rank"),
                     "Score drop":     st.column_config.NumberColumn(format="%+.4f"),
                 })
    st.caption("Remove {feature} = set that feature to 0, re-score, find new position among all ranked candidates.")
else:
    st.info("No counterfactual data available for this candidate.")


# ── c. Confidence gauge + tied-band note ─────────────────────────────────────
st.markdown("#### c. Confidence & contested bands")

col_conf, col_band = st.columns([1, 2])

with col_conf:
    color = _confidence_color(confidence)
    gauge_md = (
        f"<div style='text-align:center;padding:16px;border-radius:8px;"
        f"background:{'#e8f5e9' if color == 'green' else '#fff3e0' if color == 'orange' else '#ffebee'}'>"
        f"<div style='font-size:2.4rem;font-weight:700;color:"
        f"{'#2e7d32' if color == 'green' else '#e65100' if color == 'orange' else '#c62828'}'>"
        f"{confidence:.2f}</div>"
        f"<div style='font-size:0.85rem;color:#555'>Confidence score</div>"
        f"<div style='font-size:0.75rem;color:#777'>margin to next candidate, normalised 0–1</div>"
        f"</div>"
    )
    st.markdown(gauge_md, unsafe_allow_html=True)

with col_band:
    if tied_band:
        band_ranks = [
            audit_index[c]["base_rank"]
            for c in tied_band
            if c in audit_index
        ]
        if band_ranks:
            lo, hi = min(band_ranks), max(band_ranks)
            st.warning(
                f"**Contested band detected** — "
                f"ranks {lo}–{hi} ({len(tied_band)} candidates) are statistically "
                f"indistinguishable (score gap < 0.01 between each adjacent pair). "
                f"Ordering within this band should be treated as approximate."
            )
        else:
            st.warning(f"**Contested band** — {len(tied_band)} candidates within epsilon=0.01.")
        st.markdown("**Band members:**")
        for cid in tied_band:
            a = audit_index.get(cid, {})
            p = profiles.get(cid, {}).get("profile", {})
            mark = " **(this candidate)**" if cid == selected_cid else ""
            st.markdown(
                f"- #{a.get('base_rank','?')} `{cid}` — "
                f"{p.get('current_title','—')} @ {p.get('current_company','—')}"
                f"{mark}"
            )
    else:
        st.success("This candidate is **not** in a contested band — their rank position is well-separated from neighbours.")


# ── d. Risk flags ─────────────────────────────────────────────────────────────
st.markdown("#### d. Risk flags")

if risk_flags:
    cols = st.columns(min(len(risk_flags), 2))
    for i, flag in enumerate(risk_flags):
        cols[i % 2].error(flag)
else:
    st.success("No risk flags — clean profile on all checked dimensions.")


# ── e. Honeypot forensics ─────────────────────────────────────────────────────
st.markdown("#### e. Honeypot forensics")

if FORENSICS_TXT.exists():
    # Look for this candidate's ID in the forensics report
    if selected_cid in forensics:
        # Extract their section
        lines = forensics.splitlines()
        in_section = False
        section_lines: list[str] = []
        for line in lines:
            if f"Candidate: {selected_cid}" in line:
                in_section = True
            if in_section:
                section_lines.append(line)
                if line.startswith("  Candidate:") and section_lines and len(section_lines) > 1:
                    break
        if section_lines:
            st.error("**Contradictions found in forensics report:**")
            st.code("\n".join(section_lines), language=None)
        else:
            st.success("No contradictions found for this candidate in the forensics report.")
    else:
        st.success("No inconsistencies — candidate not flagged in honeypot forensics scan.")
else:
    st.info("Forensics report not available. Run `python eval/honeypot_forensics.py` to generate it.")


# ── f. Suggested interview focus ──────────────────────────────────────────────
st.markdown("#### f. Suggested interview focus")

prompts = _interview_prompts(audit)
if prompts:
    for i, prompt in enumerate(prompts, 1):
        st.markdown(f"**{i}.** {prompt}")
else:
    st.success("Strong across all scored dimensions — standard technical screen recommended.")

# Show reasoning string from submission CSV
reasoning_row = next((r for r in submission if r["candidate_id"] == selected_cid), None)
if reasoning_row and reasoning_row.get("reasoning"):
    with st.expander("Ranking reasoning string", expanded=False):
        st.markdown(reasoning_row["reasoning"])


st.divider()

# ── forensics summary panel ───────────────────────────────────────────────────
with st.expander("Honeypot forensics report summary", expanded=False):
    if forensics:
        # Show top section only (up to first example)
        summary_lines = []
        for line in forensics.splitlines():
            summary_lines.append(line)
            if "Example honeypots" in line:
                break
        st.code("\n".join(summary_lines), language=None)
    else:
        st.info("No forensics report found.")
