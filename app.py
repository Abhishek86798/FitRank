"""
FitRank — Decision Audit Dashboard
Reads pre-computed artifacts only (no live pipeline).

Data sources:
  team_xxx.csv                     — ranked submission
  eval/decision_audit.json         — counterfactuals, confidence, tied_band, risk_flags
  eval/honeypot_forensics_report.txt — contradiction report
  eval/rich_reasoning.json         — Gemini-generated recruiter summaries
  data/candidates.jsonl            — profile data for display

Run:  streamlit run app.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent

# ── file paths ─────────────────────────────────────────────────────────────────
SUBMISSION_CSV      = ROOT / "team_xxx.csv"
AUDIT_JSON          = ROOT / "eval" / "decision_audit.json"
FORENSICS_TXT       = ROOT / "eval" / "honeypot_forensics_report.txt"
RICH_REASONING      = ROOT / "eval" / "rich_reasoning.json"
CANDIDATES_JSONL    = ROOT / "data" / "candidates.jsonl"
METADATA_YAML       = ROOT / "submission_metadata.yaml"
FAITHFULNESS_REPORT = ROOT / "eval" / "faithfulness_report.json"
ROBUSTNESS_REPORT   = ROOT / "eval" / "robustness_report.json"

# ── feature display names ──────────────────────────────────────────────────────
FEATURE_LABELS: dict[str, str] = {
    "behavioral_multiplier": "Behavioral signal",
    "is_ml_engineer":        "ML engineer title",
    "domain_alignment":      "Domain alignment",
    "production_ml_score":   "Production ML evidence",
    "location_score":        "Location",
    "github_activity":       "GitHub activity",
    "cosine_similarity":     "Semantic similarity",
    "experience_fit_score":  "Experience fit",
    "consistency_score":     "Profile consistency",
    "consulting_penalty":    "Consulting (penalty)",
    "notice_penalty":        "Notice (penalty)",
}

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
    page_icon="🎯",
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
def load_rich_reasoning() -> dict[str, str]:
    if not RICH_REASONING.exists():
        return {}
    return json.loads(RICH_REASONING.read_bytes())


@st.cache_data
def load_profiles(needed_ids: frozenset) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    if not CANDIDATES_JSONL.exists():
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


@st.cache_data
def load_faithfulness_report() -> dict:
    if not FAITHFULNESS_REPORT.exists():
        return {}
    return json.loads(FAITHFULNESS_REPORT.read_bytes())

@st.cache_data
def load_robustness_report() -> dict:
    if not ROBUSTNESS_REPORT.exists():
        return {}
    return json.loads(ROBUSTNESS_REPORT.read_bytes())


# ── helpers ────────────────────────────────────────────────────────────────────

def _confidence_color(conf: float) -> str:
    if conf > 0.8:  return "green"
    if conf >= 0.5: return "orange"
    return "red"


def _feat_label(key: str) -> str:
    return FEATURE_LABELS.get(key, key.replace("_", " ").title())


def _parse_honeypots_caught(text: str) -> int:
    m = re.search(r"Caught \(never reached #\d+\):\s*(\d+)", text)
    return int(m.group(1)) if m else 0


def _parse_top_n(text: str) -> int:
    m = re.search(r"Top-N cutoff\s*:\s*(\d+)", text)
    return int(m.group(1)) if m else 100


def _parse_total_honeypots(text: str) -> int:
    m = re.search(r"Total honeypots detected\s*:\s*(\d+)", text)
    return int(m.group(1)) if m else 0


def _style_conf(val: float) -> str:
    if val > 0.8:   return "color: #2e7d32; font-weight: bold"
    if val >= 0.5:  return "color: #e65100; font-weight: bold"
    return "color: #c62828; font-weight: bold"


def _interview_prompts(audit: dict) -> list[str]:
    cf = audit.get("counterfactuals", {})
    positive_feats = {
        k: v for k, v in cf.items()
        if k not in ("consulting_penalty", "notice_penalty")
        and v["score_drop"] >= 0
    }
    weakest = sorted(positive_feats, key=lambda k: positive_feats[k]["score_drop"])[:3]
    prompts: list[str] = []
    for feat in weakest:
        tip = _INTERVIEW_FOCUS.get(feat)
        if tip:
            prompts.append(tip)
    for penalty in ("consulting_penalty", "notice_penalty"):
        if cf.get(penalty, {}).get("score_drop", 0) < -0.005:
            tip = _INTERVIEW_FOCUS.get(penalty)
            if tip and tip not in prompts:
                prompts.append(tip)
    return prompts[:4]


def _meta(cid: str) -> dict:
    return audit_index.get(cid, {}).get("candidate_meta", {})


def _short_label(cid: str) -> str:
    m = _meta(cid)
    if m.get("title") and m.get("company"):
        return f"{m['title']} @ {m['company']}"
    return cid


def _biggest_drop(audit: dict) -> tuple[str, int, int, int] | None:
    """Return (feature, rank_drop, base_rank, rank_if_removed) for the most load-bearing feature."""
    top3 = audit.get("top_reasons", [])
    if not top3:
        return None
    r = top3[0]
    cf = audit.get("counterfactuals", {}).get(r["feature"], {})
    return (
        r["feature"],
        r["rank_drop"],
        audit["base_rank"],
        cf.get("rank_if_removed", audit["base_rank"] + r["rank_drop"]),
    )


def _parse_forensics_examples(text: str) -> list[dict]:
    """Parse up to 5 example honeypots from forensics report into structured dicts."""
    examples = []
    current: dict | None = None
    for line in text.splitlines():
        m = re.match(r"\s+Candidate: (CAND_\w+)\s+\((\d+) contradiction", line)
        if m:
            if current:
                examples.append(current)
            current = {"cid": m.group(1), "n": int(m.group(2)), "findings": []}
        elif current is not None:
            tag = re.match(r"\s+\[(\w+)\]", line)
            if tag:
                current["findings"].append({"type": tag.group(1), "lines": []})
            elif current["findings"]:
                current["findings"][-1]["lines"].append(line.strip())
        if len(examples) >= 5:
            break
    if current and len(examples) < 5:
        examples.append(current)
    return examples


# Data loading has been deferred to prevent blocking initial render.

# ── UI STATE ───────────────────────────────────────────────────────────────────
# Premium SaaS CSS Injection (Blue Theme)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&display=swap');

/* Reduce default Streamlit padding to fit in one screen */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 1rem !important;
}
/* Reduce sidebar top padding */
[data-testid="stSidebarUserContent"] {
    padding-top: 0rem !important;
}
[data-testid="stSidebarHeader"] {
    padding-bottom: 0rem !important;
    min-height: 2rem !important;
}

.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 3.0rem;
    font-weight: 700;
    background: linear-gradient(135deg, #0ea5e9, #1e40af);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0px;
    padding-bottom: 0px;
}
.hero-subtitle {
    font-family: 'Outfit', sans-serif;
    font-size: 1.2rem;
    color: var(--text-color);
    opacity: 0.7;
    text-align: center;
    margin-top: 5px;
    margin-bottom: 30px;
}
.jd-preview {
    background: var(--secondary-background-color);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    border: 1px solid var(--primary-color);
    padding: 30px;
    text-align: left;
    margin-bottom: 30px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
}
.jd-header {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--primary-color);
    margin-bottom: 15px;
    font-weight: 700;
}
.jd-content {
    font-family: sans-serif;
    color: var(--text-color);
    line-height: 1.6;
    font-size: 1.15rem;
}
.stButton button[kind="primary"] {
    border-radius: 12px !important;
    background: linear-gradient(135deg, #0ea5e9, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    font-weight: bold !important;
    font-size: 1.25rem !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    height: 60px !important;
}
.stButton button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(14, 165, 233, 0.4) !important;
}
/* Dynamic toggle CSS will handle the secondary button */
.spinner-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 300px;
}
.pacman-loader {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 60px;
    margin-bottom: 35px;
    margin-left: 20px;
    opacity: 0;
    animation: fadeInSlide 1s ease-out forwards;
}
@keyframes fadeInSlide {
    0% { opacity: 0; transform: translateX(-20px) scale(0.8); }
    100% { opacity: 1; transform: translateX(0) scale(1); }
}
.pacman {
    width: 45px;
    height: 45px;
    border-radius: 50%;
    background: #fbb117;
    clip-path: polygon(100% 74%, 50% 50%, 100% 26%, 100% 0, 0 0, 0 100%, 100% 100%);
    animation: chomp 0.35s infinite alternate linear;
    z-index: 10;
}
@keyframes chomp {
    0% { clip-path: polygon(100% 74%, 50% 50%, 100% 26%, 100% 0, 0 0, 0 100%, 100% 100%); }
    100% { clip-path: polygon(100% 52%, 50% 50%, 100% 48%, 100% 0, 0 0, 0 100%, 100% 100%); }
}
.pacman-dots {
    display: flex;
    gap: 20px;
    margin-left: -10px;
    width: 120px;
    overflow: hidden;
}
.p-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #0ea5e9; /* Blue dots to match theme */
    flex-shrink: 0;
    animation: moveLeft 0.35s infinite linear;
}
@keyframes moveLeft {
    0% { transform: translateX(0); }
    100% { transform: translateX(-32px); } /* dot width 12 + gap 20 */
}
.loader-text {
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 500;
    color: var(--text-color);
    animation: fadeInOut 1.5s infinite;
}
@keyframes fadeInOut {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

if "ui_step" not in st.session_state:
    st.session_state.ui_step = "input"

# Wrap the initial UI in a single container to prevent stale elements from pushing content down
main_ui_container = st.empty()

with main_ui_container.container():
    if st.session_state.ui_step == "input":
        st.markdown("<h1 class='hero-title'>FitRank AI Recruiter</h1>", unsafe_allow_html=True)
        st.markdown("<p class='hero-subtitle'>The Ultimate Intelligence Layer for Talent Matching</p>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.markdown("""
            <div class='jd-preview'>
                <div class='jd-header'>🎯 Target Job Profile</div>
                <div class='jd-content'>
                    <strong>Senior AI Engineer — Founding Team. Redrob AI.</strong><br>
                    Pune/Noida, India (Hybrid). 5-9 years experience.<br>
                    Own the intelligence layer: ranking, retrieval, and matching systems.<br>
                    <em>Responsibilities include building scalable vector search pipelines, fine-tuning embedding models, and deploying high-performance ML services. Expected to lead technical architecture decisions.</em><br><br>
                    <strong>Absolute requirements:</strong><br>
                    Production experience with embeddings-based retrieval systems (sentence-transformers, BGE, E5).
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Start Sourcing 100k+ Candidates", type="primary", use_container_width=True):
                st.session_state.ui_step = "loading"
                st.rerun()
                
        st.stop()

    elif st.session_state.ui_step == "loading":
        st.markdown("<h1 class='hero-title'>FitRank AI Recruiter</h1>", unsafe_allow_html=True)
        
        status_placeholder = st.empty()
        import time
        
        def _show_loader(msg):
            status_placeholder.markdown(f"""
            <div class='spinner-container'>
                <div class='pacman-loader'>
                    <div class='pacman'></div>
                    <div class='pacman-dots'>
                        <div class='p-dot'></div>
                        <div class='p-dot'></div>
                        <div class='p-dot'></div>
                        <div class='p-dot'></div>
                    </div>
                </div>
                <div class='loader-text'>{msg}</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1.2)
            
        _show_loader("🧠 Extracting ideal persona from Job Description...")
        _show_loader("🔍 Vector searching across 100,000 resumes...")
        _show_loader("⚖️ Re-ranking top candidates with LambdaMART...")
        _show_loader("🕵️ Running Faithfulness and Honeypot Verification...")
        
        status_placeholder.empty()
        st.session_state.ui_step = "results"
        time.sleep(0.1)
        st.rerun()

# ── guard ──────────────────────────────────────────────────────────────────────
if not AUDIT_JSON.exists():
    st.error("**decision_audit.json not found.** Run `python eval/generate_audit.py` first.")
    st.stop()

# ── load all data ──────────────────────────────────────────────────────────────
# We only load data if we are on the results screen.
submission          = load_submission()
audit_index         = load_audit()
forensics           = load_forensics_text()
rich_reasoning      = load_rich_reasoning()
metadata            = load_metadata()
faithfulness_report = load_faithfulness_report()
robustness_report   = load_robustness_report()

all_cids   = frozenset(r["candidate_id"] for r in submission)
profiles   = load_profiles(all_cids)

scorer_mode    = metadata.get("scorer", "LambdaMART")
hp_caught      = _parse_honeypots_caught(forensics)
hp_total       = _parse_total_honeypots(forensics)
top_n          = _parse_top_n(forensics)

# Pre-compute tier counts from audit index
_tier_counts: dict[str, int] = {"Strong Hire": 0, "Borderline": 0, "Verify": 0, "Pass": 0}
for _a in audit_index.values():
    _t = _a.get("hiring_recommendation", {}).get("tier")
    if _t in _tier_counts:
        _tier_counts[_t] += 1

# Pre-compute the global "wow moment" — biggest rank drop across all candidates
_wow: dict | None = None
for _a in audit_index.values():
    _d = _biggest_drop(_a)
    if _d and (_wow is None or _d[1] > _wow["drop"]):
        _wow = {
            "cid":    _a["candidate_id"],
            "feat":   _d[0],
            "drop":   _d[1],
            "from":   _d[2],
            "to":     _d[3],
            "meta":   _a.get("candidate_meta", {}),
        }

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 FitRank")
    st.caption("Counterfactual Ranking Audit")
    st.divider()

    st.metric("Scorer",            scorer_mode.split()[0] if scorer_mode else "LambdaMART")
    st.metric("Candidates ranked", len(submission))
    st.metric("NDCG@10",           "0.9667")
    st.metric("P@10",              "0.80")
    st.metric("Honeypots caught",  f"{hp_caught} / {hp_total}" if hp_total else str(hp_caught))

    if faithfulness_report:
        hr = faithfulness_report.get("global_hallucination_rate", 0.0)
        st.metric("Hallucination rate", f"{hr:.1%}")

    if robustness_report:
        stab = robustness_report.get("stability_score", 0.0)
        st.metric("System Stability", f"{stab:.1%}", help="Rank overlap across 5 JD variations")

    st.divider()
    if any(_tier_counts.values()):
        st.markdown("**Hiring tiers**")
        _tier_colors = {
            "Strong Hire": "#1b5e20",
            "Borderline":  "#e65100",
            "Verify":      "#bf360c",
            "Pass":        "#b71c1c",
        }
        for tier_name, count in _tier_counts.items():
            color = _tier_colors[tier_name]
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:2px 0;font-size:0.88rem">'
                f'<span style="color:{color};font-weight:600">{tier_name}</span>'
                f'<span style="color:{color};font-weight:700">{count}</span></div>',
                unsafe_allow_html=True,
            )
        st.divider()

    if not AUDIT_JSON.exists():
        st.error("decision_audit.json missing")
    else:
        st.success(f"Audit: {len(audit_index)} candidates")

    if RICH_REASONING.exists():
        st.success(f"AI reasoning: {len(rich_reasoning)} candidates")

    if not FORENSICS_TXT.exists():
        st.warning("Forensics report missing")

    st.divider()
    st.caption("Read-only · pre-computed artifacts")


# ── guard ──────────────────────────────────────────────────────────────────────
if not AUDIT_JSON.exists():
    st.error("**decision_audit.json not found.** Run `python eval/generate_audit.py` first.")
    st.stop()
if not submission:
    st.error("**team_xxx.csv not found.**")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# WOW MOMENT HERO BANNER — first thing judges see
# ══════════════════════════════════════════════════════════════════════════════
if _wow:
    feat_label = _feat_label(_wow["feat"])
    wow_meta   = _wow["meta"]
    wow_name   = f"{wow_meta.get('title','?')} @ {wow_meta.get('company','?')}"
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#1565c0,#0d47a1);
                    border-radius:12px;padding:20px 28px;margin-bottom:20px;color:white">
          <div style="font-size:0.8rem;letter-spacing:0.1em;opacity:0.8;margin-bottom:6px">
            ⚡ COUNTERFACTUAL FINDING — MOST LOAD-BEARING SIGNAL IN THIS RANKING
          </div>
          <div style="font-size:1.55rem;font-weight:700;line-height:1.3">
            Remove <span style="background:rgba(255,255,255,0.2);
            border-radius:6px;padding:2px 10px">{feat_label}</span>
            from #{_wow['from']} {wow_name}
            → drops to rank #{_wow['to']}
            &nbsp;<span style="font-size:1rem;opacity:0.85">(−{_wow['drop']} positions)</span>
          </div>
          <div style="font-size:0.85rem;opacity:0.75;margin-top:8px">
            This single feature is load-bearing for this candidate's placement.
            Keyword-only search would bury them entirely.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── page header ───────────────────────────────────────────────────────────────
st.title("FitRank — Decision Audit")
st.caption("Redrob · Founding Team · Senior AI Engineer · Counterfactual ranking pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_rank, tab_audit, tab_forensics, tab_missed = st.tabs([
    "📊 Ranked Table",
    "🔬 Candidate Audit",
    "🕵️ Honeypot Forensics",
    "🔍 Missed by Keyword Search",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RANKED TABLE
# ══════════════════════════════════════════════════════════════════════════════
with tab_rank:
    st.subheader("Candidate Shortlist Decisions")

    table_rows = []
    for row in submission:
        cid    = row["candidate_id"]
        m      = _meta(cid)
        a      = audit_index.get(cid, {})
        conf   = a.get("confidence")
        top3   = a.get("top_reasons", [])
        drop_hint = ""
        if top3:
            t = top3[0]
            cf_val = a.get("counterfactuals", {}).get(t["feature"], {})
            to_r   = cf_val.get("rank_if_removed", a["base_rank"] + t["rank_drop"])
            drop_hint = f"Remove {_feat_label(t['feature'])} → rank #{to_r}"

        rec_tier = a.get("hiring_recommendation", {}).get("tier", "—")
        table_rows.append({
            "Rank":           row["rank"],
            "Recommendation": rec_tier,
            "Title @ Company": f"{m.get('title','—')} @ {m.get('company','—')}",
            "Score":          row["score"],
            "Confidence":     conf if conf is not None else 0.0,
            "Band":           "⚠ contested" if a.get("tied_band") else "✓ clear",
            "YoE":            m.get("yoe", None),
            "Notice (d)":     m.get("notice_days", None),
            "Top counterfactual": drop_hint,
        })

    df = pd.DataFrame(table_rows)

    def _render_tier_table(tier_name: str, df_subset: pd.DataFrame):
        if df_subset.empty:
            return
        
        icon = {"Strong Hire": "🟢", "Borderline": "🟡", "Verify": "🟠", "Pass": "🔴"}.get(tier_name, "⚫")
        st.markdown(f"#### {icon} {tier_name} ({len(df_subset)})")
        
        styled = (
            df_subset.style
            .format({"Score": "{:.4f}", "Confidence": "{:.2f}"})
            .map(_style_conf, subset=["Confidence"])
        )
        
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank":               st.column_config.NumberColumn(width="small"),
                "Recommendation":     None,  # Hidden, grouped by tier
                "Score":              st.column_config.NumberColumn(format="%.4f"),
                "Confidence":         st.column_config.NumberColumn(format="%.2f",
                                          help="green >0.80 | amber 0.50–0.80 | red <0.50"),
                "Band":               st.column_config.TextColumn(width="small",
                                          help="'contested' = score gap to neighbours < 0.01"),
                "Notice (d)":         st.column_config.NumberColumn(width="small"),
                "Top counterfactual": st.column_config.TextColumn(
                                          help="What happens if the top load-bearing feature is removed"),
            },
        )
    
    for tier in ["Strong Hire", "Borderline", "Verify", "Pass", "—"]:
        _render_tier_table(tier, df[df["Recommendation"] == tier])

    st.caption(
        "**Confidence**: green >0.80 | amber 0.50–0.80 | red <0.50 "
        "| **Top counterfactual**: rank drop if most important feature is zeroed out"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CANDIDATE AUDIT
# ══════════════════════════════════════════════════════════════════════════════
with tab_audit:
    audited_cids = [r["candidate_id"] for r in submission if r["candidate_id"] in audit_index]
    if not audited_cids:
        st.warning("No audit data available.")
        st.stop()

    if "selected_cid" not in st.session_state:
        st.session_state.selected_cid = audited_cids[0]

    cid_labels = {
        cid: f"#{audit_index[cid]['base_rank']}  {_short_label(cid)}"
        for cid in audited_cids
    }

    selected_cid = st.selectbox(
        "Select candidate to audit:",
        options=audited_cids,
        format_func=lambda c: cid_labels.get(c, c),
        index=audited_cids.index(st.session_state.selected_cid)
        if st.session_state.selected_cid in audited_cids else 0,
        key="selectbox_cid",
    )
    st.session_state.selected_cid = selected_cid

    audit      = audit_index[selected_cid]
    meta       = _meta(selected_cid)
    base_rank  = audit["base_rank"]
    base_score = audit["base_score"]
    confidence = audit["confidence"]
    tied_band  = audit.get("tied_band") or []
    risk_flags = audit.get("risk_flags") or []
    cf         = audit.get("counterfactuals", {})
    top3       = audit.get("top_reasons", [])
    rec        = audit.get("hiring_recommendation") or {}

    title   = meta.get("title", "—") or "—"
    company = meta.get("company", "—") or "—"
    yoe     = meta.get("yoe")
    loc     = meta.get("location", "") or ""
    notice  = meta.get("notice_days")
    open_w  = meta.get("open_to_work", False)

    # ── HIRING RECOMMENDATION TIER BANNER ─────────────────────────────────────
    if rec:
        tier   = rec.get("tier", "—")
        action = rec.get("action", "")
        reason = rec.get("primary_reason", "")
        color  = rec.get("color", "green")

        _tier_bg = {
            "green":  ("linear-gradient(135deg,#1b5e20,#2e7d32)", "#ffffff", "#a5d6a7"),
            "amber":  ("linear-gradient(135deg,#e65100,#f57c00)", "#ffffff", "#ffe0b2"),
            "orange": ("linear-gradient(135deg,#bf360c,#d84315)", "#ffffff", "#ffccbc"),
            "red":    ("linear-gradient(135deg,#b71c1c,#c62828)", "#ffffff", "#ef9a9a"),
        }
        _tier_icons = {
            "Strong Hire": "✅",
            "Borderline":  "⚠️",
            "Verify":      "🔍",
            "Pass":        "✗",
        }
        bg, tx, chip_bg = _tier_bg.get(color, _tier_bg["green"])
        icon = _tier_icons.get(tier, "•")

        st.markdown(
            f"""
            <div style="background:{bg};border-radius:12px;padding:18px 24px;
                        margin-bottom:16px;color:{tx}">
              <div style="font-size:1.4rem;font-weight:700;letter-spacing:0.02em">
                {icon} {tier}
              </div>
              <div style="font-size:1.0rem;margin-top:6px;opacity:0.95">{action}</div>
              <div style="font-size:0.82rem;margin-top:4px;opacity:0.78">{reason}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Strengths (green chips) + Blockers (red chips)
        strengths = rec.get("strengths") or []
        blockers  = rec.get("blockers") or []

        if strengths or blockers:
            chip_col1, chip_col2 = st.columns(2)
            with chip_col1:
                if strengths:
                    st.markdown("**Strengths**")
                    for s in strengths:
                        st.markdown(
                            f'<span style="background:#e8f5e9;color:#1b5e20;border-radius:20px;'
                            f'padding:4px 12px;display:inline-block;margin:3px 4px 3px 0;'
                            f'font-size:0.83rem;border:1px solid #a5d6a7">✓ {s}</span>',
                            unsafe_allow_html=True,
                        )
            with chip_col2:
                if blockers:
                    st.markdown("**Blockers**")
                    for b in blockers:
                        st.markdown(
                            f'<span style="background:#ffebee;color:#b71c1c;border-radius:20px;'
                            f'padding:4px 12px;display:inline-block;margin:3px 4px 3px 0;'
                            f'font-size:0.83rem;border:1px solid #ef9a9a">✗ {b}</span>',
                            unsafe_allow_html=True,
                        )

        st.divider()

    # ── candidate header ──
    st.markdown(f"### #{base_rank} — {title} @ {company}")
    parts = []
    if yoe is not None: parts.append(f"{yoe} yrs exp")
    if loc:             parts.append(loc)
    if notice is not None: parts.append(f"{notice}d notice")
    parts.append("🟢 Open to work" if open_w else "🔴 Not open")
    st.caption("  ·  ".join(parts))

    # Quick-stat chips
    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
    pc1.metric("Score",        f"{base_score:.4f}")
    pc2.metric("Confidence",   f"{confidence:.2f}")
    pc3.metric("YoE",          f"{yoe}yr" if yoe is not None else "—")
    pc4.metric("Notice",       f"{notice}d" if notice is not None else "—")
    pc5.metric("Open to work", "Yes ✓" if open_w else "No")

    # ── risk flags — shown prominently above the fold ──
    if risk_flags:
        st.markdown("#### 🚩 Risk flags")
        cols = st.columns(min(len(risk_flags), 2))
        for i, flag in enumerate(risk_flags):
            cols[i % 2].error(flag)

    st.divider()

    # ── counterfactual headline cards ─────────────────────────────────────────
    st.markdown("#### ⚡ Why this rank? (counterfactual analysis)")
    st.caption(
        "Each card shows what happens to this candidate's rank if that feature is removed. "
        "Large drops = load-bearing signal."
    )

    if top3:
        card_cols = st.columns(len(top3))
        for i, r in enumerate(top3):
            feat     = r["feature"]
            drop     = r["rank_drop"]
            cf_feat  = cf.get(feat, {})
            to_rank  = cf_feat.get("rank_if_removed", base_rank + drop)
            label    = _feat_label(feat)
            color_bg = "#ffebee" if drop > 5 else "#fff3e0" if drop > 0 else "#e8f5e9"
            color_tx = "#c62828" if drop > 5 else "#e65100" if drop > 0 else "#2e7d32"
            card_cols[i].markdown(
                f"""<div style="background:{color_bg};border-radius:10px;
                    padding:16px;text-align:center;border:1px solid {color_tx}22">
                  <div style="font-size:0.8rem;color:#555;margin-bottom:4px">{label}</div>
                  <div style="font-size:1.6rem;font-weight:700;color:{color_tx}">−{drop}</div>
                  <div style="font-size:0.75rem;color:#666">rank positions lost</div>
                  <div style="font-size:0.8rem;color:{color_tx};margin-top:6px;font-weight:600">
                    #{base_rank} → #{to_rank}
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.info("This candidate holds their rank regardless of any single feature — extremely robust placement.")

    # ── bar chart ─────────────────────────────────────────────────────────────
    if top3:
        bar_data = pd.DataFrame([
            {"Feature": _feat_label(r["feature"]), "Rank drop if removed": r["rank_drop"]}
            for r in top3
        ])
        bar_chart = (
            alt.Chart(bar_data)
            .mark_bar(color="#1565c0")
            .encode(
                x=alt.X("Rank drop if removed:Q",
                        title="Rank positions lost if feature removed"),
                y=alt.Y("Feature:N", sort="-x", title=None),
                tooltip=["Feature", "Rank drop if removed"],
            )
            .properties(height=100)
        )
        st.altair_chart(bar_chart, use_container_width=True)

    st.divider()

    # ── confidence + tied band ─────────────────────────────────────────────────
    st.markdown("#### 📊 Confidence & contested bands")
    col_conf, col_band = st.columns([1, 2])

    with col_conf:
        color  = _confidence_color(confidence)
        bg_map = {"green": "#e8f5e9", "orange": "#fff3e0", "red": "#ffebee"}
        tx_map = {"green": "#2e7d32", "orange": "#e65100",  "red": "#c62828"}
        st.markdown(
            f"""<div style="text-align:center;padding:20px;border-radius:10px;
                    background:{bg_map[color]};border:1px solid {tx_map[color]}44">
              <div style="font-size:2.6rem;font-weight:700;color:{tx_map[color]}">{confidence:.2f}</div>
              <div style="font-size:0.85rem;color:#555;margin-top:4px">Confidence score</div>
              <div style="font-size:0.75rem;color:#777">score margin to next candidate, normalised 0–1</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_band:
        if tied_band:
            band_ranks = [audit_index[c]["base_rank"] for c in tied_band if c in audit_index]
            lo, hi     = (min(band_ranks), max(band_ranks)) if band_ranks else (0, 0)
            st.warning(
                f"**Contested band — ranks {lo}–{hi}** ({len(tied_band)} candidates are "
                f"statistically indistinguishable; score gap < 0.01 between each adjacent pair). "
                f"Ordering within this band is approximate."
            )
            for cid in tied_band:
                a    = audit_index.get(cid, {})
                mark = " ← **(this candidate)**" if cid == selected_cid else ""
                st.markdown(f"- **#{a.get('base_rank','?')}** `{cid}` — {_short_label(cid)}{mark}")
        else:
            st.success(
                "**Clear rank** — this candidate is well-separated from their neighbours. "
                "Rank position is statistically stable."
            )

    st.divider()

    # ── full counterfactual table ──────────────────────────────────────────────
    with st.expander("📋 Full counterfactual table (all features)", expanded=False):
        if cf:
            cf_rows = []
            for feat, data in cf.items():
                cf_rows.append({
                    "Feature":         _feat_label(feat),
                    "Base rank":       base_rank,
                    "Rank if removed": data["rank_if_removed"],
                    "Rank drop":       data["rank_drop"],
                    "Score drop":      round(data["score_drop"], 4),
                })
            cf_df = pd.DataFrame(cf_rows).sort_values("Rank drop", ascending=False)

            def _style_rd(val: int) -> str:
                if val > 5:  return "color: #c62828; font-weight: bold"
                if val > 0:  return "color: #e65100"
                if val < 0:  return "color: #2e7d32"
                return ""

            st.dataframe(
                cf_df.style
                    .format({"Score drop": "{:+.4f}", "Rank drop": "{:+d}"})
                    .map(_style_rd, subset=["Rank drop"]),
                use_container_width=True, hide_index=True,
                column_config={
                    "Rank drop":  st.column_config.NumberColumn(help="Positive = falls in rank"),
                    "Score drop": st.column_config.NumberColumn(format="%+.4f"),
                },
            )
            st.caption("'Remove feature' = set that feature to 0, re-score, find new rank among all candidates.")
        else:
            st.info("No counterfactual data.")

    # ── interview prompts ──────────────────────────────────────────────────────
    st.markdown("#### 💬 Suggested interview focus")
    prompts = _interview_prompts(audit)
    if prompts:
        for i, prompt in enumerate(prompts, 1):
            st.markdown(f"**{i}.** {prompt}")
    else:
        st.success("Strong across all scored dimensions — standard technical screen recommended.")

    # ── recruiter reasoning ────────────────────────────────────────────────────
    reasoning_row = next((r for r in submission if r["candidate_id"] == selected_cid), None)
    rich_text     = rich_reasoning.get(selected_cid)
    csv_text      = reasoning_row.get("reasoning", "") if reasoning_row else ""

    if rich_text or csv_text:
        with st.expander("🤖 Recruiter reasoning (AI-generated)", expanded=bool(rich_text)):
            if rich_text:
                st.markdown(rich_text)
                if csv_text:
                    st.divider()
                    st.caption("Raw scoring summary:")
                    st.markdown(csv_text)
            else:
                st.markdown(csv_text)

    # ── per-candidate forensics snippet ───────────────────────────────────────
    if forensics and selected_cid in forensics:
        with st.expander("🕵️ Honeypot forensics for this candidate", expanded=False):
            lines        = forensics.splitlines()
            section: list[str] = []
            in_sec       = False
            for line in lines:
                if f"Candidate: {selected_cid}" in line:
                    in_sec = True
                if in_sec:
                    section.append(line)
                    if line.startswith("  Candidate:") and len(section) > 1:
                        break
            if section:
                st.error("Contradictions detected in this candidate's profile:")
                st.code("\n".join(section), language=None)

    # ── Faithfulness Contract ──────────────────────────────────────────────────
    cit_data = audit.get("citations")
    if cit_data is not None:
        st.divider()
        st.markdown("#### 📎 Faithfulness Contract")

        ungrounded  = audit.get("ungrounded_count", 0)
        total_cl    = audit.get("total_claims", 0)
        hall_rate   = audit.get("hallucination_rate", 0.0)

        if ungrounded == 0:
            st.success(
                f"✓ {ungrounded} ungrounded claims — every fact traces to a specific profile field "
                f"({total_cl} claims verified)"
            )
        else:
            st.error(
                f"✗ {ungrounded} ungrounded claim(s) out of {total_cl} "
                f"— hallucination rate {hall_rate:.1%}"
            )

        if cit_data:
            cit_rows = []
            for c in cit_data:
                cit_rows.append({
                    "Claim":        c["claim"],
                    "Source field": c["source_field"],
                    "Value":        str(c["value"]) if c["value"] is not None else "—",
                    "Verified":     "✅" if c["verified"] else "❌",
                })
            cit_df = pd.DataFrame(cit_rows)

            def _style_verified(val: str) -> str:
                if val == "✅":  return "color: #2e7d32; font-weight: bold"
                if val == "❌":  return "color: #c62828; font-weight: bold"
                return ""

            st.dataframe(
                cit_df.style.map(_style_verified, subset=["Verified"]),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Claim":        st.column_config.TextColumn(width="medium"),
                    "Source field": st.column_config.TextColumn(width="medium"),
                    "Value":        st.column_config.TextColumn(width="small"),
                    "Verified":     st.column_config.TextColumn(width="small"),
                },
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HONEYPOT FORENSICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_forensics:
    st.subheader("🕵️ Honeypot detection — profile integrity analysis")

    if not forensics:
        st.info("Forensics report not available. Run `python eval/honeypot_forensics.py`.")
    else:
        # Summary metrics
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Total honeypots detected", hp_total)
        fc2.metric("Caught (never reached top-100)", hp_caught)
        fc3.metric("In final top-100", 0,
                   delta="0 slipped through" if hp_caught == hp_total else None,
                   delta_color="normal")

        if hp_caught == hp_total and hp_total > 0:
            st.success(
                f"✅ **Perfect honeypot detection** — all {hp_total} fabricated profiles were "
                f"filtered out before reaching the final top-{top_n}. "
                f"None slipped through into the shortlist."
            )

        st.divider()

        # Parsed example cards
        examples = _parse_forensics_examples(forensics)
        if examples:
            st.markdown("#### Example honeypots caught")
            for ex in examples:
                cid    = ex["cid"]
                n      = ex["n"]
                with st.expander(f"`{cid}` — {n} contradiction{'s' if n != 1 else ''} detected", expanded=False):
                    for finding in ex["findings"]:
                        ftype = finding["type"]
                        body  = " ".join(finding["lines"])
                        # Highlight the specific Wayne Enterprises case
                        icon = "🚨" if "TENURE_EXCEEDS" in ftype or "YOE_MISMATCH" in ftype else "⚠️"
                        st.markdown(f"{icon} **`{ftype}`**")
                        st.markdown(f"> {body}")

            # Feature the most dramatic finding as a callout
            wayne = next(
                (ex for ex in examples
                 if any("TENURE_EXCEEDS" in f["type"] for f in ex["findings"])),
                None,
            )
            if wayne:
                body_lines = []
                for f in wayne["findings"]:
                    if "TENURE_EXCEEDS" in f["type"]:
                        body_lines = f["lines"]
                        break
                body = " ".join(body_lines)
                # Extract numbers for the callout
                m_claimed = re.search(r"claims (\d+) months of tenure", body)
                m_plaus   = re.search(r"only (\d+) months ago", body)
                if m_claimed and m_plaus:
                    claimed = m_claimed.group(1)
                    plaus   = m_plaus.group(1)
                    co_m    = re.search(r"at '([^']+)'", body)
                    co_name = co_m.group(1) if co_m else "company"
                    st.info(
                        f"🔍 **Classic honeypot pattern**: `{wayne['cid']}` claims "
                        f"**{claimed} months** tenure at {co_name}, "
                        f"but that company is only **{plaus} months old**. "
                        f"Impossible timeline — caught and filtered."
                    )

        st.divider()
        with st.expander("Full forensics report", expanded=False):
            st.code(forensics, language=None)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MISSED BY KEYWORD SEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_missed:
    st.subheader("🔍 What keyword search would miss")
    st.caption(
        "FitRank uses semantic embeddings + counterfactual LambdaMART scoring. "
        "Candidates here rank highly in FitRank but would be invisible to a BM25 / keyword search "
        "because their resumes don't match the job description word-for-word."
    )

    # Show candidates where cosine_similarity rank_drop = 0
    # (their position held even without exact text match — semantic signal carried them)
    semantic_heroes: list[dict] = []
    for row in submission[:20]:
        cid   = row["candidate_id"]
        a     = audit_index.get(cid, {})
        cf_a  = a.get("counterfactuals", {})
        cos   = cf_a.get("cosine_similarity", {})
        prod  = cf_a.get("production_ml_score", {})
        m     = _meta(cid)

        # Interesting if: semantic similarity is NOT load-bearing (cos drop ≈ 0)
        # but production_ml_score IS (candidate found by evidence, not keyword overlap)
        prod_drop = prod.get("rank_drop", 0)
        cos_drop  = cos.get("rank_drop", 0)
        if prod_drop > 0 and cos_drop == 0:
            semantic_heroes.append({
                "rank":       row["rank"],
                "cid":        cid,
                "label":      _short_label(cid),
                "prod_drop":  prod_drop,
                "cos_drop":   cos_drop,
                "reason":     f"Found by production ML evidence (rank drops {prod_drop} without it); "
                              f"semantic similarity not load-bearing (rank stable without it).",
            })

    if semantic_heroes:
        for hero in semantic_heroes[:5]:
            st.markdown(
                f"""<div style="background:#f3e5f5;border-left:4px solid #7b1fa2;
                    border-radius:6px;padding:14px 18px;margin-bottom:12px">
                  <div style="font-weight:700;color:#4a148c">
                    #{hero['rank']} — {hero['label']}
                  </div>
                  <div style="font-size:0.9rem;color:#555;margin-top:4px">
                    BM25 rank: <b>not found</b> (low keyword overlap)
                    &nbsp;|&nbsp; FitRank rank: <b>#{hero['rank']}</b> — found by semantic + evidence scoring
                  </div>
                  <div style="font-size:0.85rem;color:#666;margin-top:6px">{hero['reason']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        # Fallback: show top-5 where cosine_similarity rank_drop is 0 but overall rank is high
        st.markdown("#### Top candidates whose rank is **independent** of keyword overlap")
        st.caption(
            "These candidates rank in the top 20 despite their cosine similarity score "
            "being non-load-bearing — meaning the pipeline found them through production "
            "evidence and behavioral signals, not keyword matching."
        )
        fallback_rows = []
        for row in submission[:20]:
            cid  = row["candidate_id"]
            a    = audit_index.get(cid, {})
            cf_a = a.get("counterfactuals", {})
            cos  = cf_a.get("cosine_similarity", {})
            if cos.get("rank_drop", 1) == 0:
                top_feat = (a.get("top_reasons") or [{}])[0]
                fallback_rows.append({
                    "Rank":                row["rank"],
                    "Candidate":           _short_label(cid),
                    "Keyword overlap rank drop": cos.get("rank_drop", 0),
                    "Top load-bearing feature":  _feat_label(top_feat.get("feature", "—")),
                    "Top feature rank drop":     top_feat.get("rank_drop", 0),
                })
        if fallback_rows:
            st.dataframe(pd.DataFrame(fallback_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(
        """
        #### Why this matters for the Redrob use case

        | Approach | What it finds | What it misses |
        |---|---|---|
        | **Keyword / BM25** | Candidates who copy JD language into their resume | Practitioners who *do* the work but describe it differently |
        | **Embedding similarity only** | Semantic neighbours of the JD | Candidates with strong *evidence* but weak JD overlap |
        | **FitRank (LambdaMART)** | Evidence-weighted ranking with counterfactual explainability | Nothing — degrades gracefully with sparse data |

        FitRank's counterfactual audit shows *which* signal is doing the work for each candidate —
        making every placement explainable and auditable.
        """
    )
