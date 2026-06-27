"""
FitRank Streamlit demo — upload a candidates JSON, run the full ranking pipeline,
view ranked results with per-candidate feature breakdown.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="FitRank", page_icon="🏆", layout="wide")
st.title("FitRank — Candidate Ranking Engine")
st.caption("Senior AI Engineer · Redrob · Founding Team")

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    role_model_path = ROOT / "role_model.yaml"
    artifacts_dir   = ROOT / "artifacts"
    st.info(f"Role model: `role_model.yaml`")
    st.info(f"Artifacts: `artifacts/`")

    # Show which embedding artifact will be used
    emb_full   = artifacts_dir / "embeddings.npy"
    emb_sample = artifacts_dir / "sample_embeddings.npy"
    if emb_full.exists():
        st.success("Using full embeddings (100K corpus)")
    elif emb_sample.exists():
        st.warning("Using sample embeddings (500-candidate corpus)")
    else:
        st.error("No embeddings found — run precompute first")

    top_k         = st.slider("Retrieval top-K", 5, 50, 20, step=5)
    show_features = st.checkbox("Show feature breakdown", value=True)

# ── file upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload candidates JSON (array of candidate objects)",
    type=["json"],
    help="Accepts the same format as data/sample_candidates.json",
)

run_btn = st.button("Run ranking", type="primary", disabled=uploaded is None)

if uploaded is None:
    st.info("Upload a candidates JSON file, then click **Run ranking**.")
    st.stop()

if not run_btn:
    st.stop()

# ── run pipeline ──────────────────────────────────────────────────────────────
# Parse candidates from upload once — used both for ranking and feature display
raw_bytes   = uploaded.read()
candidates_list: list[dict] = json.loads(raw_bytes)
n_uploaded  = len(candidates_list)

# Write to temp file so rank.run() can read it
tmp_json = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb")
tmp_json.write(raw_bytes)
tmp_json.flush()
tmp_json.close()
tmp_path  = Path(tmp_json.name)
out_path  = tmp_path.with_suffix(".csv")

# submission_size must not exceed the number of uploaded candidates
submission_size = min(100, n_uploaded)

ranked_rows: list[dict] = []
pipeline_log: list[str] = []

with st.spinner(f"Ranking {n_uploaded} candidates …"):
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            from src.rank import run as rank_run
            rank_run(
                artifacts_dir   = artifacts_dir,
                candidates_path = tmp_path,
                role_model_path = role_model_path,
                output_path     = out_path,
                top_k           = min(top_k, n_uploaded),
                submission_size = submission_size,
            )
        pipeline_log = buf.getvalue().splitlines()
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        st.exception(exc)
        tmp_path.unlink(missing_ok=True)
        st.stop()

# Read results
import csv
with open(out_path, newline="", encoding="utf-8") as f:
    ranked_rows = list(csv.DictReader(f))

# Index candidates by ID for lookups
candidates_by_id: dict[str, dict] = {c["candidate_id"]: c for c in candidates_list}

# Load role model + compute feature vectors (cosine_sim re-derived from embeddings)
import numpy as np
import yaml
from src.feature_builder import build_feature_vector

with open(role_model_path, encoding="utf-8") as f:
    role_model: dict = yaml.safe_load(f)

# Load embeddings for accurate cosine re-computation in feature panel
_emb_path = emb_full if emb_full.exists() else (emb_sample if emb_sample.exists() else None)
_embeddings    = np.load(_emb_path).astype(np.float32) if _emb_path else None
_candidate_ids = None
_jd_vector     = None
_id_to_idx: dict[str, int] = {}

if _emb_path:
    _ids_path = artifacts_dir / (_emb_path.name.replace("embeddings", "candidate_ids"))
    if _ids_path.exists():
        _candidate_ids = np.load(_ids_path, allow_pickle=True)
        _id_to_idx = {str(cid): i for i, cid in enumerate(_candidate_ids)}
    _jd_path = artifacts_dir / "jd_vector.npy"
    if _jd_path.exists():
        _jd_vector = np.load(_jd_path).astype(np.float32).squeeze()

def _cosine_for(cid: str) -> float:
    if _embeddings is None or _jd_vector is None:
        return 0.0
    idx = _id_to_idx.get(cid)
    if idx is None:
        return 0.0
    return float(np.dot(_embeddings[idx], _jd_vector))

# Detect scorer mode from pipeline log
scorer_mode = "weighted-sum"
for line in pipeline_log:
    if "LambdaMART" in line:
        scorer_mode = "LambdaMART"
        break

# ── summary bar ──────────────────────────────────────────────────────────────
st.success(f"Done — {len(ranked_rows)} candidates ranked.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Candidates uploaded", n_uploaded)
c2.metric("Ranked", len(ranked_rows))
c3.metric("Top score", f"{float(ranked_rows[0]['score']):.4f}" if ranked_rows else "–")
c4.metric("Scorer", scorer_mode)

with st.expander("Pipeline log", expanded=False):
    st.code("\n".join(pipeline_log))

st.divider()

# ── ranked table ─────────────────────────────────────────────────────────────
st.subheader("Ranked results")

import pandas as pd

table_data = []
for row in ranked_rows:
    cand    = candidates_by_id.get(row["candidate_id"], {})
    profile = cand.get("profile", {})
    sigs    = cand.get("redrob_signals", {})
    table_data.append({
        "Rank":         int(row["rank"]),
        "Candidate ID": row["candidate_id"],
        "Score":        float(row["score"]),
        "Title":        profile.get("current_title", ""),
        "Company":      profile.get("current_company", ""),
        "YoE":          profile.get("years_of_experience", ""),
        "Location":     profile.get("location", ""),
        "Notice (d)":   sigs.get("notice_period_days", ""),
        "Open":         "✓" if sigs.get("open_to_work_flag") else "✗",
    })

df = pd.DataFrame(table_data)
st.dataframe(
    df,
    use_container_width=True,
    column_config={
        "Score":      st.column_config.NumberColumn(format="%.4f"),
        "Rank":       st.column_config.NumberColumn(width="small"),
        "Open":       st.column_config.TextColumn(width="small"),
        "Notice (d)": st.column_config.NumberColumn(width="small"),
    },
    hide_index=True,
)

st.divider()

# ── per-candidate expandable rows ─────────────────────────────────────────────
FEATURE_META = {
    # key: (label, is_penalty)
    "cosine_similarity":     ("Cosine similarity",    False),
    "experience_fit_score":  ("Experience fit",       False),
    "is_ml_engineer":        ("ML engineer title",    False),
    "production_ml_score":   ("Production ML",        False),
    "domain_alignment":      ("Domain alignment",     False),
    "behavioral_multiplier": ("Behavioral signal",    False),
    "consistency_score":     ("Consistency",          False),
    "location_score":        ("Location",             False),
    "github_activity":       ("GitHub activity",      False),
    "consulting_penalty":    ("Consulting penalty",   True),
    "notice_penalty":        ("Notice penalty",       True),
    "title_disqualified":    ("Title disqualified",   True),
}

n_detail = min(len(ranked_rows), 10)
st.subheader(f"Candidate detail — top {n_detail}")

for row in ranked_rows[:n_detail]:
    cid     = row["candidate_id"]
    cand    = candidates_by_id.get(cid, {})
    profile = cand.get("profile", {})
    sigs    = cand.get("redrob_signals", {})
    score   = float(row["score"])
    rank    = int(row["rank"])

    title   = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "?")
    loc     = profile.get("location", "")
    yoe     = profile.get("years_of_experience", "?")
    notice  = sigs.get("notice_period_days", "?")
    open_   = "Open to work" if sigs.get("open_to_work_flag") else "Not open"

    label = f"#{rank}  {cid}  —  {title} @ {company}  |  score {score:.4f}"
    with st.expander(label, expanded=(rank == 1)):

        # ── profile header ────────────────────────────────────────────────
        hc1, hc2, hc3, hc4 = st.columns(4)
        hc1.metric("YoE", yoe)
        hc2.metric("Location", loc or "—")
        hc3.metric("Notice", f"{notice}d" if isinstance(notice, int) else str(notice))
        hc4.metric("Availability", open_)

        st.markdown(f"**Reasoning:** {row['reasoning']}")

        if not show_features:
            continue

        st.markdown("---")
        st.markdown("**Feature breakdown**")

        # Re-compute features with accurate cosine
        cosine   = _cosine_for(cid)
        features = build_feature_vector(cand, role_model, cosine_sim=cosine)

        # Split positive vs penalty features
        pos_feats = {
            label_: features.get(k, 0.0)
            for k, (label_, is_pen) in FEATURE_META.items()
            if not is_pen and k != "title_disqualified"
        }
        pen_feats = {
            label_: features.get(k, 0.0)
            for k, (label_, is_pen) in FEATURE_META.items()
            if is_pen
        }

        fc1, fc2 = st.columns([1, 1])

        # Table of all features
        with fc1:
            feat_rows = []
            for k, (label_, is_pen) in FEATURE_META.items():
                val = features.get(k, 0.0)
                feat_rows.append({
                    "Feature": label_,
                    "Value":   round(val, 4),
                    "Type":    "penalty" if is_pen else "signal",
                })
            feat_df = pd.DataFrame(feat_rows)
            st.dataframe(
                feat_df,
                use_container_width=True,
                column_config={
                    "Value": st.column_config.NumberColumn(format="%.4f"),
                    "Type":  st.column_config.TextColumn(width="small"),
                },
                hide_index=True,
            )

        # Bar chart for positive signals
        with fc2:
            if pos_feats:
                pos_df = pd.DataFrame(
                    {"Feature": list(pos_feats.keys()), "Score": list(pos_feats.values())}
                ).set_index("Feature")
                st.bar_chart(pos_df, height=280)
            if any(v > 0 for v in pen_feats.values()):
                st.markdown("**Penalties**")
                pen_df = pd.DataFrame(
                    {"Penalty": list(pen_feats.keys()), "Value": list(pen_feats.values())}
                ).set_index("Penalty")
                st.bar_chart(pen_df, height=150, color="#ff4b4b")

# ── download ──────────────────────────────────────────────────────────────────
st.divider()
csv_bytes = out_path.read_bytes()
st.download_button(
    label="Download submission.csv",
    data=csv_bytes,
    file_name="submission.csv",
    mime="text/csv",
)

# cleanup temp files
tmp_path.unlink(missing_ok=True)
out_path.unlink(missing_ok=True)
