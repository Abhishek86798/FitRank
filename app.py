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

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="FitRank", page_icon="🏆", layout="wide")

st.title("🏆 FitRank — Candidate Ranking Engine")
st.caption("Senior AI Engineer · Redrob · Founding Team")

# ── sidebar: controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    role_model_path = Path("role_model.yaml")
    st.info(f"Role model: `{role_model_path}`")

    artifacts_dir = Path("artifacts")
    st.info(f"Artifacts dir: `{artifacts_dir}`")

    top_k = st.slider("Retrieval top-K", min_value=10, max_value=200, value=50, step=10)
    show_features = st.checkbox("Show feature breakdown", value=True)

# ── main: file upload ─────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload candidates JSON (array of candidate objects)",
    type=["json"],
    help="Accepts the same format as data/sample_candidates.json",
)

run_btn = st.button("▶ Run ranking", type="primary", disabled=uploaded is None)

if uploaded is None:
    st.info("Upload a candidates JSON file, then click **Run ranking**.")
    st.stop()

if not run_btn:
    st.stop()

# ── pipeline ──────────────────────────────────────────────────────────────────
with st.spinner("Running pipeline…"):
    # Write upload to a temp file so rank.run() can stream it
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    out_path = tmp_path.with_suffix(".csv")

    try:
        # Add project root to sys.path so src.* imports work
        root = Path(__file__).parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from src.rank import run as rank_run

        rank_run(
            artifacts_dir=artifacts_dir,
            candidates_path=tmp_path,
            role_model_path=role_model_path,
            output_path=out_path,
            top_k=top_k,
            submission_size=100,
        )
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        st.exception(exc)
        st.stop()

# ── load results ──────────────────────────────────────────────────────────────
import csv

rows: list[dict] = []
with open(out_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)

# Load candidate profiles for feature display
candidates_by_id: dict[str, dict] = {}
tmp_path.seek(0) if hasattr(tmp_path, "seek") else None
for cand in json.loads(tmp_path.read_bytes()):
    candidates_by_id[cand["candidate_id"]] = cand

# Build feature vectors for display
import yaml
from src.feature_builder import build_feature_vector

with open(role_model_path, encoding="utf-8") as f:
    role_model: dict = yaml.safe_load(f)

# ── summary metrics ───────────────────────────────────────────────────────────
st.success(f"Ranked {len(rows)} candidates.")

top = rows[:10]
col1, col2, col3 = st.columns(3)
col1.metric("Total ranked", len(rows))
col2.metric("Top score", f"{float(rows[0]['score']):.4f}" if rows else "–")
col3.metric("#1 candidate", rows[0]["candidate_id"] if rows else "–")

st.divider()

# ── ranked table ─────────────────────────────────────────────────────────────
st.subheader("Ranked results")

import pandas as pd

table_data = []
for row in rows[:50]:   # show top-50 in table
    cand = candidates_by_id.get(row["candidate_id"], {})
    profile = cand.get("profile", {})
    table_data.append({
        "Rank": int(row["rank"]),
        "Candidate ID": row["candidate_id"],
        "Score": float(row["score"]),
        "Title": profile.get("current_title", ""),
        "Company": profile.get("current_company", ""),
        "YoE": profile.get("years_of_experience", ""),
        "Location": profile.get("location", ""),
        "Reasoning": row["reasoning"][:120] + "…" if len(row["reasoning"]) > 120 else row["reasoning"],
    })

df = pd.DataFrame(table_data)
st.dataframe(
    df,
    use_container_width=True,
    column_config={
        "Score": st.column_config.NumberColumn(format="%.4f"),
        "Rank": st.column_config.NumberColumn(width="small"),
    },
    hide_index=True,
)

# ── per-candidate feature breakdown ──────────────────────────────────────────
if show_features:
    st.divider()
    st.subheader("Feature breakdown — top 10")

    FEATURE_LABELS = {
        "cosine_similarity":     "Cosine sim",
        "experience_fit_score":  "Experience fit",
        "is_ml_engineer":        "ML engineer title",
        "production_ml_score":   "Production ML",
        "domain_alignment":      "Domain alignment",
        "behavioral_multiplier": "Behavioral",
        "consistency_score":     "Consistency",
        "location_score":        "Location",
        "github_activity":       "GitHub activity",
        "consulting_penalty":    "Consulting penalty",
        "notice_penalty":        "Notice penalty",
        "title_disqualified":    "Title disqualified",
    }

    for row in rows[:10]:
        cid = row["candidate_id"]
        cand = candidates_by_id.get(cid, {})
        profile = cand.get("profile", {})
        score = float(row["score"])

        with st.expander(
            f"#{row['rank']}  {cid}  —  {profile.get('current_title', 'Unknown')} @ "
            f"{profile.get('current_company', '?')}  |  score={score:.4f}"
        ):
            st.markdown(f"**Reasoning:** {row['reasoning']}")
            st.markdown("---")

            # Feature vector
            features = build_feature_vector(cand, role_model, cosine_sim=float(row["score"]))
            feat_rows = []
            for key, label in FEATURE_LABELS.items():
                val = features.get(key, 0.0)
                feat_rows.append({"Feature": label, "Value": val})

            feat_df = pd.DataFrame(feat_rows)
            col_feat, col_bar = st.columns([1, 2])
            col_feat.dataframe(
                feat_df,
                use_container_width=True,
                column_config={"Value": st.column_config.NumberColumn(format="%.4f")},
                hide_index=True,
            )
            # Bar chart for positive features
            positive_feats = {
                FEATURE_LABELS[k]: v for k, v in features.items()
                if k not in ("title_disqualified", "consulting_penalty", "notice_penalty") and v > 0
            }
            if positive_feats:
                col_bar.bar_chart(positive_feats)

# ── download button ───────────────────────────────────────────────────────────
st.divider()
with open(out_path, "rb") as f:
    st.download_button(
        label="⬇ Download submission.csv",
        data=f,
        file_name="submission.csv",
        mime="text/csv",
    )

# cleanup
tmp_path.unlink(missing_ok=True)
out_path.unlink(missing_ok=True)
