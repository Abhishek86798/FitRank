# Takes feature vectors from feature_builder.py and returns a single float score.
# Primary mode: LightGBM LambdaMART (ltr_model.txt). Fallback: weighted-sum using weights from role_model.yaml.

from __future__ import annotations

from pathlib import Path


def score_with_weighted_sum(features: dict[str, float], role_model: dict) -> float:
    """
    Weighted-sum scorer. Returns a float in 0.0–1.0.

    title_disqualified is a hard gate: if -1.0, the candidate is immediately
    assigned a near-zero score (0.01) regardless of other features.

    All other feature weights come from role_model.feature_weights.
    Positive weights contribute to score; penalty features (consulting_penalty,
    notice_penalty) are stored as positive values in the feature dict and their
    weights in role_model are negative — so the subtraction happens via
    weight × value where weight < 0.
    """
    # Hard gate — short-circuit before computing anything else
    if features.get("title_disqualified", 0.0) == -1.0:
        return 0.01

    weights: dict[str, float] = role_model.get("feature_weights", {})

    # Soft domain gate: no ML title AND zero domain alignment → cap at 0.25.
    # Prevents behaviorally-active non-engineers from outranking weak-signal ML candidates.
    no_ml_title  = features.get("is_ml_engineer", 0.0) == 0.0
    no_domain    = features.get("domain_alignment", 0.0) == 0.0
    domain_cap   = 0.25 if (no_ml_title and no_domain) else 1.0

    # Features that contribute positively
    POSITIVE = [
        "cosine_similarity",
        "experience_fit_score",
        "is_ml_engineer",
        "production_ml_score",
        "domain_alignment",
        "behavioral_multiplier",
        "consistency_score",
        "location_score",
        "github_activity",
    ]
    # Features stored as positive values but carry a negative weight
    PENALTY = [
        "consulting_penalty",
        "notice_penalty",
    ]

    raw = 0.0
    weight_sum = 0.0

    for key in POSITIVE:
        w = weights.get(key, 0.0)
        if w <= 0:
            continue
        raw += w * features.get(key, 0.0)
        weight_sum += w

    # Normalise positive contributions to 0-1 range
    score = raw / weight_sum if weight_sum > 0 else 0.0

    # Apply penalty features as absolute deductions (already small from yaml weights)
    for key in PENALTY:
        w = abs(weights.get(key, 0.0))  # stored negative in yaml, take abs
        score -= w * features.get(key, 0.0)

    return round(max(0.0, min(domain_cap, score)), 6)


class LTRScorer:
    """
    LightGBM LambdaMART scorer. Loads model from artifacts/ltr_model.txt.
    Falls back gracefully to weighted-sum if the model file doesn't exist.
    """

    def __init__(self, model_path: str | Path, role_model: dict):
        self._role_model = role_model
        self._booster = None
        model_path = Path(model_path)
        if model_path.exists():
            try:
                import lightgbm as lgb
                self._booster = lgb.Booster(model_file=str(model_path))
            except Exception as e:
                print(f"[LTRScorer] Failed to load model ({e}), using weighted-sum fallback.")

    @property
    def is_ltr(self) -> bool:
        return self._booster is not None

    def score(self, features: dict[str, float]) -> float:
        """Score a single candidate's feature dict. Returns 0.0–1.0."""
        if features.get("title_disqualified", 0.0) == -1.0:
            return 0.01

        if self._booster is None:
            return score_with_weighted_sum(features, self._role_model)

        import numpy as np

        FEATURE_ORDER = [
            "cosine_similarity", "experience_fit_score", "is_ml_engineer",
            "production_ml_score", "domain_alignment", "consulting_penalty",
            "behavioral_multiplier", "consistency_score", "location_score",
            "notice_penalty", "github_activity",
        ]
        vec = np.array([[features.get(k, 0.0) for k in FEATURE_ORDER]], dtype=np.float32)
        raw = float(self._booster.predict(vec)[0])
        import math
        score = 1.0 / (1.0 + math.exp(-raw))
        # Apply domain cap (same logic as weighted-sum path)
        no_ml_title = features.get("is_ml_engineer", 0.0) == 0.0
        no_domain   = features.get("domain_alignment", 0.0) == 0.0
        domain_cap  = 0.25 if (no_ml_title and no_domain) else 1.0
        return round(min(domain_cap, score), 6)

    def score_batch(self, feature_list: list[dict[str, float]]) -> list[float]:
        """Score a list of feature dicts. More efficient than calling score() in a loop."""
        if self._booster is None:
            return [score_with_weighted_sum(f, self._role_model) for f in feature_list]

        import numpy as np, math

        FEATURE_ORDER = [
            "cosine_similarity", "experience_fit_score", "is_ml_engineer",
            "production_ml_score", "domain_alignment", "consulting_penalty",
            "behavioral_multiplier", "consistency_score", "location_score",
            "notice_penalty", "github_activity",
        ]
        matrix = np.array(
            [[f.get(k, 0.0) for k in FEATURE_ORDER] for f in feature_list],
            dtype=np.float32,
        )
        raws = self._booster.predict(matrix)
        scores = [round(1.0 / (1.0 + math.exp(-float(r))), 6) for r in raws]

        # Override hard-gate and domain-capped candidates
        for i, f in enumerate(feature_list):
            if f.get("title_disqualified", 0.0) == -1.0:
                scores[i] = 0.01
            elif f.get("is_ml_engineer", 0.0) == 0.0 and f.get("domain_alignment", 0.0) == 0.0:
                scores[i] = min(0.25, scores[i])

        return scores
