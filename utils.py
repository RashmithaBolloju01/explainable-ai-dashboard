"""
utils.py
--------
Shared constants and helper functions used across the project.

Keeping feature definitions, formatting helpers and the plain-English
explanation generator in one place avoids duplicating logic between
train.py, app.py and explain.py.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
# We simulate a loan-default dataset with sklearn's make_classification and
# then relabel / rescale the 6 generated features into meaningful,
# human-readable business features. This keeps the project dependency-free
# (no external dataset download) while still "telling a story" recruiters
# can relate to.
#
# Each entry defines:
#   - the display name
#   - the slider range (min, max)
#   - a default value (used to prefill the dashboard)
#   - the direction of "riskiness" (used only for the natural language
#     explanation, purely descriptive/cosmetic)
FEATURE_CONFIG = {
    "Annual_Income": {
        "label": "Annual Income ($)",
        "min": 15000,
        "max": 200000,
        "default": 55000,
        "step": 1000,
        "risky_when": "low",
    },
    "Debt_to_Income_Ratio": {
        "label": "Debt-to-Income Ratio (%)",
        "min": 0,
        "max": 100,
        "default": 35,
        "step": 1,
        "risky_when": "high",
    },
    "Credit_Score": {
        "label": "Credit Score",
        "min": 300,
        "max": 850,
        "default": 650,
        "step": 5,
        "risky_when": "low",
    },
    "Loan_Amount": {
        "label": "Requested Loan Amount ($)",
        "min": 1000,
        "max": 100000,
        "default": 20000,
        "step": 500,
        "risky_when": "high",
    },
    "Employment_Years": {
        "label": "Years of Employment",
        "min": 0,
        "max": 40,
        "default": 5,
        "step": 1,
        "risky_when": "low",
    },
    "Age": {
        "label": "Applicant Age",
        "min": 18,
        "max": 75,
        "default": 35,
        "step": 1,
        "risky_when": None,
    },
}

FEATURE_NAMES = list(FEATURE_CONFIG.keys())

CLASS_LABELS = {0: "Low Risk", 1: "Likely Default"}

MODEL_PATH = "model.pkl"
DATA_PATH = "training_data.csv"


def scale_raw_features(X_raw: np.ndarray) -> np.ndarray:
    """
    Rescale the raw numeric output of make_classification (roughly
    standard-normal values) into the human-readable ranges defined in
    FEATURE_CONFIG. This is what turns "Feature 1, Feature 2 ..." into
    "Annual Income, Debt-to-Income Ratio ...".

    Parameters
    ----------
    X_raw : np.ndarray of shape (n_samples, n_features)
        Raw features straight out of sklearn.datasets.make_classification.

    Returns
    -------
    np.ndarray of the same shape, rescaled feature-by-feature.
    """
    X_scaled = np.zeros_like(X_raw, dtype=float)

    for idx, name in enumerate(FEATURE_NAMES):
        cfg = FEATURE_CONFIG[name]
        col = X_raw[:, idx]

        # Normalize the raw column to a 0-1 range using a robust min/max
        # (clip extreme outliers from make_classification's gaussian blobs).
        col_min, col_max = np.percentile(col, 1), np.percentile(col, 99)
        col_clipped = np.clip(col, col_min, col_max)
        normalized = (col_clipped - col_min) / (col_max - col_min + 1e-9)

        # Map into the business-friendly [min, max] range for this feature.
        X_scaled[:, idx] = cfg["min"] + normalized * (cfg["max"] - cfg["min"])

    return X_scaled


def format_currency(value: float) -> str:
    """Format a number as a currency string, e.g. 55000 -> '$55,000'."""
    return f"${value:,.0f}"


def get_feature_display_value(name: str, value: float) -> str:
    """Return a nicely formatted value for a given feature (currency, %, etc)."""
    if "Income" in name or "Amount" in name:
        return format_currency(value)
    if "Ratio" in name:
        return f"{value:.1f}%"
    return f"{value:.0f}"


def build_natural_language_explanation(feature_contributions: list) -> str:
    """
    Build a short, human-readable sentence summarizing the top SHAP
    contributors to a prediction.

    Parameters
    ----------
    feature_contributions : list of tuples (feature_name, shap_value, raw_value)
        Should already be sorted by absolute SHAP value, descending.

    Returns
    -------
    str : a plain-English explanation, e.g.
          "The prediction is mainly influenced by high debt-to-income ratio
           and low credit score."
    """
    if not feature_contributions:
        return "No strong contributing factors were found for this prediction."

    top_factors = feature_contributions[:3]
    phrases = []

    for name, shap_val, raw_val in top_factors:
        cfg = FEATURE_CONFIG.get(name, {})
        label = cfg.get("label", name).split(" (")[0]  # strip units for prose
        direction = "high" if shap_val > 0 else "low"
        phrases.append(f"{direction} {label.lower()}")

    if len(phrases) == 1:
        joined = phrases[0]
    elif len(phrases) == 2:
        joined = f"{phrases[0]} and {phrases[1]}"
    else:
        joined = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"

    return f"The prediction is mainly influenced by {joined}."
