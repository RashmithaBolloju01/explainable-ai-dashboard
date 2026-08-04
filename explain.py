"""
explain.py
----------
All SHAP-related logic lives here: building the explainer, computing SHAP
values for a single prediction, and generating the various explanation
plots (waterfall, bar, summary, force) as matplotlib figures that
Streamlit can render.

Keeping this isolated from app.py makes the SHAP logic reusable / testable
on its own and keeps the Streamlit file focused on layout.
"""

import matplotlib
matplotlib.use("Agg")  # headless backend, required for Streamlit/server use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from utils import FEATURE_NAMES


def get_explainer(model):
    """
    Build a SHAP TreeExplainer for the given tree-based model.
    TreeExplainer is fast and exact for RandomForest / tree ensembles.
    """
    return shap.TreeExplainer(model)


def explain_instance(explainer, X_row: pd.DataFrame):
    """
    Compute SHAP values for a single instance (one row DataFrame) and
    return a shap.Explanation object restricted to the positive
    ("Likely Default", class = 1) class, since that's the outcome we
    want to explain to the user.

    Parameters
    ----------
    explainer : shap.TreeExplainer
    X_row : pd.DataFrame with exactly one row and FEATURE_NAMES columns

    Returns
    -------
    shap.Explanation for the positive class, shape (1, n_features)
    """
    raw_explanation = explainer(X_row)

    # Modern SHAP returns values with shape (n_samples, n_features, n_classes)
    # for binary/multiclass tree models. We slice out class index 1
    # ("Likely Default") so all downstream plots explain that outcome.
    if raw_explanation.values.ndim == 3:
        class_idx = 1
        values = raw_explanation.values[:, :, class_idx]
        base_values = raw_explanation.base_values[:, class_idx]
    else:
        values = raw_explanation.values
        base_values = raw_explanation.base_values

    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=raw_explanation.data,
        feature_names=list(X_row.columns),
    )
    return explanation


def plot_waterfall(explanation, max_display: int = 6):
    """Return a matplotlib figure with a SHAP waterfall plot for one prediction."""
    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(explanation[0], max_display=max_display, show=False)
    plt.tight_layout()
    return fig


def plot_force(explanation):
    """
    Return a matplotlib figure with a SHAP force plot for one prediction.
    Uses matplotlib=True so it renders as a static image (no JS dependency),
    which is what Streamlit needs.
    """
    fig = plt.figure(figsize=(10, 3))
    shap.plots.force(
        explanation[0].base_values,
        explanation[0].values,
        explanation[0].data,
        feature_names=explanation.feature_names,
        matplotlib=True,
        show=False,
    )
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def plot_feature_importance_bar(explanation):
    """Return a matplotlib figure with a SHAP bar plot (feature importance for this prediction)."""
    fig = plt.figure(figsize=(8, 5))
    shap.plots.bar(explanation[0], show=False)
    plt.tight_layout()
    return fig


def plot_summary(model, background_data: pd.DataFrame, sample_size: int = 100):
    """
    Return a matplotlib figure with a SHAP summary (beeswarm) plot computed
    over a sample of the background/training data. This gives a global view
    of feature importance across many examples, complementing the
    single-prediction plots above.
    """
    sample = background_data.sample(
        n=min(sample_size, len(background_data)), random_state=42
    )
    explainer = get_explainer(model)
    raw_explanation = explainer(sample)

    if raw_explanation.values.ndim == 3:
        values = raw_explanation.values[:, :, 1]
        base_values = raw_explanation.base_values[:, 1]
    else:
        values = raw_explanation.values
        base_values = raw_explanation.base_values

    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=raw_explanation.data,
        feature_names=list(sample.columns),
    )

    fig = plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(explanation, show=False)
    plt.tight_layout()
    return fig


def get_contribution_table(explanation) -> pd.DataFrame:
    """
    Build a tidy DataFrame of Feature / Value / SHAP Contribution,
    sorted by absolute contribution (most influential first).
    """
    values = explanation.values[0]
    data = explanation.data[0]
    names = explanation.feature_names

    df = pd.DataFrame(
        {
            "Feature": names,
            "Value": data,
            "SHAP Contribution": values,
        }
    )
    df["Abs Contribution"] = df["SHAP Contribution"].abs()
    df = df.sort_values("Abs Contribution", ascending=False).drop(
        columns="Abs Contribution"
    )
    df = df.reset_index(drop=True)
    return df


def get_top_contributors(explanation, top_n: int = 3):
    """
    Return a list of (feature_name, shap_value, raw_value) tuples for the
    top-N most influential features, sorted by absolute SHAP value.
    Used to build the natural-language explanation in utils.py.
    """
    table = get_contribution_table(explanation)
    top = table.head(top_n)
    return list(zip(top["Feature"], top["SHAP Contribution"], top["Value"]))
