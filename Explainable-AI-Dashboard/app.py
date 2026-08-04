"""
app.py
------
Main Streamlit application: the Explainable AI Prediction System (XAI
Dashboard). Provides a Home page and a Prediction page (with sidebar
navigation) where users can enter loan-applicant details, get a
prediction, and understand *why* the model made that prediction via
SHAP explanations.

Run with:
    streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st

from explain import (
    explain_instance,
    get_contribution_table,
    get_explainer,
    get_top_contributors,
    plot_feature_importance_bar,
    plot_force,
    plot_summary,
    plot_waterfall,
)
from model import DATA_PATH, generate_dataset, load_or_train_model
from utils import (
    CLASS_LABELS,
    FEATURE_CONFIG,
    FEATURE_NAMES,
    build_natural_language_explanation,
    get_feature_display_value,
)

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Explainable AI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Light custom styling for a more professional, "interview-ready" look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0px;
        }
        .subtitle {
            font-size: 1.05rem;
            color: #6b7280;
            margin-top: 0px;
        }
        .metric-card {
            background-color: #f8f9fb;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            text-align: center;
        }
        .result-card-risk {
            background-color: #fef2f2;
            border: 1px solid #fecaca;
            border-radius: 12px;
            padding: 1.4rem;
        }
        .result-card-safe {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 12px;
            padding: 1.4rem;
        }
        .explanation-box {
            background-color: #eff6ff;
            border-left: 4px solid #3b82f6;
            border-radius: 6px;
            padding: 1rem 1.2rem;
            font-size: 1.02rem;
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid #e5e7eb;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resources: model + background data are expensive to (re)build,
# so we cache them across reruns / user interactions.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model (training on first run)...")
def get_model_and_accuracy():
    """Load the trained model, auto-training it on first run if needed."""
    model, accuracy = load_or_train_model()
    return model, accuracy


@st.cache_data(show_spinner=False)
def get_background_data():
    """Load (or regenerate) the training dataset used for the global SHAP summary plot."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = generate_dataset()
    return df[FEATURE_NAMES]


model, accuracy = get_model_and_accuracy()
background_data = get_background_data()
explainer = get_explainer(model)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🧠 XAI Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔮 Predict & Explain"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
if accuracy is not None:
    st.sidebar.metric("Model Test Accuracy", f"{accuracy * 100:.1f}%")
st.sidebar.caption(
    "Model: RandomForestClassifier\n\nExplainability: SHAP (TreeExplainer)"
)


# ---------------------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------------------
def render_home():
    st.markdown(
        '<div class="main-title">Explainable AI Prediction System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">A loan-default risk predictor with full model transparency, '
        "powered by SHAP.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-card"><h3>🎯</h3><b>Binary Classification</b>'
            "<br>Predicts loan default risk</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><h3>🌲</h3><b>Random Forest</b>'
            "<br>Robust, high-accuracy ensemble model</div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><h3>🔍</h3><b>SHAP Explainability</b>'
            "<br>Transparent, per-prediction reasoning</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    st.subheader("📌 Project Overview")
    st.write(
        """
        This dashboard demonstrates how **Explainable AI (XAI)** techniques can make
        "black-box" machine learning models transparent and trustworthy. A
        `RandomForestClassifier` is trained on a synthetic loan-application dataset to
        predict whether an applicant is **likely to default** or is **low risk**.

        Instead of just returning a prediction, this system uses **SHAP (SHapley
        Additive exPlanations)** to break down *exactly* which features pushed the
        prediction toward each outcome, and by how much — giving stakeholders a clear,
        human-readable justification for every decision the model makes.
        """
    )

    st.subheader("🛠️ Technologies Used")
    tech_cols = st.columns(6)
    techs = [
        ("Python", "🐍"),
        ("Streamlit", "🎈"),
        ("Scikit-learn", "🔬"),
        ("Pandas", "🐼"),
        ("NumPy", "🔢"),
        ("SHAP", "🧩"),
    ]
    for col, (name, icon) in zip(tech_cols, techs):
        with col:
            st.markdown(
                f'<div class="metric-card">{icon}<br><b>{name}</b></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    st.info(
        "👈 Use the sidebar to open **Predict & Explain**, enter applicant details, "
        "and see a full breakdown of the model's reasoning."
    )


# ---------------------------------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------------------------------
def render_predict():
    st.markdown(
        '<div class="main-title">🔮 Predict & Explain</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Adjust the applicant profile below, then click Predict.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    # -- Input form -----------------------------------------------------
    with st.container():
        st.subheader("👤 Applicant Profile")
        input_values = {}

        col1, col2 = st.columns(2)
        feature_items = list(FEATURE_CONFIG.items())
        half = len(feature_items) // 2 + len(feature_items) % 2

        for i, (name, cfg) in enumerate(feature_items):
            target_col = col1 if i < half else col2
            with target_col:
                input_values[name] = st.slider(
                    cfg["label"],
                    min_value=float(cfg["min"]),
                    max_value=float(cfg["max"]),
                    value=float(cfg["default"]),
                    step=float(cfg["step"]),
                    key=f"slider_{name}",
                )

        st.write("")
        predict_clicked = st.button("🚀 Predict", type="primary", width='stretch')

    if not predict_clicked:
        st.caption("Set the applicant's values above and click **Predict** to see results.")
        return

    # -- Build input row --------------------------------------------------
    X_row = pd.DataFrame([input_values], columns=FEATURE_NAMES)

    # -- Prediction ---------------------------------------------------------
    proba = model.predict_proba(X_row)[0]
    pred_class = int(model.predict(X_row)[0])
    default_proba = proba[1]
    confidence = max(proba)

    st.write("")
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    res_col1, res_col2, res_col3 = st.columns(3)
    card_class = "result-card-risk" if pred_class == 1 else "result-card-safe"
    icon = "⚠️" if pred_class == 1 else "✅"

    with res_col1:
        st.markdown(
            f'<div class="{card_class}"><h4>{icon} Prediction</h4>'
            f"<h2>{CLASS_LABELS[pred_class]}</h2></div>",
            unsafe_allow_html=True,
        )
    with res_col2:
        st.markdown(
            f'<div class="metric-card"><h4>📈 Default Probability</h4>'
            f"<h2>{default_proba * 100:.1f}%</h2></div>",
            unsafe_allow_html=True,
        )
    with res_col3:
        st.markdown(
            f'<div class="metric-card"><h4>🎯 Model Confidence</h4>'
            f"<h2>{confidence * 100:.1f}%</h2></div>",
            unsafe_allow_html=True,
        )

    # -- SHAP explanation ---------------------------------------------------
    st.write("")
    st.markdown("---")
    st.subheader("🧩 Why did the model predict this?")

    with st.spinner("Computing SHAP explanations..."):
        explanation = explain_instance(explainer, X_row)
        contribution_table = get_contribution_table(explanation)
        top_contributors = get_top_contributors(explanation, top_n=3)

    nl_explanation = build_natural_language_explanation(top_contributors)
    st.markdown(
        f'<div class="explanation-box">💬 <b>Explanation:</b> {nl_explanation}</div>',
        unsafe_allow_html=True,
    )

    st.write("")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "💧 Waterfall Plot",
            "⚡ Force Plot",
            "📊 Feature Importance",
            "🌐 Summary Plot (Global)",
            "📋 Contribution Table",
        ]
    )

    with tab1:
        st.caption(
            "Shows how each feature pushes the prediction away from the base "
            "(average) value, toward the final predicted probability."
        )
        fig = plot_waterfall(explanation)
        st.pyplot(fig, width='stretch')

    with tab2:
        st.caption(
            "A compact view of the same push/pull effect — red features push toward "
            "'Likely Default', blue features push toward 'Low Risk'."
        )
        try:
            fig = plot_force(explanation)
            st.pyplot(fig, width='stretch')
        except Exception as e:
            st.warning(f"Force plot could not be rendered in this environment: {e}")

    with tab3:
        st.caption("Ranks features by the magnitude of their impact on this specific prediction.")
        fig = plot_feature_importance_bar(explanation)
        st.pyplot(fig, width='stretch')

    with tab4:
        st.caption(
            "A global view: shows how each feature affects predictions across many "
            "sample applicants (not just this one), revealing overall trends the "
            "model has learned."
        )
        fig = plot_summary(model, background_data)
        st.pyplot(fig, width='stretch')

    with tab5:
        st.caption("Exact SHAP contribution values for every feature in this prediction.")
        display_table = contribution_table.copy()
        display_table["Value"] = [
            get_feature_display_value(row.Feature, row.Value)
            for row in display_table.itertuples()
        ]
        display_table["SHAP Contribution"] = display_table["SHAP Contribution"].round(4)
        st.dataframe(display_table, width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == "🏠 Home":
    render_home()
else:
    render_predict()
