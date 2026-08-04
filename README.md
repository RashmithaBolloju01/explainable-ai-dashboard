# 🧠 Explainable AI Prediction System (XAI Dashboard)

An interactive **Streamlit dashboard** that predicts loan-default risk with a
`RandomForestClassifier` and explains *every single prediction* using
**SHAP (SHapley Additive exPlanations)** — turning a black-box model into a
transparent, interview-ready demo of Explainable AI in practice.

> 🔗 No external datasets required — a realistic loan-application dataset is
> generated programmatically with `sklearn.datasets.make_classification`, so
> the whole project runs out of the box.

---

## 📌 Project Overview

Machine learning models are often criticized for being "black boxes." This
project demonstrates how **Explainable AI (XAI)** techniques can address that
problem: a Random Forest is trained to predict whether a loan applicant is
**likely to default** or is **low risk**, and every prediction is paired with
a SHAP-based breakdown of *why* the model reached that conclusion — including
plain-English summaries, waterfall/force/summary plots, and a per-feature
contribution table.

**Note on the data:** the dataset is synthetically generated with
`make_classification` and then rescaled into human-readable business
features (income, credit score, etc.) purely to make the demo relatable.
The statistical relationships between features and the target are
therefore illustrative, not a reflection of real-world credit-risk
modeling — the focus of this project is the **explainability pipeline**,
not financial accuracy.

---

## ✨ Features

- **🏠 Home page** — project overview, tech stack, and quick orientation.
- **🔮 Prediction page**
  - Interactive sliders for 6 applicant features (income, debt ratio, credit
    score, loan amount, employment years, age).
  - One-click prediction with **class label**, **default probability**, and
    **model confidence**.
- **🧩 Explainable AI panel** (generated after every prediction)
  - 💧 **Waterfall plot** — how each feature pushes the prediction from the
    baseline to the final result.
  - ⚡ **Force plot** — compact visual of the same push/pull effect.
  - 📊 **Feature importance bar chart** — ranks features by impact for this
    prediction.
  - 🌐 **SHAP summary (beeswarm) plot** — global feature impact across many
    sample applicants.
  - 📋 **Contribution table** — exact feature values + SHAP contributions.
  - 💬 **Natural-language explanation**, e.g. *"The prediction is mainly
    influenced by high debt-to-income ratio and low credit score."*
- **🎨 Professional UI** — wide layout, sidebar navigation, card-style
  metrics, tabs, and clean spacing designed to look good in interviews/demos.
- **⚙️ Auto-training** — if no saved model is found, the app trains one
  automatically on first launch and reports test accuracy.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core language |
| **Streamlit** | Interactive web dashboard |
| **Scikit-learn** | RandomForestClassifier + synthetic dataset generation |
| **Pandas / NumPy** | Data manipulation |
| **Matplotlib** | Rendering SHAP plots |
| **SHAP** | Model explainability (TreeExplainer) |

---

## 📂 Project Structure

```
Explainable-AI-Dashboard/
│
├── app.py               # Streamlit dashboard (UI, navigation, layout)
├── model.py              # Dataset generation, training, save/load helpers
├── train.py              # Standalone script: train + save model manually
├── explain.py             # All SHAP logic (explainer, plots, contribution table)
├── utils.py               # Feature config, formatting, NL explanation builder
├── requirements.txt      # Python dependencies
└── README.md              # This file
```

---

## 🚀 Installation

1. **Clone or download** this project folder.
2. (Recommended) create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run

Simply launch the Streamlit app:

```bash
streamlit run app.py
```

- On the **first run**, no `model.pkl` exists yet, so the app will
  automatically generate the dataset, train the Random Forest, save the
  model, and display the test accuracy in the sidebar.
- On subsequent runs, the saved model is loaded instantly.

If you'd prefer to train the model manually beforehand (e.g. to inspect
accuracy in the terminal first), run:

```bash
python train.py
```

Then launch the dashboard as usual with `streamlit run app.py`.

---

## 🖼️ Screenshots

> _Add screenshots here after running the app locally, e.g.:_

| Home Page | Prediction + SHAP Explanation |
|---|---|
| `screenshots/home.png` | `screenshots/predict.png` |

---

## 🔮 Future Improvements

- Add support for uploading a custom CSV dataset instead of the synthetic one.
- Add model comparison (Random Forest vs. Logistic Regression vs. XGBoost)
  with SHAP explanations for each.
- Persist prediction history and let users compare past explanations.
- Add SHAP dependence plots for deeper feature-interaction analysis.
- Deploy to Streamlit Community Cloud / Docker for a live public demo link.
- Add authentication + a database backend for multi-user usage tracking.

---

## 📄 License

This project is open-source and free to use for learning, portfolio, and
demonstration purposes.
