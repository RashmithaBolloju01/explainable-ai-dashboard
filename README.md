# Explainable AI Prediction System (XAI Dashboard)

An interactive Streamlit dashboard that predicts loan-default risk using a Random Forest classifier and explains individual predictions with SHAP (SHapley Additive exPlanations).

The project demonstrates how Explainable AI improves model transparency by showing how applicant features influence each prediction through visualizations, feature contributions, and natural-language explanations.

## Overview

The dashboard classifies loan applicants into high-risk and low-risk categories and displays the predicted default probability alongside a SHAP-based explanation.

The project uses a synthetically generated dataset, eliminating the need for external data sources and allowing the application to run locally.

**Note:** The synthetic dataset demonstrates the explainability workflow, not real-world credit-risk assessment. Its feature relationships do not represent actual lending patterns.

## Features

- **Home Page:** Project overview, technology stack, and navigation.
- **Loan Prediction:** Interactive inputs for income, debt-to-income ratio, credit score, loan amount, employment years, and age.
- **Risk Assessment:** Predicted class, default probability, and model confidence.
- **SHAP Waterfall Plot:** Shows how individual features influence a prediction.
- **SHAP Force Plot:** Visualizes positive and negative feature contributions.
- **Feature Importance:** Ranks features by their impact on an individual prediction.
- **SHAP Summary Plot:** Displays global feature importance across multiple applicants.
- **Contribution Table:** Lists feature values and their corresponding SHAP contributions.
- **Natural-Language Explanations:** Summarizes the main factors influencing each prediction.
- **Automatic Model Training:** Generates the dataset, trains the classifier, and saves the model when no saved model exists.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive dashboard |
| Scikit-learn | Dataset generation and model training |
| RandomForestClassifier | Loan-default prediction |
| SHAP | Model explainability |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Visualization rendering |

## Project Structure

```text
Explainable-AI-Dashboard/
├── app.py
├── model.py
├── train.py
├── explain.py
├── utils.py
├── requirements.txt
└── README.md
```

- `app.py` — Dashboard interface, navigation, and prediction workflow.
- `model.py` — Dataset generation, model training, and model persistence.
- `train.py` — Standalone model training script.
- `explain.py` — SHAP explanations, visualizations, and feature contributions.
- `utils.py` — Feature configuration, formatting, and natural-language explanations.
- `requirements.txt` — Project dependencies.

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Explainable-AI-Dashboard
```

Replace `<repository-url>` with your GitHub repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

Start the dashboard:

```bash
streamlit run app.py
```

On the first launch, the application generates a synthetic dataset, trains the Random Forest classifier, saves the model, and displays its test accuracy. Subsequent launches load the saved model.

To train the model separately, run:

```bash
python train.py
```

## Future Improvements

- Support custom CSV dataset uploads.
- Compare multiple classification models.
- Store prediction history and compare explanations.
- Add SHAP dependence plots for feature interactions.
- Deploy the dashboard using Streamlit Community Cloud or Docker.
- Add authentication and database integration.

## License

This project is intended for educational, portfolio, and demonstration purposes.