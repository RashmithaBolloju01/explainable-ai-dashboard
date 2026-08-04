"""
model.py
--------
Handles dataset generation, model training, saving and loading.

The model is a RandomForestClassifier trained on a synthetic loan-default
dataset created with sklearn.datasets.make_classification, so the whole
project runs with zero external data downloads.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils import FEATURE_NAMES, MODEL_PATH, DATA_PATH, scale_raw_features


def generate_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic, business-friendly loan-default dataset.

    Uses sklearn's make_classification to create a realistic binary
    classification problem, then rescales the raw features into
    interpretable ranges (income, debt ratio, credit score, etc.) using
    utils.scale_raw_features.

    Returns
    -------
    pd.DataFrame with feature columns + a 'Default' target column
    (1 = likely default, 0 = low risk).
    """
    X_raw, y = make_classification(
        n_samples=n_samples,
        n_features=len(FEATURE_NAMES),
        n_informative=5,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[0.65, 0.35],  # slightly imbalanced, like real loan data
        flip_y=0.03,           # a little label noise for realism
        class_sep=1.2,
        random_state=random_state,
    )

    X_scaled = scale_raw_features(X_raw)

    df = pd.DataFrame(X_scaled, columns=FEATURE_NAMES)
    df["Default"] = y
    return df


def train_model(df: pd.DataFrame = None, random_state: int = 42):
    """
    Train a RandomForestClassifier on the (optionally provided) dataset.

    Returns
    -------
    model : trained RandomForestClassifier
    accuracy : float, accuracy on a held-out test split
    X_test, y_test : the held-out test data (useful for demos / debugging)
    """
    if df is None:
        df = generate_dataset(random_state=random_state)

    X = df[FEATURE_NAMES]
    y = df["Default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, X_test, y_test


def save_model(model, path: str = MODEL_PATH):
    """Persist the trained model to disk with pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str = MODEL_PATH):
    """Load a previously trained model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_or_train_model(path: str = MODEL_PATH):
    """
    Convenience function used by app.py:
    - If a saved model already exists on disk, load it.
    - Otherwise, generate the dataset, train a fresh model, save it, and
      return it along with the reported test accuracy.

    Returns
    -------
    model : trained RandomForestClassifier
    accuracy : float or None (None if the model was simply loaded from disk
               and accuracy wasn't recomputed)
    """
    if os.path.exists(path):
        model = load_model(path)
        accuracy = None
        # Try to recover a previously saved accuracy score, if present.
        acc_path = path.replace(".pkl", "_accuracy.txt")
        if os.path.exists(acc_path):
            with open(acc_path, "r") as f:
                accuracy = float(f.read().strip())
        return model, accuracy

    # No saved model found -> train a new one from scratch.
    df = generate_dataset()
    model, accuracy, _, _ = train_model(df)
    save_model(model, path)

    # Save the dataset and accuracy for reference / reproducibility.
    df.to_csv(DATA_PATH, index=False)
    with open(path.replace(".pkl", "_accuracy.txt"), "w") as f:
        f.write(str(accuracy))

    return model, accuracy


if __name__ == "__main__":
    # Allow `python model.py` to behave like a quick sanity check.
    m, acc = load_or_train_model()
    print(f"Model ready. Test accuracy: {acc}")
