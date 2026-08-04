"""
train.py
--------
Standalone training script.

Run this directly with:
    python train.py

It will:
1. Generate the synthetic loan-default dataset.
2. Train a RandomForestClassifier.
3. Print the test accuracy.
4. Save the trained model to model.pkl and the dataset to training_data.csv.

Note: app.py also auto-trains the model on first run if model.pkl does not
exist yet, so running this script manually is optional but useful for
reproducing results or retraining from scratch.
"""

from model import generate_dataset, train_model, save_model
from utils import MODEL_PATH, DATA_PATH


def main():
    print("=" * 60)
    print("Explainable AI Prediction System — Model Training")
    print("=" * 60)

    print("\n[1/4] Generating synthetic loan-default dataset...")
    df = generate_dataset(n_samples=2000)
    print(f"      -> Dataset shape: {df.shape}")
    print(f"      -> Class balance:\n{df['Default'].value_counts(normalize=True)}")

    print("\n[2/4] Training RandomForestClassifier...")
    model, accuracy, X_test, y_test = train_model(df)
    print(f"      -> Done. Test accuracy: {accuracy:.4f}")

    print("\n[3/4] Saving model and accuracy score...")
    save_model(model, MODEL_PATH)
    with open(MODEL_PATH.replace(".pkl", "_accuracy.txt"), "w") as f:
        f.write(str(accuracy))
    print(f"      -> Model saved to '{MODEL_PATH}'")

    print("\n[4/4] Saving training dataset for reference...")
    df.to_csv(DATA_PATH, index=False)
    print(f"      -> Dataset saved to '{DATA_PATH}'")

    print("\nTraining complete! You can now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
