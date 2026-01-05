import os
import json
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# Paths
# =========================================================
TEST_CSV = os.path.join("output", "npm_test.csv")
ARTIFACT_DIR = "artifacts"

MODEL_PATH = os.path.join(ARTIFACT_DIR, "weaklink_xgb_model.pkl")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "weaklink_features.json")


# =========================================================
# Evaluate
# =========================================================
def evaluate():
    # --- sanity checks ---
    for path in [TEST_CSV, MODEL_PATH, FEATURES_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    # --- load data ---
    df = pd.read_csv(TEST_CSV)

    if "label" not in df.columns:
        raise ValueError("Test CSV does not contain 'label' column")

    y_true = df["label"]

    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)

    X = df[features]

    model = joblib.load(MODEL_PATH)

    # --- predict ---
    y_pred = model.predict(X)

    # --- metrics ---
    acc = accuracy_score(y_true, y_pred)

    print("=== Weak-Link Model Evaluation ===")
    print(f"Accuracy: {acc * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    evaluate()
