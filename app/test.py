# test_all_models.py
import os
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

# =========================
# Paths
# =========================
TEST_CSV = os.path.join("../data", "npm_test.csv")
ARTIFACT_DIR = "../artifacts"

MODEL_PATH = os.path.join(ARTIFACT_DIR, "xgboost_model.pkl")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "features.json")


def main():
    # -------------------------
    # Load test data
    # -------------------------
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Missing test CSV: {TEST_CSV}")

    df = pd.read_csv(TEST_CSV)

    if "label" not in df.columns:
        raise ValueError("Test CSV must contain 'label' column")

    y_true = df["label"]

    # -------------------------
    # Load feature list
    # -------------------------
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")

    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)

    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features in test CSV: {missing}")

    X = df[features]

    # -------------------------
    # Load model
    # -------------------------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    # -------------------------
    # Test evaluation
    # -------------------------
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print("\n" + "#" * 40)
    print(" FINAL MODEL (Test Set)")
    print("#" * 40)
    print("Model   : XGBoost")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"ROC-AUC : {auc:.4f}")
    print("#" * 40)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
