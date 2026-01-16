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
BASE_DIR = os.path.dirname(__file__)
TEST_CSV = os.path.join(BASE_DIR, "../data/npm_test.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "../artifacts")

# =========================
# Models to evaluate
# =========================
MODELS = {
    "weaklink": "weaklink_xgb_model.pkl",
    "donpai": "donpai_rf_model.pkl",
    "combined_voting": "combined_voting_voting_model.pkl"
}


def evaluate_model(name, model, X, y_true):
    y_pred = model.predict(X)

    auc = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y_true, y_prob)

    acc = accuracy_score(y_true, y_pred)

    print("\n" + "#" * 60)
    print(f"MODEL: {name.upper()}")
    print("#" * 60)
    print(f"Accuracy: {acc * 100:.2f}%")
    if auc is not None:
        print(f"ROC-AUC : {auc:.4f}")
    print("#" * 60)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


def main():
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Missing test CSV: {TEST_CSV}")

    df = pd.read_csv(TEST_CSV)

    if "label" not in df.columns:
        raise ValueError("Test CSV must contain 'label' column")

    y_true = df["label"].astype(int)

    for name, model_file in MODELS.items():
        model_path = os.path.join(ARTIFACT_DIR, model_file)
        features_path = os.path.join(ARTIFACT_DIR, f"{name}_features.json")

        if not os.path.exists(model_path):
            print(f"[SKIP] Missing model: {model_file}")
            continue

        if not os.path.exists(features_path):
            print(f"[SKIP] Missing features file for {name}")
            continue

        with open(features_path, "r") as f:
            features = json.load(f)

        missing = set(features) - set(df.columns)
        if missing:
            raise ValueError(f"[{name}] Missing features in test CSV: {missing}")

        X = df[features]
        model = joblib.load(model_path)

        evaluate_model(name, model, X, y_true)


if __name__ == "__main__":
    main()
