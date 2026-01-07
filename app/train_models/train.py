import os
import json
import joblib
import pandas as pd
from pathlib import Path

from app.features.model_features import ALL_FEATURES
from app.models.model import build_xgboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# =========================
# Paths (FIXED)
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"

INPUT_CSV = DATA_DIR / "npm_train.csv"


def main():
    # -------------------------
    # Load data
    # -------------------------
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    X = df[ALL_FEATURES]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # =========================
    # Model: XGBoost (FINAL)
    # =========================
    print("\n=== Model: XGBoost (FINAL) ===")
    model = build_xgboost()

    model.fit(X_train, y_train)

    # -------------------------
    # Validation evaluation
    # -------------------------
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    print(classification_report(y_val, y_pred))
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"ROC-AUC : {auc:.4f}")

    # -------------------------
    # Save artifacts
    # -------------------------
    joblib.dump(model, ARTIFACT_DIR / "xgboost_model.pkl")

    with open(ARTIFACT_DIR / "features.json", "w") as f:
        json.dump(ALL_FEATURES, f, indent=2)

    print("\n[+] Model and features saved to artifacts/")


if __name__ == "__main__":
    main()
