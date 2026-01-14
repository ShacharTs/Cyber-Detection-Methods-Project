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

def train_model(X_train, y_train):
    """
    Method 1: Builds and trains the XGBoost model.
    """
    print("\n=== Training Model: XGBoost (FINAL) ===")
    model = build_xgboost()
    model.fit(X_train, y_train)
    return model

def evaluate_and_save(model, X_val, y_val):
    """
    Method 2: Evaluates the model and saves it to artifacts.
    """
    # Validation evaluation
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    print("\n--- Evaluation Report ---")
    print(classification_report(y_val, y_pred))
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"ROC-AUC : {auc:.4f}")

    # Save artifacts
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "xgboost_model.pkl")

    with open(ARTIFACT_DIR / "features.json", "w") as f:
        json.dump(ALL_FEATURES, f, indent=2)

    print("\n[+] Model and features saved to artifacts/")
    return acc

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input data at: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    X = df[ALL_FEATURES]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Call Method 1
    trained_model = train_model(X_train, y_train)

    # Call Method 2
    evaluate_and_save(trained_model, X_val, y_val)

if __name__ == "__main__":
    main()