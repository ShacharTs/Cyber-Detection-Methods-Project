import os
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import features and model builder
from app.features.model_features import (
    WEAKLINK_FEATURES_LIST,
    DONAPI_FEATURES_LIST,
    BUNI_FEATURES_LIST,
    INSTALL_FEATURES_LIST,
    ALL_FEATURES
)
from app.models.model import build_xgboost

# Paths setup
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
INPUT_CSV = DATA_DIR / "npm_train.csv"

# Research Strategies
WEAKLINK_STRATEGY = WEAKLINK_FEATURES_LIST
DONPAI_STRATEGY = DONAPI_FEATURES_LIST + BUNI_FEATURES_LIST + INSTALL_FEATURES_LIST


def train_and_save(name, features, X_train, X_val, y_train, y_val):
    print(f"\n--- Training {name.upper()} Model ---")
    model = build_xgboost()
    model.fit(X_train[features], y_train)

    # Save model and its specific feature list for the processor
    joblib.dump(model, ARTIFACT_DIR / f"{name}_model.pkl")
    with open(ARTIFACT_DIR / f"{name}_features.json", "w") as f:
        json.dump(features, f, indent=2)

    acc = accuracy_score(y_val, model.predict(X_val[features]))
    print(f"Done! {name} Accuracy: {acc * 100:.2f}%")
    return acc


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"CSV not found at {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    y = df["label"]
    X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.2, random_state=42)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Train the 3 variants to see the Gap
    train_and_save("weaklink", WEAKLINK_STRATEGY, X_train, X_val, y_train, y_val)
    train_and_save("donpai", DONPAI_STRATEGY, X_train, X_val, y_train, y_val)
    train_and_save("combined", ALL_FEATURES, X_train, X_val, y_train, y_val)


if __name__ == "__main__":
    main()