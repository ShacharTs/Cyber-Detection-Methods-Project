import os
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from app.features.model_features import (
    WEAKLINK_FEATURES_LIST,
    BUNI_FEATURES_LIST,
    INSTALL_FEATURES_LIST,
    ALL_FEATURES
)
from app.models.model import (
    build_xgboost,
    build_random_forest,
    build_voting_classifier
)

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
INPUT_CSV = DATA_DIR / "npm_train.csv"

# =========================
# Feature strategies
# =========================
WEAKLINK_STRATEGY = sorted(set(WEAKLINK_FEATURES_LIST))

DONPAI_STRATEGY = sorted(set(
    BUNI_FEATURES_LIST +
    INSTALL_FEATURES_LIST
))

HYBRID_STRATEGY = sorted(set(ALL_FEATURES))


def train_and_save(name, model_type, features, X_train, X_val, y_train, y_val):
    print("\n" + "=" * 50)
    print(f"TRAINING {name.upper()} STRATEGY")
    print(f"Model Type: {model_type.upper()}")
    print(f"Feature Count: {len(features)}")
    print("=" * 50)

    model_type = model_type.lower()

    if model_type == "rf":
        model = build_random_forest()

    elif model_type == "xgb":
        model = build_xgboost()

    elif model_type == "voting":
        model = build_voting_classifier(
            voting="soft",
            weights=[2, 1]  # XGB > RF
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train
    model.fit(X_train[features], y_train)

    # Save model
    model_path = ARTIFACT_DIR / f"{name}_{model_type}_model.pkl"
    joblib.dump(model, model_path)

    # Save features
    with open(ARTIFACT_DIR / f"{name}_features.json", "w") as f:
        json.dump(features, f, indent=2)

    # Eval
    preds = model.predict(X_val[features])
    acc = accuracy_score(y_val, preds)

    print(f"SUCCESS: {name} training complete.")
    print(f"Validation Accuracy: {acc * 100:.2f}%")

    return acc


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing dataset at: {INPUT_CSV}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        df,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 1. WeakLink – XGBoost
    train_and_save(
        "weaklink",
        "xgb",
        WEAKLINK_STRATEGY,
        X_train,
        X_val,
        y_train,
        y_val
    )

    # 2. DonPai – Random Forest
    train_and_save(
        "donpai",
        "rf",
        DONPAI_STRATEGY,
        X_train,
        X_val,
        y_train,
        y_val
    )

    # 3. Combined – XGBoost
    train_and_save(
        "combined",
        "xgb",
        HYBRID_STRATEGY,
        X_train,
        X_val,
        y_train,
        y_val
    )

    # 4. Combined Voting – XGB + RF
    train_and_save(
        "combined_voting",
        "voting",
        HYBRID_STRATEGY,
        X_train,
        X_val,
        y_train,
        y_val
    )

    print("\n" + "=" * 50)
    print("ALL MODELS TRAINED AND SAVED TO /artifacts")
    print("=" * 50)


if __name__ == "__main__":
    main()
