import os
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import features and both model builders
from app.features.model_features import (
    WEAKLINK_FEATURES_LIST,
    DONAPI_FEATURES_LIST,
    BUNI_FEATURES_LIST,
    INSTALL_FEATURES_LIST,
    STATIC_FEATURES_LIST,
    ALL_FEATURES
)
from app.models.model import build_xgboost, build_random_forest

# Paths setup
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
INPUT_CSV = DATA_DIR / "npm_train.csv"


# Weaklink Strategy: Focused on supply chain metadata (W2, W3, W4, W5)
WEAKLINK_STRATEGY = sorted(list(set(WEAKLINK_FEATURES_LIST)))

# DonPai Strategy: Focused on behavioral sequences and static analysis patterns
DONPAI_STRATEGY = sorted(list(set(
    DONAPI_FEATURES_LIST +
    BUNI_FEATURES_LIST +
    INSTALL_FEATURES_LIST +
    STATIC_FEATURES_LIST
)))

# Hybrid Strategy: Our Innovation - Combining all features for maximum detection
HYBRID_STRATEGY = sorted(list(set(ALL_FEATURES)))


def train_and_save(name, model_type, features, X_train, X_val, y_train, y_val):
    """
    Trains a model based on the selected algorithm, saves artifacts, and prints metrics.
    """
    print(f"\n" + "=" * 50)
    print(f"TRAINING {name.upper()} STRATEGY")
    print(f"Model Type: {model_type.upper()}")
    print(f"Feature Count: {len(features)}")
    print("=" * 50)

    # Build model
    if model_type.lower() == "rf":
        model = build_random_forest()
    else:
        model = build_xgboost()

    # Train model
    model.fit(X_train[features], y_train)

    # Save Model Artifact
    model_filename = f"{name}_{model_type}_model.pkl"
    joblib.dump(model, ARTIFACT_DIR / model_filename)

    # Save Feature List (for inference later)
    with open(ARTIFACT_DIR / f"{name}_features.json", "w") as f:
        json.dump(features, f, indent=2)

    # Evaluation
    predictions = model.predict(X_val[features])
    acc = accuracy_score(y_val, predictions)

    print(f"SUCCESS: {name} training complete.")
    print(f"Accuracy: {acc * 100:.2f}%")

    # Optional: detailed report for your documentation
    # print(classification_report(y_val, predictions))

    return acc


def main():
    # Ensure environment is ready
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing dataset at: {INPUT_CSV}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Data
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    y = df["label"]

    # Split Data (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. WEAKLINK: Based on npm metadata study [cite: 18, 72]
    # Uses XGBoost to capture non-linear metadata relationships
    train_and_save("weaklink", "xgb", WEAKLINK_STRATEGY, X_train, X_val, y_train, y_val)

    # 2. DONPAI: Behavioral & Static analysis
    # Uses Random Forest to match traditional behavioral detection approaches
    train_and_save("donpai", "rf", DONPAI_STRATEGY, X_train, X_val, y_train, y_val)

    # 3. HYBRID (Ours): The full feature set innovation
    # Combines all signals into a single high-performance XGBoost model
    train_and_save("combined", "xgb", HYBRID_STRATEGY, X_train, X_val, y_train, y_val)

    print("\n" + "=" * 50)
    print("ALL MODELS TRAINED AND SAVED TO /ARTIFACTS")
    print("=" * 50)


if __name__ == "__main__":
    main()