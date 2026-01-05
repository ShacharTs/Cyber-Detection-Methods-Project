# train_weaklink.py
import os
import json
import joblib
import pandas as pd

from features import WEAKLINK_FEATURES_LIST
from model import build_xgboost, build_random_forest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


INPUT_CSV = os.path.join("output", "npm_train.csv")
ARTIFACT_DIR = "artifacts"


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(INPUT_CSV)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    X = df[WEAKLINK_FEATURES_LIST]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "XGBoost": build_xgboost(),
        "RandomForest": build_random_forest(),
    }

    for name, model in models.items():
        print(f"\n==============================")
        print(f"Training model: {name}")
        print(f"==============================")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        print(classification_report(y_test, y_pred))
        print("ROC-AUC:", roc_auc_score(y_test, y_prob))

        model_path = os.path.join(
            ARTIFACT_DIR,
            f"weaklink_{name.lower()}_model.pkl"
        )
        joblib.dump(model, model_path)

    with open(os.path.join(ARTIFACT_DIR, "weaklink_features.json"), "w") as f:
        json.dump(WEAKLINK_FEATURES_LIST, f, indent=2)

    print("\n=== Training finished. All models saved. ===")


if __name__ == "__main__":
    main()
