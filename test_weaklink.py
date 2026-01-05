# test_weaklink.py
import os
import json
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report


TEST_CSV = os.path.join("output", "npm_test.csv")
ARTIFACT_DIR = "artifacts"

MODELS = {
    "XGBoost": os.path.join(ARTIFACT_DIR, "weaklink_xgboost_model.pkl"),
    "RandomForest": os.path.join(ARTIFACT_DIR, "weaklink_randomforest_model.pkl"),
}

FEATURES_PATH = os.path.join(ARTIFACT_DIR, "weaklink_features.json")


def main():
    df = pd.read_csv(TEST_CSV)
    y_true = df["label"]

    with open(FEATURES_PATH) as f:
        features = json.load(f)

    X = df[features]

    for name, path in MODELS.items():
        if not os.path.exists(path):
            continue

        model = joblib.load(path)
        y_pred = model.predict(X)

        print(f"\n==============================")
        print(f"Test results: {name}")
        print(f"==============================")
        print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
        print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
