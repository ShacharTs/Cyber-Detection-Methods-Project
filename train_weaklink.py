import os
import json
import joblib
import pandas as pd
import xgboost as xgb

from features import WEAKLINK_FEATURES_LIST
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


# =========================================================
# Paths
# =========================================================
INPUT_CSV = os.path.join("output", "npm_train.csv")
ARTIFACT_DIR = "artifacts"

MODEL_PATH = os.path.join(ARTIFACT_DIR, "weaklink_xgb_model.pkl")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "weaklink_features.json")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "weaklink_metrics.json")


# =========================================================
# Train
# =========================================================
def train():
    # --- sanity ---
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Dataset not found: {INPUT_CSV}")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # --- load data ---
    df = pd.read_csv(INPUT_CSV)

    X = df[WEAKLINK_FEATURES_LIST]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # --- model ---
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

    # --- train ---
    model.fit(X_train, y_train)

    # --- evaluate ---
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "roc_auc": roc_auc,
        "classification_report": report
    }

    # --- save artifacts ---
    joblib.dump(model, MODEL_PATH)

    with open(FEATURES_PATH, "w") as f:
        json.dump(WEAKLINK_FEATURES_LIST, f, indent=2)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== Training complete ===")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    train()
