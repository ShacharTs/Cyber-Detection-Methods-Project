# predict_and_evaluate.py
import os
import json
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# =========================
# Paths
# =========================
MODEL_PATH = os.path.join("artifacts", "xgboost_model.pkl")
FEATURES_PATH = os.path.join("artifacts", "features.json")

FEATURES_CSV = os.path.join("output", "npm_val_without_labels.csv")
LABELS_CSV   = os.path.join("output", "npm_val_with_label.csv")

OUT_DIR = "evaluation_results"
OUT_CSV = os.path.join(OUT_DIR, "val_predictions_with_metrics.csv")

LABEL_CANDIDATES = {
    "label",
    "true_label",
    "ground_truth",
    "is_malware",
    "y"
}


def main():
    # -------------------------
    # Sanity checks
    # -------------------------
    for path in [MODEL_PATH, FEATURES_PATH, FEATURES_CSV, LABELS_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # -------------------------
    # Load artifacts
    # -------------------------
    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r") as f:
        FEATURES = json.load(f)

    # -------------------------
    # Load data
    # -------------------------
    X_df = pd.read_csv(FEATURES_CSV)
    y_df = pd.read_csv(LABELS_CSV)

    # -------------------------
    # Detect label column
    # -------------------------
    label_cols = [c for c in y_df.columns if c in LABEL_CANDIDATES]
    if len(label_cols) != 1:
        raise ValueError(
            f"Expected exactly one label column, found: {label_cols}"
        )

    label_col = label_cols[0]

    # -------------------------
    # Feature validation
    # -------------------------
    missing = [f for f in FEATURES if f not in X_df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    if len(X_df) != len(y_df):
        raise ValueError("Row count mismatch between features and labels")

    X = X_df[FEATURES]
    y_true = y_df[label_col].astype(int)

    # -------------------------
    # Predict
    # -------------------------
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # -------------------------
    # Metrics
    # -------------------------
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print("\n=== VALIDATION RESULTS ===\n")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nAccuracy : {acc * 100:.2f}%")
    print(f"ROC-AUC  : {auc:.4f}")

    # -------------------------
    # Save combined output
    # -------------------------
    out_df = X_df.copy()
    out_df["true_label"] = y_true
    out_df["predicted_label"] = y_pred
    out_df["malware_probability"] = y_prob

    out_df.to_csv(OUT_CSV, index=False)

    print(f"\n[+] Full evaluation output saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
