#!/usr/bin/env python3
"""
train_model_newcsv.py

Train and evaluate a RandomForest model from a CSV like npm_train.csv.

- Drops leakage-prone columns by default:
  path, days_since_last_update, inactive_package_flag, num_install_scripts, install_script_complexity

- Saves:
  1) model bundle (model + feature list) to --model-out
  2) evaluation report to --report-out

Usage:
  python train_model_newcsv.py --csv npm_train.csv --model-out rf_model.joblib --report-out results_training.txt
"""

import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


DEFAULT_DROP = {
    "path",
    "days_since_last_update",
    "inactive_package_flag",
    "num_install_scripts",
    "install_script_complexity",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Training CSV with 'label' column")
    ap.add_argument("--model-out", default="rf_model.joblib", help="Output .joblib bundle")
    ap.add_argument("--report-out", default="results_training.txt", help="Output report text file")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # optional extra drops
    ap.add_argument("--drop", nargs="*", default=[], help="Extra columns to drop (space-separated)")
    ap.add_argument("--keep-package-name", action="store_true",
                    help="Keep package_name as a feature (NOT recommended unless already numeric/encoded)")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "label" not in df.columns:
        raise ValueError("Training CSV must contain a 'label' column.")

    # Build drop set
    drop_cols = set(DEFAULT_DROP)
    drop_cols.update(args.drop)

    # By default drop package_name (often non-numeric)
    if not args.keep_package_name and "package_name" in df.columns:
        drop_cols.add("package_name")

    # Keep only existing columns (avoid KeyError if some columns not present)
    drop_cols = {c for c in drop_cols if c in df.columns}

    # Build feature list
    features = [c for c in df.columns if c not in drop_cols and c != "label"]

    if not features:
        raise ValueError("No features left after dropping columns.")

    # Prepare X/y
    X = df[features].fillna(0)

    # Ensure numeric
    try:
        X = X.astype(float)
    except Exception as e:
        bad = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
        raise ValueError(
            f"Some feature columns are not numeric: {bad}. "
            f"Drop them or encode them first."
        ) from e

    y = df["label"].astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError(f"Need at least 2 classes in label. Found: {np.unique(y)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=args.test_size, random_state=args.seed
    )

    # Train
    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)

    auc_line = ""
    if hasattr(clf, "predict_proba") and len(np.unique(y_test)) > 1:
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        auc_line = f"ROC-AUC: {auc:.4f}\n"

    # Top features
    importances = sorted(
        zip(features, clf.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )[:30]
    top_feats = "\n".join([f"{name}: {imp:.6f}" for name, imp in importances])

    # Save model bundle (model + exact feature order)
    bundle = {
        "model": clf,
        "features": features,
        "dropped_columns": sorted(list(drop_cols)),
    }
    joblib.dump(bundle, args.model_out)

    # Write report
    with open(args.report_out, "w", encoding="utf-8") as f:
        f.write(f"Training file: {args.csv}\n")
        f.write(f"Num rows: {len(df)}\n")
        f.write(f"Num features: {len(features)}\n")
        f.write(f"Dropped columns: {sorted(list(drop_cols))}\n\n")
        f.write("Classification report:\n")
        f.write(report + "\n")
        if auc_line:
            f.write("\n" + auc_line)
        f.write("\nTop 30 features:\n")
        f.write(top_feats + "\n")

    print(f"[+] Saved model bundle to: {args.model_out}")
    print(f"[+] Wrote training report to: {args.report_out}")
    print(f"[+] Dropped columns: {sorted(list(drop_cols))}")


if __name__ == "__main__":
    main()

