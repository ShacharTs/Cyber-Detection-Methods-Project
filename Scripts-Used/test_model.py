#!/usr/bin/env python3
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Saved model bundle (.joblib)")
    ap.add_argument("--csv", required=True, help="Test CSV (same feature columns; label optional)")
    ap.add_argument("--out", default="predictions.csv", help="Output CSV with predictions")
    ap.add_argument("--report-out", default="results_test.txt", help="Write test metrics here if label exists")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    clf = bundle["model"]
    features = bundle["features"]

    df = pd.read_csv(args.csv)

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Test CSV is missing required columns: {missing}")

    X = df[features].fillna(0)

    y_pred = clf.predict(X)
    df_out = df.copy()
    df_out["pred_label"] = y_pred

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[:, 1]
        df_out["pred_proba_malicious"] = proba

    df_out.to_csv(args.out, index=False)
    print(f"[+] Wrote predictions to: {args.out}")

    # If labels exist, evaluate
    if "label" in df.columns:
        y_true = df["label"].astype(int)

        report = classification_report(y_true, y_pred, digits=4)
        auc_line = ""
        if hasattr(clf, "predict_proba") and len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, df_out["pred_proba_malicious"])
            auc_line = f"ROC-AUC: {auc:.4f}\n"

        with open(args.report_out, "w", encoding="utf-8") as f:
            f.write(f"Test file: {args.csv}\n")
            f.write(f"Num rows: {len(df)}\n\n")
            f.write("Classification report:\n")
            f.write(report + "\n")
            if auc_line:
                f.write("\n" + auc_line)

        print(f"[+] Wrote test report to: {args.report_out}")
    else:
        print("[i] No 'label' column found in test CSV, so no evaluation was performed.")


if __name__ == "__main__":
    main()

