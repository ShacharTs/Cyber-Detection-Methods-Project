#!/usr/bin/env python3
"""
train_model.py

Train and evaluate a model from features.csv.
Writes evaluation results to results_training.txt.

Usage:
  python train_model.py --csv features.csv --model rf_donapi_lite.joblib --filter drop_metadata
  python train_model.py --csv features.csv --drop-empty-js --drop-empty-code
"""

import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


METADATA_COLS = [
    "num_dependencies",
    "num_maintainers",
    "has_scripts",
    "repository_exists",
    "description_length",
    "num_js_files",
]

OBF_COLS = [
    "line_ratio", "space_ratio", "num_long_strings", "max_string_len",
    "special_char_count", "substring_calls", "identifier_entropy",
    "avg_identifier_length", "num_lines",
]

BEHAVIOR_COUNT_PREFIX = "bf_"
BEHAVIOR_NGRAM_PREFIXES = ("buni_", "bbi_", "btri_")


def select_features(df: pd.DataFrame, mode: str):
    drop_always = {"path", "label"}
    cols = [c for c in df.columns if c not in drop_always]

    if mode == "none":
        return cols

    if mode == "drop_metadata":
        return [c for c in cols if c not in METADATA_COLS]

    if mode == "behavior_only":
        keep = set()
        keep.update([c for c in OBF_COLS if c in df.columns])
        keep.update([c for c in df.columns if c.startswith(BEHAVIOR_COUNT_PREFIX)])
        keep.update([c for c in df.columns if c.startswith(BEHAVIOR_NGRAM_PREFIXES)])
        return [c for c in cols if c in keep]

    raise ValueError(f"Unknown filter mode: {mode}")


def apply_row_filters(df: pd.DataFrame, drop_empty_js: bool, drop_empty_code: bool):
    """
    Row filtering to reduce junk rows / extraction failures.

    --drop-empty-js:
        Drops rows with num_js_files == 0 (if column exists)

    --drop-empty-code:
        Drops rows where all code-derived columns are zero:
        obfuscation + behavior counts + behavior ngrams
    """
    before = len(df)

    if drop_empty_js and "num_js_files" in df.columns:
        df = df[df["num_js_files"].fillna(0).astype(float) > 0].copy()

    if drop_empty_code:
        code_cols = []

        # Obfuscation columns (if present)
        code_cols.extend([c for c in OBF_COLS if c in df.columns])

        # Behavior counts
        code_cols.extend([c for c in df.columns if c.startswith(BEHAVIOR_COUNT_PREFIX)])

        # Behavior ngram features
        code_cols.extend([c for c in df.columns if c.startswith(BEHAVIOR_NGRAM_PREFIXES)])

        # Deduplicate list while preserving order
        seen = set()
        code_cols = [c for c in code_cols if not (c in seen or seen.add(c))]

        if code_cols:
            # Keep rows where at least one code-derived feature is non-zero
            sums = df[code_cols].fillna(0).astype(float).sum(axis=1)
            df = df[sums > 0].copy()
        # If no code_cols exist, we can’t apply this filter safely; do nothing.

    after = len(df)
    dropped = before - after
    return df, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="rf_donapi_lite.joblib")
    ap.add_argument("--filter", default="none",
                    choices=["none", "drop_metadata", "behavior_only"])
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-report", default="results_training.txt")

    # ✅ NEW FLAGS
    ap.add_argument("--drop-empty-js", action="store_true",
                    help="Drop rows where num_js_files == 0 (if column exists)")
    ap.add_argument("--drop-empty-code", action="store_true",
                    help="Drop rows where all code-derived features are zero")

    args = ap.parse_args()

    log_lines = []

    df = pd.read_csv(args.csv)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    # ✅ Apply row filters BEFORE feature selection / split
    df, dropped = apply_row_filters(df, args.drop_empty_js, args.drop_empty_code)
    log_lines.append(f"Row filters: drop_empty_js={args.drop_empty_js}, drop_empty_code={args.drop_empty_code}")
    log_lines.append(f"Rows dropped by filters: {dropped}")
    log_lines.append(f"Rows remaining: {len(df)}")
    log_lines.append("")

    y = df["label"]
    if len(np.unique(y)) < 2:
        raise ValueError(f"Need at least 2 classes. Found: {np.unique(y)}")

    features = select_features(df, args.filter)
    X = df[features].fillna(0).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=args.test_size, random_state=args.seed
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=args.seed,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    log_lines.append(f"Filter mode: {args.filter}")
    log_lines.append(f"Number of features: {len(features)}")
    log_lines.append("")
    log_lines.append("Classification report:")
    log_lines.append(classification_report(y_test, y_pred, digits=4))

    if hasattr(clf, "predict_proba") and len(np.unique(y_test)) > 1:
        y_proba = clf.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, y_proba)
            log_lines.append(f"ROC-AUC: {auc:.4f}")
        except Exception:
            pass

    log_lines.append("")
    log_lines.append("Top 30 features:")
    importances = sorted(
        zip(features, clf.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )[:30]
    for name, imp in importances:
        log_lines.append(f"{name}: {imp:.6f}")

    # Save model bundle
    joblib.dump(
        {"model": clf, "features": features, "filter_mode": args.filter},
        args.model
    )

    # Write report to file
    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"[+] Training complete")
    print(f"[+] Model saved to: {args.model}")
    print(f"[+] Report written to: {args.out_report}")


if __name__ == "__main__":
    main()

