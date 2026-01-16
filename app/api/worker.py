import os
import json
from pathlib import Path
from celery import Celery
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# Initialize Celery
celery_app = Celery(
    "prediction_tasks",
    broker=os.getenv("REDIS_URL", "redis://redis:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://redis:6379/0")
)

from app.api.inference import InferenceEngine
from app.api.processor import Processor

processor = Processor()
engine = InferenceEngine()
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/shared/results"))


@celery_app.task(name="process_prediction_task")
def process_prediction_task(csv_content: bytes, job_id: str, strategy: str = "combined",
                            original_filename: str = "data.csv"):
    # Load raw data
    df_raw = processor.csv_bytes_to_raw_df(csv_content)

    # Run inference across models
    all_preds = engine.predict_all(df_raw, processor)

    metrics_summary = {}
    prediction_stats = {}
    has_ground_truth = "label" in df_raw.columns

    # Performance calculations
    for name, preds in all_preds.items():
        mal_count = int(sum(preds))
        prediction_stats[name] = {
            "malicious": mal_count,
            "benign": len(preds) - mal_count
        }

        if has_ground_truth:
            y_true = df_raw["label"].astype(float).astype(int)
            y_pred = np.array(preds).astype(int)

            # Precision, Recall, and F1 calculation
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', pos_label=1, zero_division=0
            )

            metrics_summary[name] = {
                "accuracy": float((y_true == y_pred).mean()),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1)
            }

    # Pie Chart Logic (Who detected what)
    w_preds = np.array(all_preds['weaklink'])
    d_preds = np.array(all_preds['donpai'])

    prediction_stats["overlap_analysis"] = {
        "Both Detected": int(sum((w_preds == 1) & (d_preds == 1))),
        "WeakLink Only": int(sum((w_preds == 1) & (d_preds == 0))),
        "DonPai Only": int(sum((w_preds == 0) & (d_preds == 1))),
        "Total Benign": int(sum((w_preds == 0) & (d_preds == 0)))
    }

    # Feature Activity (Total count of active behaviors > 0)
    exclude = ["label", "package_name", "Package Name", "Ground Truth", "temp_pred"]
    feature_cols = [c for c in df_raw.columns if c not in exclude]
    df_raw['temp_pred'] = all_preds['combined']

    malicious_df = df_raw[df_raw['temp_pred'] == 1]
    benign_df = df_raw[df_raw['temp_pred'] == 0]

    prediction_stats["feature_analysis"] = {
        "mal_count": (malicious_df[feature_cols] > 0).sum().to_dict(),
        "ben_count": (benign_df[feature_cols] > 0).sum().to_dict()
    }

    # Save results
    meta_path = RESULTS_DIR / f"{job_id}.json"
    with open(meta_path, "w") as f:
        json.dump({
            "status": "completed",
            "has_ground_truth": has_ground_truth,
            "all_metrics": metrics_summary,
            "prediction_stats": prediction_stats,
            "job_id": job_id,
            "original_filename": original_filename
        }, f)

    df_raw.to_csv(RESULTS_DIR / f"{job_id}.csv", index=False)