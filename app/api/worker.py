import json
import os
from pathlib import Path

import numpy as np
from celery import Celery
from sklearn.metrics import precision_recall_fscore_support

# Initialize Celery
celery_app = Celery(
    "prediction_tasks",
    broker=os.getenv("REDIS_URL", "redis://redis:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://redis:6379/0")
)

from app.api.inference import InferenceEngine
from app.api.processor import Processor
from app.features.model_features import ALL_FEATURES

processor = Processor()
engine = InferenceEngine()
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/shared/results"))


@celery_app.task(name="process_prediction_task")
def process_prediction_task(
        csv_content: bytes,
        job_id: str,
        strategy: str = "combined",
        original_filename: str = "data.csv"
):
    # 1. Load raw data
    df_raw = processor.csv_bytes_to_raw_df(csv_content)

    # 2. Run inference across all strategies
    all_preds = engine.predict_all(df_raw, processor)

    metrics_summary = {}
    prediction_stats = {}
    has_ground_truth = "label" in df_raw.columns

    # --------------------------------------------------
    # 3. Add Predictions to Dataframe (FIXED: Professional Naming)
    # --------------------------------------------------
    for name, preds in all_preds.items():
        # Add each model's prediction as a clear column (e.g., weaklink_prediction)
        df_raw[f"{name}_prediction"] = preds

        mal_count = int(sum(preds))
        prediction_stats[name] = {
            "malicious": mal_count,
            "benign": len(preds) - mal_count
        }

        if has_ground_truth:
            y_true = df_raw["label"].astype(int)
            y_pred = np.array(preds).astype(int)

            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred,
                average="binary",
                pos_label=1,
                zero_division=0
            )

            metrics_summary[name] = {
                "accuracy": float((y_true == y_pred).mean()),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1)
            }

    # Identify the primary result based on user selection
    primary_col = f"{strategy}_prediction" if f"{strategy}_prediction" in df_raw.columns else "combined_prediction"
    # Create a clean final column for the user
    df_raw["final_result"] = df_raw[primary_col]

    # --------------------------------------------------
    # 4. Overlap & Feature Analysis (Internal)
    # --------------------------------------------------
    if "weaklink" in all_preds and "donpai" in all_preds:
        w_preds = np.array(all_preds["weaklink"])
        d_preds = np.array(all_preds["donpai"])
        prediction_stats["overlap_analysis"] = {
            "Both Detected": int(((w_preds == 1) & (d_preds == 1)).sum()),
            "WeakLink Only": int(((w_preds == 1) & (d_preds == 0)).sum()),
            "DonPai Only": int(((w_preds == 0) & (d_preds == 1)).sum()),
            "Total Benign": int(((w_preds == 0) & (d_preds == 0)).sum())
        }

    feature_cols = [f for f in ALL_FEATURES if f in df_raw.columns]
    malicious_df = df_raw[df_raw["final_result"] == 1]
    benign_df = df_raw[df_raw["final_result"] == 0]

    prediction_stats["feature_analysis"] = {
        "mal_count": {k: int(v) for k, v in (malicious_df[feature_cols] > 0).sum().to_dict().items()},
        "ben_count": {k: int(v) for k, v in (benign_df[feature_cols] > 0).sum().to_dict().items()}
    }

    # --------------------------------------------------
    # 5. Save Metadata and Clean CSV
    # --------------------------------------------------
    meta_path = RESULTS_DIR / f"{job_id}.json"
    with open(meta_path, "w") as f:
        json.dump({
            "status": "completed",
            "has_ground_truth": has_ground_truth,
            "all_metrics": metrics_summary,
            "prediction_stats": prediction_stats,
            "job_id": job_id,
            "original_filename": original_filename,
            "selected_strategy": strategy
        }, f)

    # Save CSV: No more 'temp_pred', only clear, named results
    df_raw.to_csv(RESULTS_DIR / f"{job_id}.csv", index=False)