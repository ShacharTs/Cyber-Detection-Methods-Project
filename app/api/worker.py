import os
import json
from pathlib import Path
from celery import Celery
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from app.api.inference import InferenceEngine
from app.api.processor import Processor

# Initialize Celery with Redis
celery_app = Celery(
    "prediction_tasks",
    broker=os.getenv("REDIS_URL", "redis://redis:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://redis:6379/0")
)

processor = Processor()
engine = InferenceEngine()
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/shared/results"))


@celery_app.task(name="process_prediction_task")
def process_prediction_task(csv_content: bytes, job_id: str, strategy: str = "combined",
                            original_filename: str = "data.csv"):
    """
    Processes CSV for malware detection and prepares comparative research data.
    """
    # 1. Load data
    df_raw = processor.csv_bytes_to_raw_df(csv_content)

    # 2. Run inference for ALL models
    all_preds = engine.predict_all(df_raw, processor)

    # 3. Metrics and Distribution Calculation
    metrics_summary = {}
    prediction_stats = {}
    has_ground_truth = "label" in df_raw.columns

    for name, preds in all_preds.items():
        # Distribution stats for scan mode graphs
        mal_count = int(sum(preds))
        prediction_stats[name] = {
            "malicious": mal_count,
            "benign": len(preds) - mal_count
        }

        # Calculate performance metrics if labels are available
        if has_ground_truth:
            y_true = df_raw["label"].astype(str)
            y_pred = pd.Series(preds).astype(str)
            acc = (y_true == y_pred).mean()
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', pos_label="1", zero_division=0
            )
            metrics_summary[name] = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1)
            }

    # 4. Save Metadata for UI
    meta_path = RESULTS_DIR / f"{job_id}.json"
    with open(meta_path, "w") as f:
        selected_metrics = metrics_summary.get(strategy) if has_ground_truth else None
        json.dump({
            "status": "completed",
            "selected_strategy": strategy,
            "original_filename": original_filename,
            "has_ground_truth": has_ground_truth,
            "accuracy": selected_metrics["accuracy"] if selected_metrics else None,
            "all_metrics": metrics_summary,
            "prediction_stats": prediction_stats,
            "n_rows": len(df_raw)
        }, f)

    # 5. Filter and Save Output CSV for Download
    # Identify the package name column
    package_col = next((col for col in df_raw.columns if col.lower() in ['package_name', 'name']), df_raw.columns[0])

    output_cols = [package_col]
    df_raw.rename(columns={package_col: "Package Name"}, inplace=True)
    output_cols = ["Package Name"]

    # Add Ground Truth (Malicious/Benign) if label column exists
    if has_ground_truth:
        df_raw["Ground Truth"] = df_raw["label"].astype(str).map({
            "1": "Malicious", "0": "Benign", "1.0": "Malicious", "0.0": "Benign"
        })
        output_cols.append("Ground Truth")

    # Add predictions for each model
    model_labels = {
        "weaklink": "WeakLink Prediction",
        "donpai": "DonPai Prediction",
        "combined": "Hybrid Prediction (Ours)"
    }

    for model_key, label in model_labels.items():
        if model_key in all_preds:
            df_raw[label] = pd.Series(all_preds[model_key]).map({1: "Malicious", 0: "Benign"})
            output_cols.append(label)

    # Save final filtered CSV
    df_final = df_raw[output_cols].copy()
    df_final.to_csv(RESULTS_DIR / f"{job_id}.csv", index=False)