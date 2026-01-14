import os
import json
from pathlib import Path
from celery import Celery
import pandas as pd

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
def process_prediction_task(csv_content: bytes, job_id: str, strategy: str = "combined"):
    """
    Processes CSV for malware detection.
    Standardizes on XGBoost for combined/weaklink and RF for donpai.
    """
    # 1. Load data
    df_raw = processor.csv_bytes_to_raw_df(csv_content)

    # 2. Run inference using Hybrid approach
    all_preds = engine.predict_all(df_raw, processor)

    # 3. Accuracy Calculation
    accuracies = {}
    has_ground_truth = "label" in df_raw.columns

    if has_ground_truth:
        for name, preds in all_preds.items():
            # Aligning with research validation metrics
            correct = (df_raw["label"].astype(str) == pd.Series(preds).astype(str)).mean()
            accuracies[name] = float(correct)

    # 4. Save Metadata for UI
    meta_path = RESULTS_DIR / f"{job_id}.json"
    with open(meta_path, "w") as f:
        # If no labels exist, accuracy is None (Scan mode)
        selected_accuracy = accuracies.get(strategy) if has_ground_truth else None

        json.dump({
            "status": "completed",
            "selected_strategy": strategy,
            "has_ground_truth": has_ground_truth, # Variable name fixed
            "accuracy": selected_accuracy,
            "all_accuracies": accuracies,
            "n_rows": len(df_raw)
        }, f)

    # 5. Map results and save CSV
    predictions = all_preds.get(strategy, [])
    df_raw["predicted_label"] = predictions
    # Mapping numeric outputs to research categories
    df_raw["detection_result"] = pd.Series(predictions).map({1: "1", 0: "0"})
    df_raw["model_strategy"] = strategy

    df_raw.to_csv(RESULTS_DIR / f"{job_id}.csv", index=False)