import os
import json
from pathlib import Path
from celery import Celery
import pandas as pd

from app.api.inference import InferenceEngine
from app.api.processor import Processor

# Initialize Celery with Redis as broker and backend
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
    Processes CSV using specific or multiple models based on the selected strategy.

    Args:
        csv_content: Raw bytes of the uploaded CSV.
        job_id: Unique identifier for the task.
        strategy: The chosen model ('weaklink', 'donpai', or 'combined').
    """
    # 1. Load data into a raw dataframe
    df_raw = processor.csv_bytes_to_raw_df(csv_content)

    # 2. Run all models to get a full comparison
    # engine.predict_all returns a dict: {'weaklink': [...], 'donpai': [...], 'combined': [...]}
    all_preds = engine.predict_all(df_raw, processor)

    # 3. Calculate accuracies if ground truth 'label' exists
    accuracies = {}
    if "label" in df_raw.columns:
        for name, preds in all_preds.items():
            # Standardize comparison by converting to string
            correct = (df_raw["label"].astype(str) == pd.Series(preds).astype(str)).mean()
            accuracies[name] = float(correct)

    # 4. Save metadata for the UI
    meta_path = RESULTS_DIR / f"{job_id}.json"
    with open(meta_path, "w") as f:
        # Extract the accuracy for the SPECIFIC strategy selected by the user
        selected_accuracy = accuracies.get(strategy, 0.0)

        json.dump({
            "status": "completed",
            "selected_strategy": strategy,
            "accuracy": selected_accuracy,  # Key used by the UI badge
            "all_accuracies": accuracies,  # Full breakdown for the logs
            "n_rows": len(df_raw)
        }, f)

    # 5. Save the results to CSV
    # We add the predictions of the selected model and a label for clarity
    df_raw["predicted_label"] = all_preds.get(strategy, [])
    df_raw["model_strategy"] = strategy

    df_raw.to_csv(RESULTS_DIR / f"{job_id}.csv", index=False)