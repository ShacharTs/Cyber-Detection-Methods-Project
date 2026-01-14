import os
import json
import uuid
from pathlib import Path
from celery import Celery
import pandas as pd

from app.api.inference import InferenceEngine
from app.api.processor import Processor

# 1. Initialize Celery with Redis as the Message Broker
celery_app = Celery(
    "prediction_tasks",
    broker=os.getenv("REDIS_URL", "redis://redis:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://redis:6379/0")
)

# 2. Initialize engines once at startup to optimize performance
processor = Processor()
engine = InferenceEngine()

# Use a directory shared between API and Worker containers
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/shared/results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@celery_app.task(name="process_prediction_task")
def process_prediction_task(csv_content: bytes, job_id: str):
    """
    Background worker: Processes CSV, runs ML model, and saves results.
    """
    try:
        # Step A: Transform raw CSV bytes into model-ready features
        df_model = processor.csv_bytes_to_df(csv_content)

        # Step B: Perform Inference
        out = engine.predict(df_model)
        preds = out.get("predictions", [])

        # Step C: Load original data for the final output CSV
        df_raw = processor.csv_bytes_to_raw_df(csv_content)

        # Robust column cleanup (BOM/whitespace)
        df_raw.columns = [str(c).replace("\ufeff", "").strip() for c in df_raw.columns]
        df_raw["predictions"] = preds

        # Step D: Accuracy Calculation (if a label exists)
        accuracy = None
        lower_cols = {c.lower(): c for c in df_raw.columns}
        if "label" in lower_cols:
            label_col = lower_cols["label"]
            y_true = df_raw[label_col].astype(str)
            y_pred = df_raw["predictions"].astype(str)
            accuracy = float((y_true == y_pred).mean())

        # Step E: Save the prediction CSV
        csv_path = RESULTS_DIR / f"{job_id}.csv"
        df_raw.to_csv(csv_path, index=False)

        # Step F: Save metadata (JSON) so API doesn't have to read the CSV
        meta_path = RESULTS_DIR / f"{job_id}.json"
        with open(meta_path, "w") as f:
            json.dump({
                "job_id": job_id,
                "status": "completed",
                "accuracy": accuracy,
                "n_rows": len(df_raw)
            }, f)

        return {"status": "success", "job_id": job_id}

    except Exception as e:
        return {"status": "error", "message": str(e)}