from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from app.api.inference import InferenceEngine
from app.api.processor import Processor
from app.api.worker import process_prediction_task


# ----------------------------
# Schemas
# ----------------------------
class PredictJSONRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., description="List of JSON rows.")


class PredictResponse(BaseModel):
    n_rows: int
    predictions: List[Any]
    topk: Optional[Dict[str, Any]] = None


# ----------------------------
# App
# ----------------------------
def create_app() -> FastAPI:
    app = FastAPI(title="ML Inference API - Distributed", version="2.0.0")

    # Shared storage directory
    RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/shared/results"))

    # Components for synchronous/health checks
    processor = Processor()
    engine = InferenceEngine()

    @app.get("/", response_class=HTMLResponse)
    def root():
        ui_path = Path(__file__).resolve().parents[1] / "ui" / "ui.html"
        return ui_path.read_text(encoding="utf-8") if ui_path.exists() else "UI Not Found"

    @app.post("/process_csv")
    async def process_csv(file: UploadFile = File(...)):
        """Queue a CSV for processing."""
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        job_id = str(uuid.uuid4())

        # Send to Redis via Celery
        process_prediction_task.delay(content, job_id)

        return JSONResponse({
            "job_id": job_id,
            "status": "Accepted",
            "check_status_url": f"/status/{job_id}",
            "download_url": f"/download/{job_id}"
        }, status_code=202)

    @app.get("/status/{job_id}")
    def get_status(job_id: str):
        """Checks if worker finished by reading the metadata file."""
        meta_path = RESULTS_DIR / f"{job_id}.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return json.load(f)
        return {"job_id": job_id, "status": "Processing"}

    @app.get("/download/{job_id}")
    def download(job_id: str):
        """Serves the completed file."""
        path = RESULTS_DIR / f"{job_id}.csv"
        if not path.exists():
            raise HTTPException(status_code=404, detail="File not ready")
        return FileResponse(path, media_type="text/csv", filename=f"result_{job_id}.csv")

    return app


app = create_app()