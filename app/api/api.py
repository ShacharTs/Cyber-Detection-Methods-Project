from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from app.api.inference import InferenceEngine
from app.api.processor import Processor
from app.api.worker import process_prediction_task


def create_app() -> FastAPI:
    app = FastAPI(title="ML Inference API - Distributed", version="2.1.0")
    RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/shared/results"))
    processor = Processor()
    engine = InferenceEngine()

    @app.get("/", response_class=HTMLResponse)
    def root():
        ui_path = Path(__file__).resolve().parents[1] / "ui" / "ui.html"
        return ui_path.read_text(encoding="utf-8") if ui_path.exists() else "UI Not Found"

    @app.post("/process_csv")
    async def process_csv(
            file: UploadFile = File(...),
            strategy: str = Form("combined")
    ):
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        job_id = str(uuid.uuid4())

        # Pass the original filename to the worker task
        process_prediction_task.delay(content, job_id, strategy, file.filename)

        return JSONResponse({
            "job_id": job_id,
            "status": "Accepted",
            "strategy_selected": strategy,
            "check_status_url": f"/status/{job_id}",
            "download_url": f"/download/{job_id}"
        }, status_code=202)

    @app.get("/status/{job_id}")
    def get_status(job_id: str):
        meta_path = RESULTS_DIR / f"{job_id}.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return json.load(f)
        return {"job_id": job_id, "status": "Processing"}

    @app.get("/download/{job_id}")
    def download(job_id: str):
        """
        Serves the completed file with a dynamic name: [OriginalName]_[Strategy].csv
        """
        path = RESULTS_DIR / f"{job_id}.csv"
        meta_path = RESULTS_DIR / f"{job_id}.json"

        if not path.exists():
            raise HTTPException(status_code=404, detail="File not ready")

        # Default filename in case metadata is missing
        final_filename = f"result_{job_id}.csv"

        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
                orig_name = meta.get("original_filename", "results.csv")
                strategy = meta.get("selected_strategy", "combined")

                # Construct name: package_data_combined.csv
                stem = Path(orig_name).stem
                final_filename = f"{stem}_{strategy}.csv"

        return FileResponse(
            path,
            media_type="text/csv",
            filename=final_filename
        )

    return app


app = create_app()