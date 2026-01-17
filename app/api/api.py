import json
import os
import uuid
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from app.api.worker import process_prediction_task


def create_app() -> FastAPI:
    app = FastAPI(title="NPM Security API", version="2.2.0")
    RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/shared/results"))

    @app.get("/", response_class=HTMLResponse)
    def root():
        ui_path = Path(__file__).resolve().parents[1] / "ui" / "ui.html"
        return ui_path.read_text(encoding="utf-8") if ui_path.exists() else "UI Not Found"

    @app.post("/process_csv")
    async def process_csv(file: UploadFile = File(...), strategy: str = Form("combined")):
        content = await file.read()
        job_id = str(uuid.uuid4())
        process_prediction_task.delay(content, job_id, strategy, file.filename)
        return JSONResponse({"job_id": job_id, "status": "Accepted"})

    @app.get("/status/{job_id}")
    def get_status(job_id: str):
        meta_path = RESULTS_DIR / f"{job_id}.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return json.load(f)
        return {"status": "Processing"}

    @app.get("/download/{job_id}")
    def download(job_id: str):
        path = RESULTS_DIR / f"{job_id}.csv"
        meta_path = RESULTS_DIR / f"{job_id}.json"
        if not path.exists():
            raise HTTPException(status_code=404)

        final_filename = f"result_{job_id}.csv"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
                stem = Path(meta.get("original_filename", "data.csv")).stem
                strat = meta.get("selected_strategy", "combined")
                final_filename = f"{stem}_{strat}_analysis.csv"

        return FileResponse(path, media_type="text/csv", filename=final_filename)

    return app


app = create_app()