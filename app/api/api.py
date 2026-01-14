from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from app.api.inference import InferenceEngine
from app.api.processor import Processor


# ----------------------------
# Schemas
# ----------------------------
class PredictJSONRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., description="List of JSON rows (dicts).")


class PredictResponse(BaseModel):
    n_rows: int
    predictions: List[Any]
    topk: Optional[Dict[str, Any]] = None


# ----------------------------
# App factory
# ----------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title=os.getenv("API_TITLE", "ML Inference API"),
        version=os.getenv("API_VERSION", "1.0.0"),
    )

    processor = Processor()
    engine = InferenceEngine()

    # Results storage (ephemeral by default)
    RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/tmp/results"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_INDEX: Dict[str, str] = {}  # job_id -> filepath

    # UI
    UI_PATH = Path(__file__).resolve().parents[1] / "ui" / "ui.html"

    @app.get("/", response_class=HTMLResponse)
    def root():
        if UI_PATH.exists():
            return UI_PATH.read_text(encoding="utf-8")
        return "<h3>UI not found</h3><p>Expected at: app/ui/ui.html</p>"

    # Health / schema
    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "model_path": engine.model_path,
            "features_path": str(processor.features_path),
            "n_features": len(processor.expected_features),
            "expected_features": processor.expected_features,
        }

    # JSON predict
    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictJSONRequest) -> PredictResponse:
        try:
            df_model = processor.json_rows_to_df(req.rows)
            out = engine.predict(df_model)
            preds = out.get("predictions", [])
            def _flatten_preds(preds):
                flat = []
                for p in preds:
                # if p is like [0] or np.array([0])
                    if isinstance(p, (list, tuple)) and len(p) == 1:
                        flat.append(p[0])
                    else:
                        flat.append(p)
                return flat

            preds = _flatten_preds(preds)
            if not isinstance(preds, list):
                preds = list(preds)
            return PredictResponse(
                n_rows=int(out.get("n_rows", len(preds))),
                predictions=preds,
                topk=out.get("topk"),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # CSV process (predictions + optional accuracy + downloadable file)
    @app.post("/process_csv")
    async def process_csv(file: UploadFile = File(...)):
        """
        Upload a CSV.

        - If it contains a 'label' column (case/whitespace/BOM tolerant), compute accuracy.
        - Always return a downloadable CSV that contains a 'predictions' column.
        """
        try:
            content = await file.read()
            if not content:
                raise ValueError("Empty CSV file")

            # 1) RAW DF for label + output
            df_raw = processor.csv_bytes_to_raw_df(content)

            # Normalize column names (BOM + whitespace)
            df_raw.columns = [str(c).replace("\ufeff", "").strip() for c in df_raw.columns]

            # Detect label robustly (case-insensitive)
            lower_map = {c.lower(): c for c in df_raw.columns}
            label_col = lower_map.get("label")
            has_label = label_col is not None

            # 2) Model DF (21 features only)
            df_model = processor.csv_bytes_to_df(content)

            # 3) Predict
            out = engine.predict(df_model)
            preds = out.get("predictions", [])
            if not isinstance(preds, list):
                preds = list(preds) if preds is not None else []

            if len(preds) != len(df_raw):
                raise ValueError(f"Predictions length {len(preds)} != rows {len(df_raw)}")

            # 4) Output CSV = raw + predictions
            df_out = df_raw.copy()
            df_out["predictions"] = preds

            # 5) Accuracy (only if label exists)
            accuracy = None
            if has_label:
                y_true = df_out[label_col].astype(str)
                y_pred = df_out["predictions"].astype(str)
                accuracy = float((y_true == y_pred).mean())

            # 6) Save output
            job_id = str(uuid.uuid4())
            out_path = RESULTS_DIR / f"{job_id}.csv"
            df_out.to_csv(out_path, index=False)
            RESULT_INDEX[job_id] = str(out_path)

            return JSONResponse(
                {
                    "job_id": job_id,
                    "has_label": has_label,
                    "accuracy": accuracy,  # null if no label
                    "download_url": f"/download/{job_id}",
                    "n_rows": int(len(df_out)),
                }
            )

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"CSV processing failed: {e}")

    # Download processed CSV
    @app.get("/download/{job_id}")
    def download(job_id: str):
        path = RESULT_INDEX.get(job_id)
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="File not found (job expired or invalid id).")

        return FileResponse(
            path,
            media_type="text/csv",
            filename=f"result_{job_id}.csv",
        )

    return app


app = create_app()
