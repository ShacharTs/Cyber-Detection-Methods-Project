from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import joblib


class InferenceEngine:
    """
    Expects an sklearn-like model at MODEL_PATH that implements:
      - predict(X) OR predict_proba(X)
    """

    def __init__(self) -> None:
        self.model_path = os.getenv("MODEL_PATH", "/app/artifacts/xgboost_model.pkl")
        self.top_k = int(os.getenv("TOP_K", "1"))
        self._model: Optional[Any] = None

    def _load(self) -> Any:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Mount/copy your model and set MODEL_PATH."
            )
        return joblib.load(self.model_path)

    def ensure_loaded(self) -> None:
        if self._model is None:
            self._model = self._load()

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.ensure_loaded()
        assert self._model is not None

        X = df.values
        out: Dict[str, Any] = {"n_rows": int(len(df))}

        model = self._model

        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(X))
            k = max(1, self.top_k)
            k = min(k, proba.shape[1])

            topk_idx = np.argsort(-proba, axis=1)[:, :k]
            topk_scores = np.take_along_axis(proba, topk_idx, axis=1)

            classes = getattr(model, "classes_", None)
            if classes is not None:
                classes = np.asarray(classes)
                topk_labels = classes[topk_idx].tolist()
            else:
                topk_labels = topk_idx.tolist()

            out["topk"] = {"k": k, "labels": topk_labels, "scores": topk_scores.tolist()}

            # top-1
            if k == 1:
                out["predictions"] = topk_labels  # already list-of-rows
            else:
                out["predictions"] = [row[0] for row in topk_labels]

        else:
            preds = model.predict(X)
            out["predictions"] = preds.tolist() if hasattr(preds, "tolist") else list(preds)
        preds = out.get("predictions", [])
        if isinstance(preds, list):
            out["predictions"] = [p[0] if isinstance(p, (list, tuple)) and len(p) == 1 else p for p in preds]
        return out
