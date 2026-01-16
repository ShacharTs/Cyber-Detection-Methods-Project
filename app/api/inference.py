import os
import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Any, Dict


class InferenceEngine:
    def __init__(self) -> None:
        self.artifact_dir = Path(os.getenv("ARTIFACT_DIR", "/app/artifacts"))

        self.model_names = [
            "weaklink",
            "donpai",
            "combined",
            "combined_voting"
        ]

        self.models = {}
        self.features = {}
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return

        strategy_mapping = {
            "weaklink": "weaklink_xgb",
            "donpai": "donpai_rf",
            "combined": "combined_xgb",
            "combined_voting": "combined_voting_voting"
        }

        for name, prefix in strategy_mapping.items():
            model_path = self.artifact_dir / f"{prefix}_model.pkl"
            feat_path = self.artifact_dir / f"{name}_features.json"

            if model_path.exists() and feat_path.exists():
                self.models[name] = joblib.load(model_path)
                with open(feat_path, "r") as f:
                    self.features[name] = json.load(f)

        self._loaded = True


    def predict_all(self, df_raw: pd.DataFrame, processor: Any) -> Dict[str, list]:
        """Runs all strategies as required by worker.py."""
        self._ensure_loaded()
        all_results = {}

        for name, model in self.models.items():
            # Prepare data specifically for this model's feature set
            feat_list = self.features.get(name, [])
            df_proc = processor.prepare_for_model(df_raw, feat_list)

            # Generate predictions
            preds = model.predict(df_proc)
            all_results[name] = preds.tolist()

        return all_results