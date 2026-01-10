from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class Processor:
    """
    Builds a model-ready dataframe using an explicit feature list from features.json.

    Behavior:
    - Reads features from FEATURES_PATH (default: /app/artifacts/features.json)
    - Drops non-feature columns (label, predictions, Unnamed: 0, etc.)
    - Adds missing features (fill_missing)
    - Keeps ONLY features, in the correct order
    """

    def __init__(self) -> None:
        self.fill_missing = float(os.getenv("FILL_MISSING", "0.0"))
        self.features_path = Path(os.getenv("FEATURES_PATH", "/app/artifacts/features.json"))
        self.expected_features = self._load_features()

    def _load_features(self) -> List[str]:
        if not self.features_path.exists():
            raise FileNotFoundError(
                f"features.json not found at {self.features_path}. "
                f"Mount/copy it and set FEATURES_PATH."
            )
        data = json.loads(self.features_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise ValueError("features.json must be a JSON array of strings (feature names).")
        return data

    def csv_bytes_to_df(self, csv_bytes: bytes) -> pd.DataFrame:
        if not csv_bytes:
            raise ValueError("Empty CSV file")
        df_raw = pd.read_csv(BytesIO(csv_bytes))
        df_model = self._to_model_df(df_raw)
        return df_model

    def json_rows_to_df(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            raise ValueError("No rows provided")
        df_raw = pd.DataFrame(rows)
        df_model = self._to_model_df(df_raw)
        return df_model

    def csv_bytes_to_raw_df(self, csv_bytes: bytes) -> pd.DataFrame:
        """Useful for output files/accuracy checks (keeps original columns)."""
        if not csv_bytes:
            raise ValueError("Empty CSV file")
        return pd.read_csv(BytesIO(csv_bytes))

    def _to_model_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop columns that are never features
        drop = [c for c in ["label", "predictions"] if c in df.columns]
        if "Unnamed: 0" in df.columns:
            drop.append("Unnamed: 0")
        if drop:
            df = df.drop(columns=drop)

        # Add missing features
        for f in self.expected_features:
            if f not in df.columns:
                df[f] = self.fill_missing

        # Keep only the 21 expected features, in order
        df = df[self.expected_features].copy()

        # Force numeric (XGBoost expects numbers)
        # Coerce errors to NaN then fill
        df = df.apply(pd.to_numeric, errors="coerce").fillna(self.fill_missing)

        return df
