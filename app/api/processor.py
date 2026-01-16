from __future__ import annotations

import os
from io import BytesIO
from typing import Any, Dict, List

import pandas as pd


class Processor:
    """
    Builds model-ready dataframes for various research strategies (WeakLink, DonPai, Combined).

    Behavior:
    - No longer loads a static features.json on initialization.
    - Uses prepare_for_model() to process a raw dataframe against a specific feature list.
    - Ensures data is strictly numeric for XGBoost compatibility.
    """

    def __init__(self) -> None:
        # Default value for missing features
        self.fill_missing = float(os.getenv("FILL_MISSING", "0.0"))

    def csv_bytes_to_raw_df(self, csv_bytes: bytes) -> pd.DataFrame:
        """
        Loads raw data from CSV bytes, preserving all original columns.
        This is used by the Worker to keep metadata for the final output.
        """
        if not csv_bytes:
            raise ValueError("Empty CSV file")
        # Use BytesIO to read the byte stream as a file
        return pd.read_csv(BytesIO(csv_bytes))

    def json_rows_to_raw_df(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """Loads raw data from a list of JSON dictionaries."""
        if not rows:
            raise ValueError("No rows provided")
        return pd.DataFrame(rows)

    def prepare_for_model(self, df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        """
        Processes a raw dataframe into the feature set required by a specific model.

        Args:
            df: The raw input dataframe containing all columns.
            feature_list: The specific list of features (e.g., DonPai list) required for the model.
        """
        df_model = df.copy()

        # Step 1: Ensure all expected features exist in the dataframe
        for f in feature_list:
            if f not in df_model.columns:
                # Add missing feature column with default filler
                df_model[f] = self.fill_missing

        # Step 2: Keep ONLY the requested features in the exact order specified
        # This is critical for XGBoost consistency
        df_model = df_model[feature_list].copy()

        # Step 3: Force numeric conversion
        # XGBoost requires numeric inputs. Coerce errors to NaN and then fill.
        df_model = df_model.apply(pd.to_numeric, errors="coerce").fillna(self.fill_missing)

        return df_model