"""
data_cleaning.py
----------------
Main CleaningPipeline that orchestrates:

1. Column normalization
2. Missing value handling
3. Numeric anomaly correction
4. SIS normalization
5. Metadata extraction
6. NFR metric recording

This pipeline is non-destructive: no columns are dropped.
"""

import time
import pandas as pd

from src.data_cleaning.cleaning_utils import (
    normalize_columns,
    fill_missing_values,
    fix_numeric_anomalies,
    extract_metadata,
)

from src.data_cleaning.schema_validation import normalize_sis_fields
from src.data_cleaning.cleaning_rules import CLEANING_NFR_THRESHOLDS


class CleaningPipeline:
    """
    CleaningPipeline
    ----------------
    Applies universal cleaning rules and records NFR metrics.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        # Initialize metrics
        self.metrics = {
            "cleaning_time": None,
            "record_count": len(df),
            "numeric_anomalies_fixed": 0,
            "missing_values_filled": 0,
            "column_normalized": False,
        }

        # Metadata for feature engineering
        self.metadata = {}

    def clean(self):
        """Run the full cleaning pipeline."""
        start = time.time()

        # Column normalization
        self.df = normalize_columns(self.df)
        self.metrics["column_normalized"] = True

        # Missing values (CoW-safe)
        self.metrics["missing_values_filled"] = fill_missing_values(self.df)

        # Numeric anomalies
        self.metrics["numeric_anomalies_fixed"] = fix_numeric_anomalies(self.df)

        # SIS normalization
        self.df = normalize_sis_fields(self.df)

        # Metadata extraction
        self.metadata = extract_metadata(self.df)

        end = time.time()
        self.metrics["cleaning_time"] = end - start

        return self.df, self.metrics, self.metadata
