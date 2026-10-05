"""
schema_validation.py
--------------------
Validates and normalizes SIS schema fields after cleaning.

This ensures that feature engineering receives consistent SIS fields
across all datasets.
"""

import pandas as pd
from src.data_ingestion.common_scheme import SIS_REQUIRED_FIELDS


# -------------------------------------------------------------------
# SIS Schema Validation
# -------------------------------------------------------------------
def validate_sis_schema(df):
    """
    Check whether all SIS fields exist in the cleaned dataset.

    Returns:
        list: missing SIS fields (empty list means valid).
    """
    missing = [f for f in SIS_REQUIRED_FIELDS if f not in df.columns]
    return missing


# -------------------------------------------------------------------
# SIS Field Normalization
# -------------------------------------------------------------------
def normalize_sis_fields(df):
    """
    Ensure SIS fields are numeric where required.

    - attendance → numeric
    - exam_score → numeric
    """
    if "attendance" in df.columns:
        df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")

    if "exam_score" in df.columns:
        df["exam_score"] = pd.to_numeric(df["exam_score"], errors="coerce")

    return df
