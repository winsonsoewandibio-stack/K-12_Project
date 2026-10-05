"""
cleaning_utils.py
-----------------
Utility functions used by the CleaningPipeline. These functions keep
the main pipeline readable and modular.

Each function performs ONE specific cleaning task.
"""

import pandas as pd
from src.data_cleaning.cleaning_rules import (
    MISSING_VALUE_STRATEGIES,
    NUMERIC_ANOMALY_RULES,
    COLUMN_NORMALIZATION_RULES,
)


# -------------------------------------------------------------------
# Column Normalization
# -------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to ensure consistency across datasets.

    - Lowercase all column names
    - Strip whitespace
    """
    if COLUMN_NORMALIZATION_RULES["lowercase_columns"]:
        df.columns = [c.lower() for c in df.columns]

    if COLUMN_NORMALIZATION_RULES["strip_whitespace"]:
        df.columns = [c.strip() for c in df.columns]

    return df


# -------------------------------------------------------------------
# Missing Value Handling (Copy-on-Write Safe)
# -------------------------------------------------------------------
def fill_missing_values(df: pd.DataFrame) -> int:
    """
    Fill missing values using type-based strategies.

    IMPORTANT:
    Pandas Copy-on-Write (CoW) requires SAFE assignment:
        df[col] = df[col].fillna(value)
    instead of:
        df[col].fillna(value, inplace=True)

    Returns:
        int: number of missing values filled.
    """
    missing_filled = 0

    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count == 0:
            continue

        # Numeric columns → median
        if df[col].dtype in ["int64", "float64"]:
            fill_value = df[col].median()
        else:
            # Categorical or boolean → mode
            fill_value = df[col].mode()[0]

        # SAFE: Copy-on-Write compatible
        df[col] = df[col].fillna(fill_value)

        missing_filled += missing_count

    return missing_filled


# -------------------------------------------------------------------
# Numeric Anomaly Correction (already safe)
# -------------------------------------------------------------------
def fix_numeric_anomalies(df: pd.DataFrame) -> int:
    """
    Fix negative numeric values and enforce min/max thresholds.

    Returns:
        int: number of anomalies corrected.
    """
    numeric_cols = df.select_dtypes(include=["number"])
    anomalies_fixed = 0

    for col in numeric_cols.columns:
        anomalies = df[col] < NUMERIC_ANOMALY_RULES["min_value"]
        anomaly_count = anomalies.sum()

        if anomaly_count > 0:
            df.loc[anomalies, col] = NUMERIC_ANOMALY_RULES["min_value"]
            anomalies_fixed += anomaly_count

    return anomalies_fixed


# -------------------------------------------------------------------
# Metadata Extraction (for Feature Engineering)
# -------------------------------------------------------------------
def extract_metadata(df: pd.DataFrame) -> dict:
    """
    Extract metadata needed for feature engineering.

    Returns:
        dict: metadata describing column types and SIS fields.
    """
    return {
        "numeric_columns": list(df.select_dtypes(include=["number"]).columns),

        # ⭐ UPDATED: Pandas warning removed
        "categorical_columns": list(df.select_dtypes(include=["object", "string"]).columns),

        "boolean_columns": list(df.select_dtypes(include=["bool"]).columns),

        "sis_fields": ["student_id", "grade_level", "attendance", "exam_score"],
        "total_columns": len(df.columns),
    }
