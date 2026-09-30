"""
Fairness-aligned cleaning rules.

Implements:
- Missingness indicators
- Safe imputation (0 / Unknown / False)
- Impossible value capping
- Categorical normalization
- Numeric normalization
- Data type enforcement
"""

import pandas as pd

from src.data_cleaning.cleaning_utils import (
    cap_range,
    normalize_percentage,
    is_numeric_series,
    is_boolean_series,
    convert_none_to_nan,
)


# ---------------------------------------------------------------------
# CATEGORY NORMALIZATION
# ---------------------------------------------------------------------

def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize categorical values to ensure consistency.
    """
    df = df.copy()

    CATEGORY_MAPS = {
        "gender": {
            "male": "Male",
            "m": "Male",
            "female": "Female",
            "f": "Female",
        },
        "internet_access": {
            "yes": "Yes",
            "y": "Yes",
            "no": "No",
            "n": "No",
        },
        "school_type": {
            "urban": "Urban",
            "rural": "Rural",
        }
    }

    for col, mapping in CATEGORY_MAPS.items():
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(mapping)
                .fillna(df[col])
            )

    return df


# ---------------------------------------------------------------------
# MISSINGNESS INDICATORS
# ---------------------------------------------------------------------

def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create <col>_missing indicator columns for missing values.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            df[f"{col}_missing"] = df[col].isnull().astype(int)
    return df


# ---------------------------------------------------------------------
# SAFE IMPUTATION
# ---------------------------------------------------------------------

def safe_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values with safe defaults:
    - numeric → 0
    - boolean → False
    - categorical → "Unknown"
    """
    df = df.copy()

    for col in df.columns:
        if col.endswith("_missing"):
            continue

        series = df[col]

        if series.isnull().any():
            if is_numeric_series(series):
                df[col] = series.fillna(0)
            elif is_boolean_series(series):
                df[col] = series.fillna(False)
            else:
                df[col] = series.fillna("Unknown")

    return df


# ---------------------------------------------------------------------
# IMPOSSIBLE VALUE REMOVAL
# ---------------------------------------------------------------------

def remove_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap impossible values (not outliers).
    """
    df = df.copy()

    if "attendance_percentage" in df.columns:
        df = cap_range(df, "attendance_percentage", 0, 100)

    if "final_exam_score" in df.columns:
        df = cap_range(df, "final_exam_score", 0, 100)

    if "sleep_hours" in df.columns:
        df = cap_range(df, "sleep_hours", 0, 24)

    if "study_hours_per_day" in df.columns:
        df = cap_range(df, "study_hours_per_day", 0, 24)

    return df


# ---------------------------------------------------------------------
# NUMERIC NORMALIZATION
# ---------------------------------------------------------------------

def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numeric fields AFTER imputation.
    """
    df = df.copy()

    if "attendance_percentage" in df.columns:
        df = normalize_percentage(df, "attendance_percentage", "attendance_normalized")

    return df


# ---------------------------------------------------------------------
# DATA TYPE ENFORCEMENT
# ---------------------------------------------------------------------

def enforce_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce correct data types after cleaning.
    """
    df = df.copy()

    numeric_cols = [
        "age",
        "study_hours_per_day",
        "sleep_hours",
        "attendance_percentage",
        "final_exam_score",
        "attendance_normalized",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    # Convert categorical columns to string
    for col in df.columns:
        if df[col].dtype == object and not col.endswith("_missing"):
            df[col] = df[col].astype(str)

    # Missingness indicators stay int
    return df


# ---------------------------------------------------------------------
# FULL CLEANING PIPELINE
# ---------------------------------------------------------------------

def apply_cleaning_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full fairness-aligned cleaning pipeline.
    """
    df = df.copy()
    df = convert_none_to_nan(df)
    df = remove_impossible_values(df)
    df = normalize_categories(df)
    df = add_missing_indicators(df)
    df = safe_impute(df)
    df = normalize_numeric(df)
    df = enforce_data_types(df)
    return df
