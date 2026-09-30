"""
Utility functions for data cleaning.

These helpers support the cleaning rules. They do NOT implement policy.
"""

import pandas as pd


def cap_range(df: pd.DataFrame, col: str, min_val: float, max_val: float) -> pd.DataFrame:
    """
    Cap values in a numeric column to a given [min_val, max_val] range.
    Used ONLY for impossible values (e.g., attendance > 100).
    """
    df = df.copy()
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].clip(lower=min_val, upper=max_val)
    return df


def normalize_percentage(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    """
    Normalize a percentage column (0–100) to a 0–1 scale.
    """
    df = df.copy()
    if source_col in df.columns and pd.api.types.is_numeric_dtype(df[source_col]):
        df[target_col] = df[source_col] / 100.0
    return df


def is_numeric_series(series: pd.Series) -> bool:
    """Check if a pandas Series is numeric."""
    return pd.api.types.is_numeric_dtype(series)


def is_boolean_series(series: pd.Series) -> bool:
    """Check if a pandas Series is boolean."""
    return pd.api.types.is_bool_dtype(series)


def convert_none_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Python None values to pandas NA for consistent missing handling.
    """
    df = df.copy()
    return df.where(pd.notnull(df), None).replace({None: pd.NA})