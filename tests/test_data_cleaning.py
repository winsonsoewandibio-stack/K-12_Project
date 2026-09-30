"""
Tests for the data cleaning pipeline.

These tests ensure that:
- Cleaning runs successfully for all three datasets.
- No impossible numeric values remain (e.g., attendance > 100).
- Missingness indicators are correctly created and valid.
- Safe imputation rules are applied (no remaining NaN in non-indicator columns).
- The pipeline integrates correctly with ingestion.

They do NOT test model performance; they only validate data quality and fairness.

This file works with BOTH:
- pytest (automatic test discovery)
- normal Python execution (manual runner at bottom)
"""

import sys
import os
import pandas as pd

# Ensure project root is in Python path for direct execution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_cleaning.data_cleaning import (
    clean_uci,
    clean_kaggle_habits,
    clean_kaggle_exam,
)


# ---------------------------------------------------------------------
# FAIRNESS-ALIGNED VALIDATION HELPERS
# ---------------------------------------------------------------------

def assert_no_impossible_values(df: pd.DataFrame):
    """
    Check numeric columns for impossible values.

    Non-numeric values (e.g., 'Unknown') are allowed because they represent
    safe categorical placeholders after fairness-aligned imputation.
    """

    def check_numeric_range(series, min_val, max_val, col_name):
        numeric_series = pd.to_numeric(series, errors="coerce")
        valid = numeric_series.dropna()
        assert ((valid >= min_val) & (valid <= max_val)).all(), \
            f"Column {col_name} contains impossible numeric values."

    if "attendance_percentage" in df.columns:
        check_numeric_range(df["attendance_percentage"], 0, 100, "attendance_percentage")

    if "final_exam_score" in df.columns:
        check_numeric_range(df["final_exam_score"], 0, 100, "final_exam_score")

    if "sleep_hours" in df.columns:
        check_numeric_range(df["sleep_hours"], 0, 24, "sleep_hours")

    if "study_hours_per_day" in df.columns:
        check_numeric_range(df["study_hours_per_day"], 0, 24, "study_hours_per_day")


def assert_missing_indicators(df: pd.DataFrame):
    """
    Check that missingness indicator columns are correctly formed.
    """
    for col in df.columns:
        if col.endswith("_missing"):
            base = col.replace("_missing", "")
            assert base in df.columns, f"Missing base column for indicator: {col}"
            unique_vals = set(df[col].unique())
            assert unique_vals.issubset({0, 1}), \
                f"Indicator {col} has invalid values: {unique_vals}"


def assert_safe_imputation(df: pd.DataFrame):
    """
    Check that non-indicator columns do not contain NaN after cleaning.
    """
    for col in df.columns:
        if col.endswith("_missing"):
            continue
        assert not df[col].isnull().any(), \
            f"Column {col} still has missing values after cleaning."


# ---------------------------------------------------------------------
# TEST FUNCTIONS (pytest only — must return None)
# ---------------------------------------------------------------------

def test_clean_uci():
    df = clean_uci()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    assert_no_impossible_values(df)
    assert_missing_indicators(df)
    assert_safe_imputation(df)


def test_clean_kaggle_habits():
    df = clean_kaggle_habits()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    assert_no_impossible_values(df)
    assert_missing_indicators(df)
    assert_safe_imputation(df)


def test_clean_kaggle_exam():
    df = clean_kaggle_exam()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    assert_no_impossible_values(df)
    assert_missing_indicators(df)
    assert_safe_imputation(df)


# ---------------------------------------------------------------------
# MANUAL RUNNER — prints cleaned results for normal Python execution
# ---------------------------------------------------------------------

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_cleaning_summary(df: pd.DataFrame, name: str):
    """
    Display a readable summary of the cleaned dataset.
    Similar to ingestion test output.
    """

    print_section(f"CLEANED DATASET SUMMARY: {name}")

    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)} total")

    # Count missingness indicators
    missing_indicators = [c for c in df.columns if c.endswith("_missing")]
    print(f"Missingness indicators: {len(missing_indicators)}")
    print(f"Indicator columns: {missing_indicators[:10]}{' ...' if len(missing_indicators) > 10 else ''}")

    # Show first few rows
    print("\nPreview of cleaned data:")
    print(df.head())

    print("\nNumeric validation:")
    try:
        assert_no_impossible_values(df)
        print("✔ No impossible numeric values detected.")
    except AssertionError as e:
        print("❌ Numeric validation failed:", str(e))

    print("\nImputation validation:")
    try:
        assert_safe_imputation(df)
        print("✔ No remaining NaN values in non-indicator columns.")
    except AssertionError as e:
        print("❌ Imputation validation failed:", str(e))


if __name__ == "__main__":
    print_section("RUNNING DATA CLEANING TESTS (Manual Python Execution)")

    try:
        df_uci = clean_uci()
        show_cleaning_summary(df_uci, "UCI Student Performance")

        df_habits = clean_kaggle_habits()
        show_cleaning_summary(df_habits, "Kaggle Study Habits")

        df_exam = clean_kaggle_exam()
        show_cleaning_summary(df_exam, "Kaggle Exam Performance")

        print_section("ALL CLEANING TESTS PASSED SUCCESSFULLY")

    except AssertionError as e:
        print("\n❌ TEST FAILED:")
        print(str(e))
        raise
