"""
test_data_cleaning.py
---------------------
This test suite validates the entire cleaning module:

1. Functional cleaning correctness for all datasets.
2. Schema preservation (non-destructive cleaning).
3. SIS schema validation after cleaning.
4. NFR compliance (cleaning_time, scalability).
5. Human-readable output (df.head(), shape, metrics, metadata).
6. Integration with ingestion → cleaning pipeline.

This file is intentionally verbose because it serves as a
demonstration artifact for your capstone project.
"""

import sys
import os
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import ingestion loaders
from src.data_ingestion.load_dataset_uci import load_uci_dataset
from src.data_ingestion.load_dataset_kaggle_habit import load_kaggle_student_habits
from src.data_ingestion.load_dataset_kaggle_exam import load_kaggle_exam_performance

# Import cleaning pipeline
from src.data_cleaning.data_cleaning import CleaningPipeline

# Import SIS + NFR rules
from src.data_cleaning.schema_validation import validate_sis_schema
from src.data_cleaning.cleaning_rules import CLEANING_NFR_THRESHOLDS


# -------------------------------------------------------------------
# Helper printing utilities
# -------------------------------------------------------------------

def print_section(title):
    """Prints a formatted section header for readability."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_cleaning_summary(df, metrics, metadata):
    """
    Prints a human-readable summary of the dataset after cleaning.
    This is used as proof that cleaning works correctly.
    """
    print("\n--- CLEANED DATASET SUMMARY ---")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(list(df.columns))

    print("\nHead:")
    print(df.head())

    print("\n--- CLEANING METRICS ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n--- METADATA (for Feature Engineering) ---")
    for k, v in metadata.items():
        print(f"{k}: {v}")


# -------------------------------------------------------------------
# Functional Cleaning Tests
# -------------------------------------------------------------------

def test_cleaning_uci():
    print_section("FUNCTIONAL CLEANING TEST — UCI DATASET")

    raw_df, _ = load_uci_dataset()
    pipeline = CleaningPipeline(raw_df)
    cleaned_df, metrics, metadata = pipeline.clean()

    assert isinstance(cleaned_df, pd.DataFrame)
    assert len(cleaned_df) > 0

    print_cleaning_summary(cleaned_df, metrics, metadata)


def test_cleaning_kaggle_habits():
    print_section("FUNCTIONAL CLEANING TEST — KAGGLE HABITS DATASET")

    raw_df, _ = load_kaggle_student_habits()
    pipeline = CleaningPipeline(raw_df)
    cleaned_df, metrics, metadata = pipeline.clean()

    assert isinstance(cleaned_df, pd.DataFrame)
    assert len(cleaned_df) > 0

    print_cleaning_summary(cleaned_df, metrics, metadata)


def test_cleaning_kaggle_exam():
    print_section("FUNCTIONAL CLEANING TEST — KAGGLE EXAM DATASET")

    raw_df, _ = load_kaggle_exam_performance()
    pipeline = CleaningPipeline(raw_df)
    cleaned_df, metrics, metadata = pipeline.clean()

    assert isinstance(cleaned_df, pd.DataFrame)
    assert len(cleaned_df) > 0

    print_cleaning_summary(cleaned_df, metrics, metadata)


# -------------------------------------------------------------------
# Schema Preservation Tests (non-destructive cleaning)
# -------------------------------------------------------------------

def test_schema_preservation_uci():
    print_section("SCHEMA PRESERVATION — UCI")

    raw_df, _ = load_uci_dataset()
    pipeline = CleaningPipeline(raw_df)
    cleaned_df, _, _ = pipeline.clean()

    for col in raw_df.columns:
        assert col.lower().strip() in cleaned_df.columns

    print("Schema preserved for UCI dataset.")


def test_schema_preservation_kaggle_habits():
    print_section("SCHEMA PRESERVATION — KAGGLE HABITS")

    raw_df, _ = load_kaggle_student_habits()
    pipeline = CleaningPipeline(raw_df)
    cleaned_df, _, _ = pipeline.clean()

    for col in raw_df.columns:
        assert col.lower().strip() in cleaned_df.columns

    print("Schema preserved for Kaggle Habits dataset.")


def test_schema_preservation_kaggle_exam():
    print_section("SCHEMA PRESERVATION — KAGGLE EXAM")

    raw_df, _ = load_kaggle_exam_performance()
    pipeline = CleaningPipeline(raw_df)
    cleaned_df, _, _ = pipeline.clean()

    for col in raw_df.columns:
        assert col.lower().strip() in cleaned_df.columns

    print("Schema preserved for Kaggle Exam dataset.")


# -------------------------------------------------------------------
# SIS Schema Validation Tests
# -------------------------------------------------------------------

def test_sis_validation_all():
    print_section("SIS VALIDATION — ALL DATASETS")

    datasets = {
        "UCI": load_uci_dataset,
        "Kaggle Habits": load_kaggle_student_habits,
        "Kaggle Exam": load_kaggle_exam_performance,
    }

    for name, loader in datasets.items():
        print_section(f"SIS VALIDATION — {name}")

        raw_df, _ = loader()
        pipeline = CleaningPipeline(raw_df)
        cleaned_df, _, _ = pipeline.clean()

        missing = validate_sis_schema(cleaned_df)
        assert len(missing) == 0

        print(f"SIS fields validated for {name} dataset.")


# -------------------------------------------------------------------
# NFR Tests (cleaning_time, scalability)
# -------------------------------------------------------------------

def test_cleaning_nfr_all():
    print_section("NFR TEST — CLEANING MODULE (ALL DATASETS)")

    datasets = {
        "UCI": load_uci_dataset,
        "Kaggle Habits": load_kaggle_student_habits,
        "Kaggle Exam": load_kaggle_exam_performance,
    }

    for name, loader in datasets.items():
        print_section(f"NFR TEST — {name}")

        raw_df, _ = loader()
        pipeline = CleaningPipeline(raw_df)
        cleaned_df, metrics, metadata = pipeline.clean()

        assert metrics["cleaning_time"] <= CLEANING_NFR_THRESHOLDS["cleaning_time_max"]
        assert metrics["record_count"] <= CLEANING_NFR_THRESHOLDS["max_records"]

        print_cleaning_summary(cleaned_df, metrics, metadata)

    print("\nAll NFR tests passed for all datasets.")


# -------------------------------------------------------------------
# Direct Execution Support
# -------------------------------------------------------------------

if __name__ == "__main__":
    test_cleaning_uci()
    test_cleaning_kaggle_habits()
    test_cleaning_kaggle_exam()
    test_schema_preservation_uci()
    test_schema_preservation_kaggle_habits()
    test_schema_preservation_kaggle_exam()
    test_sis_validation_all()
    test_cleaning_nfr_all()
    print_section("ALL CLEANING TESTS COMPLETED SUCCESSFULLY")
