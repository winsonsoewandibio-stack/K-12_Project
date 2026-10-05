"""
test_feature_engineering.py
---------------------------
This test suite validates the entire feature engineering module:

1. Functional FE correctness for all datasets.
2. Schema validation (no duplicates, no missing values).
3. NFR compliance (fe_time, max_features).
4. Human-readable output (df.head(), shape, metrics).
5. Integration with ingestion → cleaning → feature engineering pipeline.

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

# Import feature engineering pipeline
from src.feature_engineering.feature_engineering import FeatureEngineeringPipeline

# Import NFR rules
from src.feature_engineering.feature_rules import FEATURE_NFR_THRESHOLDS


# -------------------------------------------------------------------
# Helper printing utilities
# -------------------------------------------------------------------

def print_section(title):
    """Prints a formatted section header for readability."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_fe_summary(df, metrics):
    """
    Prints a human-readable summary of the dataset after feature engineering.
    This is used as proof that FE works correctly.
    """
    print("\n--- FEATURE ENGINEERING SUMMARY ---")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(list(df.columns))

    print("\nHead:")
    print(df.head())

    print("\n--- FEATURE ENGINEERING METRICS ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")


# -------------------------------------------------------------------
# Functional Feature Engineering Tests
# -------------------------------------------------------------------

def test_fe_uci():
    print_section("FEATURE ENGINEERING TEST — UCI DATASET")

    raw_df, _ = load_uci_dataset()
    cleaned_df, _, metadata = CleaningPipeline(raw_df).clean()
    fe_df, fe_metrics = FeatureEngineeringPipeline(cleaned_df, metadata).engineer()

    assert isinstance(fe_df, pd.DataFrame)
    assert len(fe_df) > 0

    print_fe_summary(fe_df, fe_metrics)


def test_fe_kaggle_habits():
    print_section("FEATURE ENGINEERING TEST — KAGGLE HABITS DATASET")

    raw_df, _ = load_kaggle_student_habits()
    cleaned_df, _, metadata = CleaningPipeline(raw_df).clean()
    fe_df, fe_metrics = FeatureEngineeringPipeline(cleaned_df, metadata).engineer()

    assert isinstance(fe_df, pd.DataFrame)
    assert len(fe_df) > 0

    print_fe_summary(fe_df, fe_metrics)


def test_fe_kaggle_exam():
    print_section("FEATURE ENGINEERING TEST — KAGGLE EXAM DATASET")

    raw_df, _ = load_kaggle_exam_performance()
    cleaned_df, _, metadata = CleaningPipeline(raw_df).clean()
    fe_df, fe_metrics = FeatureEngineeringPipeline(cleaned_df, metadata).engineer()

    assert isinstance(fe_df, pd.DataFrame)
    assert len(fe_df) > 0

    print_fe_summary(fe_df, fe_metrics)


# -------------------------------------------------------------------
# Schema Validation Tests
# -------------------------------------------------------------------

def test_fe_schema_all():
    print_section("SCHEMA VALIDATION — ALL DATASETS")

    datasets = {
        "UCI": load_uci_dataset,
        "Kaggle Habits": load_kaggle_student_habits,
        "Kaggle Exam": load_kaggle_exam_performance,
    }

    for name, loader in datasets.items():
        print_section(f"SCHEMA VALIDATION — {name}")

        raw_df, _ = loader()
        cleaned_df, _, metadata = CleaningPipeline(raw_df).clean()
        fe_df, _ = FeatureEngineeringPipeline(cleaned_df, metadata).engineer()

        assert fe_df.columns.is_unique
        assert fe_df.isna().sum().sum() == 0

        print(f"Schema validated for {name} dataset.")


# -------------------------------------------------------------------
# NFR Tests (fe_time, max_features)
# -------------------------------------------------------------------

def test_fe_nfr_all():
    print_section("NFR TEST — FEATURE ENGINEERING MODULE (ALL DATASETS)")

    datasets = {
        "UCI": load_uci_dataset,
        "Kaggle Habits": load_kaggle_student_habits,
        "Kaggle Exam": load_kaggle_exam_performance,
    }

    for name, loader in datasets.items():
        print_section(f"NFR TEST — {name}")

        raw_df, _ = loader()
        cleaned_df, _, metadata = CleaningPipeline(raw_df).clean()
        fe_df, fe_metrics = FeatureEngineeringPipeline(cleaned_df, metadata).engineer()

        assert fe_metrics["fe_time"] <= FEATURE_NFR_THRESHOLDS["fe_time_max"]
        assert fe_metrics["total_features"] <= FEATURE_NFR_THRESHOLDS["max_features"]

        print_fe_summary(fe_df, fe_metrics)

    print("\nAll NFR tests passed for all datasets.")


# -------------------------------------------------------------------
# Direct Execution Support
# -------------------------------------------------------------------

if __name__ == "__main__":
    test_fe_uci()
    test_fe_kaggle_habits()
    test_fe_kaggle_exam()
    test_fe_schema_all()
    test_fe_nfr_all()
    print_section("ALL FEATURE ENGINEERING TESTS COMPLETED SUCCESSFULLY")
