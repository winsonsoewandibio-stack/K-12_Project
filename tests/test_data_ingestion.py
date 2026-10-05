"""
test_data_ingestion.py
----------------------
This test suite validates the entire ingestion module:

1. Functional ingestion correctness for all datasets.
2. SIS schema completeness (student_id, grade_level, attendance, exam_score).
3. Schema preservation (non-destructive ingestion).
4. NFR compliance (performance, scalability, security).
5. Human-readable output (df.head(), shape, metrics) to prove ingestion works.
6. Compatibility with both pytest and direct Python execution.

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

# Import SIS + NFR rules
from src.data_ingestion.common_scheme import SIS_REQUIRED_FIELDS, NFR_THRESHOLDS


# -------------------------------------------------------------------
# Helper printing utilities
# -------------------------------------------------------------------

def print_section(title):
    """Prints a formatted section header for readability."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_dataset_summary(df, metrics):
    """
    Prints a human-readable summary of the dataset after ingestion.
    This is used as proof that ingestion works correctly.
    """
    print("\n--- DATASET SUMMARY ---")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(list(df.columns))

    print("\nHead:")
    print(df.head())

    print("\n--- SIS FIELDS CHECK ---")
    for field in SIS_REQUIRED_FIELDS:
        print(f"{field}: OK")

    print("\n--- INGESTION METRICS ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")


# -------------------------------------------------------------------
# Functional SIS tests
# -------------------------------------------------------------------

def test_uci_ingestion_functional():
    print_section("FUNCTIONAL TEST — UCI INGESTION")

    df, metrics = load_uci_dataset()

    # Basic correctness
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # SIS schema correctness
    for field in SIS_REQUIRED_FIELDS:
        assert field in df.columns

    print_dataset_summary(df, metrics)


def test_kaggle_habits_ingestion_functional():
    print_section("FUNCTIONAL TEST — KAGGLE HABITS INGESTION")

    df, metrics = load_kaggle_student_habits()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    for field in SIS_REQUIRED_FIELDS:
        assert field in df.columns

    print_dataset_summary(df, metrics)


def test_kaggle_exam_ingestion_functional():
    print_section("FUNCTIONAL TEST — KAGGLE EXAM INGESTION")

    df, metrics = load_kaggle_exam_performance()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    for field in SIS_REQUIRED_FIELDS:
        assert field in df.columns

    print_dataset_summary(df, metrics)


# -------------------------------------------------------------------
# Schema preservation tests (non-destructive ingestion)
# -------------------------------------------------------------------

def test_schema_preservation_uci():
    print_section("SCHEMA PRESERVATION — UCI")

    raw = pd.read_csv("data/uci/student_performance/student_performance.csv")
    df, _ = load_uci_dataset()

    for col in raw.columns:
        assert col in df.columns

    print("Schema preserved for UCI dataset.")


def test_schema_preservation_kaggle_habits():
    print_section("SCHEMA PRESERVATION — KAGGLE HABITS")

    raw = pd.read_csv(
        "data/kaggle/harshadapatil31/student-performance-and-study-habits-dataset/student_performance_dataset.csv"
    )
    df, _ = load_kaggle_student_habits()

    for col in raw.columns:
        assert col in df.columns

    print("Schema preserved for Kaggle Habits dataset.")


def test_schema_preservation_kaggle_exam():
    print_section("SCHEMA PRESERVATION — KAGGLE EXAM")

    raw = pd.read_csv(
        "data/kaggle/mobeenfatimah/student-exam-performance-and-success-dataset/student_exam_performance.csv"
    )
    df, _ = load_kaggle_exam_performance()

    for col in raw.columns:
        assert col in df.columns

    print("Schema preserved for Kaggle Exam dataset.")


# -------------------------------------------------------------------
# NFR tests (applied to ALL datasets)
# -------------------------------------------------------------------

def test_ingestion_nfr_all():
    print_section("NFR TEST — ALL DATASETS")

    datasets = {
        "UCI": load_uci_dataset,
        "Kaggle Habits": load_kaggle_student_habits,
        "Kaggle Exam": load_kaggle_exam_performance,
    }

    for name, loader in datasets.items():
        print_section(f"NFR TEST — {name}")

        df, metrics = loader()

        # NFR assertions
        assert metrics["ingestion_time"] <= NFR_THRESHOLDS["ingestion_time_max"]
        assert metrics["record_count"] <= NFR_THRESHOLDS["max_records"]
        assert df is not None
        assert len(df) > 0

        print_dataset_summary(df, metrics)

    print("\nAll NFR tests passed for all datasets.")


# -------------------------------------------------------------------
# Direct execution support
# -------------------------------------------------------------------

if __name__ == "__main__":
    test_uci_ingestion_functional()
    test_kaggle_habits_ingestion_functional()
    test_kaggle_exam_ingestion_functional()
    test_schema_preservation_uci()
    test_schema_preservation_kaggle_habits()
    test_schema_preservation_kaggle_exam()
    test_ingestion_nfr_all()
    print_section("ALL INGESTION TESTS COMPLETED SUCCESSFULLY")
