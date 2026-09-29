# tests/test_data_ingestion.py

import sys
import os

# Add project root to Python path automatically
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.data_ingestion.common_scheme import UNIFIED_SCHEMA, convert_letter_grade
from src.data_ingestion.load_dataset_uci import load_uci_student_performance
from src.data_ingestion.load_dataset_kaggle_habits import load_kaggle_student_habits
from src.data_ingestion.load_dataset_kaggle_exam import load_kaggle_exam_performance


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def assert_schema(df):
    """Ensure the DataFrame contains ALL unified schema columns."""
    missing = [col for col in UNIFIED_SCHEMA if col not in df.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"


def test_uci_ingestion():
    print_section("TESTING UCI STUDENT PERFORMANCE INGESTION (Unified Schema)")

    df = load_uci_student_performance()

    print("DataFrame shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # Unified schema check
    assert_schema(df)

    # UCI-specific fields must NOT be None
    assert df["gender"].notnull().all()
    assert df["mother_education"].notnull().all()
    assert df["father_education"].notnull().all()
    assert df["weekly_study_time"].notnull().all()
    assert df["alcohol_use_workday"].notnull().all()
    assert df["alcohol_use_weekend"].notnull().all()
    assert df["final_exam_score"].notnull().all()

    # Fields UCI does NOT have must be None
    assert df["sleep_quality"].isnull().all()
    assert df["exam_anxiety_level"].isnull().all()

    print("✔ UCI ingestion passed.")


def test_kaggle_habits_ingestion():
    print_section("TESTING KAGGLE STUDY HABITS INGESTION (Unified Schema)")

    df = load_kaggle_student_habits()

    print("DataFrame shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # Unified schema check
    assert_schema(df)

    # Kaggle Habits-specific fields must NOT be None
    assert df["study_time_hours"].notnull().all()
    assert df["attendance_percentage"].notnull().any(), \
    "attendance_percentage should have at least some non-null values"
    assert df["previous_grade_letter"].notnull().all()
    assert df["final_exam_score"].notnull().all()

    # Grade conversion check
    assert df["previous_grade_numeric"].apply(lambda x: x in [50, 60, 70, 80, 90]).any()
    assert df["final_grade_numeric"].apply(lambda x: x in [50, 60, 70, 80, 90]).any()

    # Fields Kaggle Habits does NOT have must be None
    assert df["G1"].isnull().all()
    assert df["exam_anxiety_level"].isnull().all()

    print("✔ Kaggle Study Habits ingestion passed.")


def test_kaggle_exam_ingestion():
    print_section("TESTING KAGGLE EXAM PERFORMANCE INGESTION (Unified Schema)")

    df = load_kaggle_exam_performance()

    print("DataFrame shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # Unified schema check
    assert_schema(df)

    # Kaggle Exam-specific fields must NOT be None
    assert df["study_hours_per_day"].notnull().any()
    assert df["attendance_percentage"].notnull().any()
    assert df["previous_exam_score"].notnull().any()
    assert df["final_exam_score"].notnull().any()
    assert df["performance_level"].notnull().any()

    # Grade conversion check
    assert df["final_grade_numeric"].apply(lambda x: x in [50, 60, 70, 80, 90]).any()

    # Fields Kaggle Exam does NOT have must be None
    assert df["G1"].isnull().all()
    assert df["weekly_study_time"].isnull().all()

    print("✔ Kaggle Exam Performance ingestion passed.")


# ---------------------------------------------------------
# MAIN RUNNER — allows running without pytest
# ---------------------------------------------------------
if __name__ == "__main__":
    test_uci_ingestion()
    test_kaggle_habits_ingestion()
    test_kaggle_exam_ingestion()

    print("\nAll ingestion tests completed successfully.")
