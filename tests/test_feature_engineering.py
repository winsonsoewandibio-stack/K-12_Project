"""
test_feature_engineering.py
---------------------------
This test suite validates the entire feature engineering module:

1. Functional FE correctness for all datasets.
2. Schema validation (no duplicates, no missing values).
3. NFR compliance (fe_time, max_features).
4. Human-readable output (df.head(), shape, metrics).
5. Integration with ingestion → cleaning → feature engineering pipeline.
6. Unified Feature Engineering Orchestrator test using SAFE unified metadata.

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

# Import orchestrator
from src.feature_engineering.feature_engineering_orchestrator import FeatureEngineeringOrchestrator

# Import NFR rules
from src.feature_engineering.feature_rules import FEATURE_NFR_THRESHOLDS


# -------------------------------------------------------------------
# SAFE UNIFIED METADATA BUILDER (Option A)
# -------------------------------------------------------------------

def build_safe_unified_metadata(metadata_list):
    """
    SAFE unified metadata builder:

    - Column is categorical if ANY dataset treats it as categorical
    - Column is numeric ONLY if ALL datasets treat it as numeric
    - ID columns ALWAYS categorical
    """

    categorical_sets = [set(m["categorical_columns"]) for m in metadata_list]
    numeric_sets = [set(m["numeric_columns"]) for m in metadata_list]

    # Union of categorical columns
    categorical_union = set.union(*categorical_sets)

    # Intersection of numeric columns (safe rule)
    numeric_intersection = set.intersection(*numeric_sets)

    # Final metadata
    unified_metadata = {
        "categorical_columns": list(categorical_union),
        "numeric_columns": list(numeric_intersection)
    }

    # Force ID columns to be categorical
    ID_COLUMNS = ["student_id", "school_id", "class_id"]
    for col in ID_COLUMNS:
        if col in unified_metadata["numeric_columns"]:
            unified_metadata["numeric_columns"].remove(col)
        if col not in unified_metadata["categorical_columns"]:
            unified_metadata["categorical_columns"].append(col)

    print("\n[INFO] SAFE Unified Metadata:")
    print(unified_metadata)

    return unified_metadata


# -------------------------------------------------------------------
# Helper printing utilities
# -------------------------------------------------------------------

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_fe_summary(df, metrics):
    print("\n--- FEATURE ENGINEERING SUMMARY ---")
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nHead:", df.head())
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
# NFR Tests
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
# Unified Orchestrator Test
# -------------------------------------------------------------------

def test_fe_orchestrator_unified():
    print_section("UNIFIED FEATURE ENGINEERING ORCHESTRATOR TEST")

    cleaned_datasets = {}

    # UCI dataset
    raw_uci, _ = load_uci_dataset()
    cleaned_uci, _, metadata_uci = CleaningPipeline(raw_uci).clean()
    cleaned_datasets["uci"] = cleaned_uci

    # Kaggle Habits dataset
    raw_habits, _ = load_kaggle_student_habits()
    cleaned_habits, _, metadata_habits = CleaningPipeline(raw_habits).clean()
    cleaned_datasets["habits"] = cleaned_habits

    # Kaggle Exam dataset
    raw_exam, _ = load_kaggle_exam_performance()
    cleaned_exam, _, metadata_exam = CleaningPipeline(raw_exam).clean()
    cleaned_datasets["exam"] = cleaned_exam

    # SAFE unified metadata (Option A)
    metadata = build_safe_unified_metadata([metadata_uci, metadata_habits, metadata_exam])

    orchestrator = FeatureEngineeringOrchestrator(cleaned_datasets, metadata)
    unified_df = orchestrator.run()

    assert isinstance(unified_df, pd.DataFrame)
    assert len(unified_df) > 0

    print("\nUnified Engineered Dataset:")
    print(unified_df.head())
    print(f"Shape: {unified_df.shape}")

    save_path = "artifacts/feature_engineering/engineered_dataset.csv"
    assert os.path.exists(save_path)

    print(f"[INFO] Unified engineered dataset saved at: {save_path}")


# -------------------------------------------------------------------
# Direct Execution
# -------------------------------------------------------------------

if __name__ == "__main__":
    test_fe_uci()
    test_fe_kaggle_habits()
    test_fe_kaggle_exam()
    test_fe_schema_all()
    test_fe_nfr_all()
    test_fe_orchestrator_unified()
    print_section("ALL FEATURE ENGINEERING TESTS COMPLETED SUCCESSFULLY")
