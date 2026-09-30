"""
ENHANCED FEATURE ENGINEERING TESTS
Includes:
    • Pass/fail messages
    • Detailed feature validation display
    • Clear conditional logic reporting
    • Engineered dataset preview + statistics
    • Compatible with pytest AND normal Python execution
"""

import sys
import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_cleaning.data_cleaning import (
    clean_uci,
    clean_kaggle_habits,
    clean_kaggle_exam,
)

from src.feature_engineering.feature_engineering import engineer_features
from src.feature_engineering.schema_feature_validation import (
    MANDATORY_FEATURES,
    CONDITIONAL_FEATURES,
)


# ---------------------------------------------------------------------
# DISPLAY ENGINEERED DATA
# ---------------------------------------------------------------------
def display_engineered_data(df_engineered, dataset_name):
    print("\n" + "=" * 70)
    print(f"DATA PROCESSING RESULT — {dataset_name}")
    print("=" * 70)

    # Shape
    print(f"\nShape: {df_engineered.shape[0]} rows × {df_engineered.shape[1]} columns")

    # Columns
    print("\nColumns:")
    for col in df_engineered.columns:
        print(f"  • {col}")

    # Preview
    print("\nPreview (first 5 rows):")
    print(df_engineered.head())

    # Summary statistics (numeric only)
    print("\nSummary Statistics:")
    try:
        # Pandas 2.x
        print(df_engineered.describe(include='all', datetime_is_numeric=True))
    except TypeError:
        # Pandas 1.x fallback
        print(df_engineered.describe(include='all'))

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------
# VALIDATION + DISPLAY
# ---------------------------------------------------------------------
def assert_features_exist(df_engineered, cleaned_columns, dataset_name):
    df_columns = set(df_engineered.columns)

    print("\n" + "=" * 70)
    print(f"FEATURE VALIDATION REPORT — {dataset_name}")
    print("=" * 70)

    # ------------------------------
    # 1. Mandatory features
    # ------------------------------
    print("\nMANDATORY FEATURES:")
    missing_mandatory = []
    for feat in MANDATORY_FEATURES:
        if feat in df_columns:
            print(f"  ✔ {feat}")
        else:
            print(f"  ✘ {feat} (MISSING)")
            missing_mandatory.append(feat)

    if missing_mandatory:
        raise AssertionError(
            f"{dataset_name} missing mandatory features: {missing_mandatory}"
        )

    # ------------------------------
    # 2. Conditional features
    # ------------------------------
    print("\nCONDITIONAL FEATURES:")
    missing_conditional = []

    for base, required_features in CONDITIONAL_FEATURES.items():

        # Determine trigger condition
        if isinstance(base, str):
            trigger = base in cleaned_columns
        else:
            trigger = all(col in cleaned_columns for col in base)

        # Display trigger status
        print(f"\n  Base Column(s): {base}")
        print(f"  Triggered: {'YES' if trigger else 'NO'}")

        if not trigger:
            print("  → Skipping conditional features (base column missing)")
            continue

        # Check required features
        for feat in required_features:
            if feat in df_columns:
                print(f"    ✔ {feat}")
            else:
                print(f"    ✘ {feat} (MISSING)")
                missing_conditional.append(feat)

    if missing_conditional:
        raise AssertionError(
            f"{dataset_name} missing conditional engineered features: {missing_conditional}"
        )

    print("\n✔ ALL FEATURES VALIDATED SUCCESSFULLY\n")


# ---------------------------------------------------------------------
# PYTEST TESTS
# ---------------------------------------------------------------------
def test_feature_engineering_uci():
    df_clean = clean_uci()
    df_feat = engineer_features(df_clean, "UCI Dataset")

    display_engineered_data(df_feat, "UCI Dataset")
    assert_features_exist(df_feat, df_clean.columns, "UCI Dataset")


def test_feature_engineering_kaggle_habits():
    df_clean = clean_kaggle_habits()
    df_feat = engineer_features(df_clean, "Kaggle Habits Dataset")

    display_engineered_data(df_feat, "Kaggle Habits Dataset")
    assert_features_exist(df_feat, df_clean.columns, "Kaggle Habits Dataset")


def test_feature_engineering_kaggle_exam():
    df_clean = clean_kaggle_exam()
    df_feat = engineer_features(df_clean, "Kaggle Exam Dataset")

    display_engineered_data(df_feat, "Kaggle Exam Dataset")
    assert_features_exist(df_feat, df_clean.columns, "Kaggle Exam Dataset")


# ---------------------------------------------------------------------
# MANUAL RUNNER
# ---------------------------------------------------------------------
def manual_run():
    print("\nRunning Feature Engineering Tests (Manual Execution)\n")

    for name, cleaner in [
        ("UCI Dataset", clean_uci),
        ("Kaggle Habits Dataset", clean_kaggle_habits),
        ("Kaggle Exam Dataset", clean_kaggle_exam),
    ]:
        print(f"\n=== Testing {name} ===")
        df_clean = cleaner()
        df_feat = engineer_features(df_clean, name)

        # Show engineered dataset
        display_engineered_data(df_feat, name)

        # Validate features
        assert_features_exist(df_feat, df_clean.columns, name)

        print(f"✔ {name} passed.\n")

    print("\n✔ ALL FEATURE ENGINEERING TESTS PASSED SUCCESSFULLY\n")


if __name__ == "__main__":
    manual_run()
