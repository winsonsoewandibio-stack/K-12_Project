"""
FEATURE ENGINEERING PIPELINE ORCHESTRATOR

This module connects:
    • CLEANED DATA  → from src.data_cleaning
    • FEATURE RULES → from feature_rules.py
    • SCHEMA CHECKS → from schema_feature_validation.py

IMPORTANT FIX:
    We now pass CLEANED COLUMN NAMES into the schema validator.
    This prevents conditional features from being incorrectly required
    when the cleaned dataset does NOT contain the base column.
"""

import os
import pandas as pd

from src.data_cleaning.data_cleaning import (
    clean_uci,
    clean_kaggle_habits,
    clean_kaggle_exam,
)

from src.feature_engineering.feature_rules import apply_feature_rules
from src.feature_engineering.schema_feature_validation import validate_feature_schema


def engineer_features(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Apply feature engineering rules + schema validation.

    FIX:
        Pass df.columns (CLEANED columns) into validator.
        This ensures conditional features are only required
        when the CLEANED dataset contains the base column.
    """
    df_feat = apply_feature_rules(df)

    # FIX — pass CLEANED columns
    validate_feature_schema(df_feat, name, df.columns)

    return df_feat


def save_engineered(df: pd.DataFrame, name: str):
    os.makedirs("data/engineered", exist_ok=True)
    path = os.path.join("data", "engineered", f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"Saved engineered dataset to: {path}")


def print_summary(df: pd.DataFrame, name: str):
    print("\n" + "=" * 70)
    print(f"FEATURE ENGINEERING SUMMARY: {name}")
    print("=" * 70)
    print(df.head())


def main():
    uci_clean = clean_uci()
    kaggle_habits_clean = clean_kaggle_habits()
    kaggle_exam_clean = clean_kaggle_exam()

    uci_feat = engineer_features(uci_clean, "UCI Dataset")
    kaggle_habits_feat = engineer_features(kaggle_habits_clean, "Kaggle Habits Dataset")
    kaggle_exam_feat = engineer_features(kaggle_exam_clean, "Kaggle Exam Dataset")

    save_engineered(uci_feat, "uci_engineered")
    save_engineered(kaggle_habits_feat, "kaggle_habits_engineered")
    save_engineered(kaggle_exam_feat, "kaggle_exam_engineered")

    print_summary(uci_feat, "UCI Dataset")
    print_summary(kaggle_habits_feat, "Kaggle Habits Dataset")
    print_summary(kaggle_exam_feat, "Kaggle Exam Dataset")


if __name__ == "__main__":
    main()
