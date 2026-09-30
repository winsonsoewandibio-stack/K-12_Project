"""
High-level data cleaning pipeline orchestrator.

This module connects:
- ingestion outputs (UCI + Kaggle datasets)
- fairness-aligned cleaning rules
- saving cleaned datasets to disk

It does NOT define the rules themselves; those live in `cleaning_rules.py`.
It does NOT define helper utilities; those live in `cleaning_utils.py`.

Responsibilities:
- Load raw unified DataFrames from ingestion.
- Apply `apply_cleaning_rules()` to each dataset.
- Save cleaned versions to `data/cleaned/`.
- Provide functions that other modules (e.g., feature engineering) can call.
"""

import os
import pandas as pd

from src.data_ingestion.load_dataset_uci import load_uci_student_performance
from src.data_ingestion.load_dataset_kaggle_habits import load_kaggle_student_habits
from src.data_ingestion.load_dataset_kaggle_exam import load_kaggle_exam_performance

from src.data_cleaning.cleaning_rules import apply_cleaning_rules


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full fairness-aligned cleaning pipeline to a single DataFrame.

    This is a thin wrapper around `apply_cleaning_rules()`, kept here for
    readability and future extension (e.g., dataset-specific tweaks).
    """
    return apply_cleaning_rules(df)


def clean_uci() -> pd.DataFrame:
    """
    Load the UCI student performance dataset (already ingested and unified),
    then apply the cleaning pipeline.

    Returns:
        Cleaned UCI DataFrame, ready for feature engineering and modeling.
    """
    df = load_uci_student_performance()
    df_clean = clean_dataset(df)
    return df_clean


def clean_kaggle_habits() -> pd.DataFrame:
    """
    Load the Kaggle study habits dataset (already ingested and unified),
    then apply the cleaning pipeline.

    Returns:
        Cleaned Kaggle Habits DataFrame.
    """
    df = load_kaggle_student_habits()
    df_clean = clean_dataset(df)
    return df_clean


def clean_kaggle_exam() -> pd.DataFrame:
    """
    Load the Kaggle exam performance dataset (already ingested and unified),
    then apply the cleaning pipeline.

    Returns:
        Cleaned Kaggle Exam DataFrame.
    """
    df = load_kaggle_exam_performance()
    df_clean = clean_dataset(df)
    return df_clean


def save_cleaned(df: pd.DataFrame, name: str) -> None:
    """
    Save a cleaned DataFrame to the `data/cleaned/` folder as CSV.

    Parameters:
        df   : cleaned DataFrame
        name : base filename (without extension), e.g. "uci_cleaned"

    This function ensures the folder exists and prints the path for traceability.
    """
    os.makedirs("data/cleaned", exist_ok=True)
    path = os.path.join("data", "cleaned", f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"Saved cleaned dataset to: {path}")


def main():
    """
    End-to-end cleaning runner.

    When you run:

        python src/data_cleaning/data_cleaning.py

    This function will:
    - load all three datasets via ingestion
    - clean them using fairness-aligned rules
    - save them to `data/cleaned/` as CSVs
    """
    uci_clean = clean_uci()
    kaggle_habits_clean = clean_kaggle_habits()
    kaggle_exam_clean = clean_kaggle_exam()

    save_cleaned(uci_clean, "uci_cleaned")
    save_cleaned(kaggle_habits_clean, "kaggle_habits_cleaned")
    save_cleaned(kaggle_exam_clean, "kaggle_exam_cleaned")


if __name__ == "__main__":
    main()
