"""
load_dataset_kaggle_habit.py
----------------------------
Loads the Kaggle Habits dataset from the local data folder.
Adds SIS fields non-destructively (keeps all original columns).
"""

import pandas as pd

from src.data_ingestion.ingestion import DataIngestionPipeline
from src.data_ingestion.common_scheme import CANONICAL_COLUMNS, SIS_REQUIRED_FIELDS


def load_kaggle_student_habits(
    path="data/kaggle/harshadapatil31/student-performance-and-study-habits-dataset/student_performance_dataset.csv"
):
    """
    Load Kaggle Habits dataset and apply SIS mapping.
    """

    raw_df = pd.read_csv(path)

    temp_path = "data/temp_kaggle_habits_ingestion.csv"
    raw_df.to_csv(temp_path, index=False)

    pipeline = DataIngestionPipeline(temp_path)
    df = pipeline.load()
    metrics = pipeline.metrics

    df = df.rename(columns=CANONICAL_COLUMNS)

    # SIS mapping (non-destructive)
    df["student_id"] = df["student_id"]
    df["grade_level"] = df["previous_grade"]
    df["attendance"] = df["attendance_percent"]
    df["exam_score"] = df["final_exam_score"]

    for field in SIS_REQUIRED_FIELDS:
        if field not in df.columns:
            raise ValueError(f"SIS field missing in Kaggle Habits dataset: {field}")

    return df, metrics
