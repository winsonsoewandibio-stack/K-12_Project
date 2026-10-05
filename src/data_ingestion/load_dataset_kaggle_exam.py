"""
load_dataset_kaggle_exam.py
---------------------------
Loads the Kaggle Exam dataset from the local data folder.
Adds SIS fields non-destructively (keeps all original columns).
"""

import pandas as pd

from src.data_ingestion.ingestion import DataIngestionPipeline
from src.data_ingestion.common_scheme import CANONICAL_COLUMNS, SIS_REQUIRED_FIELDS


def load_kaggle_exam_performance(
    path="data/kaggle/mobeenfatimah/student-exam-performance-and-success-dataset/student_exam_performance.csv"
):
    """
    Load Kaggle Exam dataset and apply SIS mapping.
    """

    raw_df = pd.read_csv(path)

    temp_path = "data/temp_kaggle_exam_ingestion.csv"
    raw_df.to_csv(temp_path, index=False)

    pipeline = DataIngestionPipeline(temp_path)
    df = pipeline.load()
    metrics = pipeline.metrics

    df = df.rename(columns=CANONICAL_COLUMNS)

    # SIS mapping (non-destructive)
    df["student_id"] = df["student_id"]
    df["grade_level"] = df["education_level"]
    df["attendance"] = df["attendance_percentage"]
    df["exam_score"] = df["exam_score"]

    for field in SIS_REQUIRED_FIELDS:
        if field not in df.columns:
            raise ValueError(f"SIS field missing in Kaggle Exam dataset: {field}")

    return df, metrics

