"""
load_dataset_uci.py
-------------------
Loads the UCI student performance dataset from the local data folder.
Adds SIS fields non-destructively (keeps all original columns).
"""

import pandas as pd

from src.data_ingestion.ingestion import DataIngestionPipeline
from src.data_ingestion.common_scheme import CANONICAL_COLUMNS, SIS_REQUIRED_FIELDS


def load_uci_dataset(path="data/uci/student_performance/student_performance.csv"):
    """
    Load UCI dataset and apply SIS mapping.
    """

    # Load raw dataset
    raw_df = pd.read_csv(path)

    # Save temporary file for ingestion pipeline
    temp_path = "data/temp_uci_ingestion.csv"
    raw_df.to_csv(temp_path, index=False)

    # Run ingestion pipeline
    pipeline = DataIngestionPipeline(temp_path)
    df = pipeline.load()
    metrics = pipeline.metrics

    # Canonical normalization
    df = df.rename(columns=CANONICAL_COLUMNS)

    # SIS mapping (non-destructive)
    df["student_id"] = df.index
    df["grade_level"] = df["G1"]
    df["attendance"] = df["absences"]
    df["exam_score"] = df["G3"]

    # Validate SIS schema
    for field in SIS_REQUIRED_FIELDS:
        if field not in df.columns:
            raise ValueError(f"SIS field missing in UCI dataset: {field}")

    return df, metrics

