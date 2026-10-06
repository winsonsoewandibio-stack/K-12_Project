"""
training_pipeline.py
--------------------
End-to-end training pipeline for a SINGLE dataset at a time.
"""

import os
import pandas as pd

from src.data_ingestion.load_dataset_uci import load_uci_dataset
from src.data_ingestion.load_dataset_kaggle_habit import load_kaggle_student_habits
from src.data_ingestion.load_dataset_kaggle_exam import load_kaggle_exam_performance

from src.data_cleaning.data_cleaning import CleaningPipeline
from src.feature_engineering.feature_engineering import FeatureEngineeringPipeline

from src.model_training.training_utils import train_and_evaluate_model
from src.model_training.model_registry import save_model


def get_dataset_loader(dataset_name: str):
    loaders = {
        "uci": load_uci_dataset,
        "habits": load_kaggle_student_habits,
        "exam": load_kaggle_exam_performance,
    }
    return loaders[dataset_name]


def get_target_column(dataset_name: str, df: pd.DataFrame) -> str:
    if dataset_name == "uci":
        if "g3" in df.columns:
            return "g3"
        raise ValueError("Could not detect target column for UCI dataset.")

    if dataset_name == "habits":
        if "final_grade" in df.columns:
            return "final_grade"
        raise ValueError("Target column 'final_grade' not found.")

    if dataset_name == "exam":
        if "exam_score" in df.columns:
            return "exam_score"
        raise ValueError("Target column 'exam_score' not found.")

    raise ValueError(f"No target detection rule for dataset: {dataset_name}")


def run_training_pipeline(dataset_name: str):
    print(f"\n[INFO] Starting training pipeline for dataset: {dataset_name}")

    loader = get_dataset_loader(dataset_name)

    raw_df, _ = loader()
    print(f"[INFO] Raw {dataset_name} dataset shape: {raw_df.shape}")

    cleaned_df, _, metadata = CleaningPipeline(raw_df).clean()
    print(f"[INFO] Cleaned {dataset_name} dataset shape: {cleaned_df.shape}")

    fe_df, fe_metrics = FeatureEngineeringPipeline(cleaned_df, metadata).engineer()
    print(f"[INFO] Engineered {dataset_name} dataset shape: {fe_df.shape}")
    print(f"[INFO] FE metrics: {fe_metrics}")

    target_column = get_target_column(dataset_name, fe_df)
    print(f"[INFO] Detected target column: {target_column}")

    model, metrics = train_and_evaluate_model(fe_df, target_column)

    print(f"[INFO] Training metrics for {dataset_name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    save_model(model, dataset_name, metrics)

    print(f"[INFO] Training pipeline for dataset '{dataset_name}' completed.")
