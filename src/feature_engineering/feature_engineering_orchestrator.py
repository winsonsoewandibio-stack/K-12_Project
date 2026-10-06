"""
feature_engineering_orchestrator.py
-----------------------------------
Runs feature engineering across ALL cleaned datasets,
adds safe dummy columns for missing features,
aligns schemas, and produces ONE unified engineered dataset.

This version includes SAFE FILE WRITING to avoid Windows
PermissionError when overwriting CSV files.
"""

import os
import pandas as pd
import tempfile
import shutil

from src.feature_engineering.feature_engineering import FeatureEngineeringPipeline


# -----------------------------------------------------------
# SAFE DUMMY COLUMN HANDLER
# -----------------------------------------------------------
def add_missing_columns(df, metadata):
    """
    Ensures the dataset contains ALL columns required by metadata.
    Missing columns are filled with safe dummy values.

    - Categorical → "unknown"
    - Numeric → 0

    Also ensures ID columns are treated as strings.
    """

    # Convert ID columns to string
    ID_COLUMNS = ["student_id", "school_id", "class_id"]
    for col in ID_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Add missing categorical columns
    for col in metadata["categorical_columns"]:
        if col not in df.columns:
            df[col] = "unknown"

    # Add missing numeric columns
    for col in metadata["numeric_columns"]:
        if col not in df.columns:
            df[col] = 0

    return df


class FeatureEngineeringOrchestrator:

    def __init__(self, cleaned_datasets: dict, metadata: dict):
        self.cleaned_datasets = cleaned_datasets
        self.metadata = metadata

    def run(self):
        engineered_frames = []

        # -----------------------------------------------------------
        # 1. Run FE on each dataset
        # -----------------------------------------------------------
        for name, df in self.cleaned_datasets.items():
            print(f"[INFO] Running FE for dataset: {name}")

            df = add_missing_columns(df, self.metadata)

            fe_pipeline = FeatureEngineeringPipeline(df, self.metadata)
            df_feat, metrics = fe_pipeline.engineer()

            df_feat["source_dataset"] = name
            engineered_frames.append(df_feat)

        # -----------------------------------------------------------
        # 2. Align schemas
        # -----------------------------------------------------------
        unified_df = pd.concat(engineered_frames, axis=0, ignore_index=True)

        print(f"[INFO] Unified engineered dataset shape: {unified_df.shape}")

        # -----------------------------------------------------------
        # 3. SAFE FILE WRITE (atomic replace)
        # -----------------------------------------------------------
        output_dir = "artifacts/feature_engineering/"
        os.makedirs(output_dir, exist_ok=True)

        final_path = os.path.join(output_dir, "engineered_dataset.csv")

        # Write to a temporary file first
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            unified_df.to_csv(tmp.name, index=False)
            temp_path = tmp.name

        # Atomically replace the old file
        shutil.move(temp_path, final_path)

        print(f"[INFO] Unified engineered dataset saved to: {final_path}")

        return unified_df

