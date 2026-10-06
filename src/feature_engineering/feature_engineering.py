"""
feature_engineering.py
----------------------
Main FeatureEngineeringPipeline.
"""

import time
import os
import pandas as pd

from src.feature_engineering.feature_utils import (
    encode_categorical,
    scale_numeric,
    create_interaction_features,
)

from src.feature_engineering.feature_rules import (
    ENCODING_RULES,
    SCALING_RULES,
    FEATURE_NFR_THRESHOLDS,
)

from src.feature_engineering.schema_validation import validate_feature_schema


class FeatureEngineeringPipeline:

    def __init__(self, df: pd.DataFrame, metadata: dict):
        self.df = df.copy()
        self.metadata = metadata

        self.metrics = {
            "fe_time": None,
            "encoded_columns": 0,
            "scaled_columns": 0,
            "interaction_features": 0,
            "total_features": None,
        }

    def engineer(self):
        start = time.time()

        # 1. Encode categorical variables
        categorical_cols = self.metadata["categorical_columns"]
        if categorical_cols:
            self.df = encode_categorical(self.df, categorical_cols)
            self.metrics["encoded_columns"] = len(categorical_cols)

        # 2. Scale numeric variables
        numeric_cols = self.metadata["numeric_columns"]
        if numeric_cols:
            self.df = scale_numeric(self.df, numeric_cols, SCALING_RULES["numeric_scaling"])
            self.metrics["scaled_columns"] = len(numeric_cols)

        # 3. Interaction features
        interaction_df = create_interaction_features(self.df, numeric_cols)
        self.df = pd.concat([self.df, interaction_df], axis=1)
        self.metrics["interaction_features"] = interaction_df.shape[1]

        # 4. Schema validation
        validate_feature_schema(self.df)

        # 5. NFR metrics
        end = time.time()
        self.metrics["fe_time"] = end - start
        self.metrics["total_features"] = self.df.shape[1]

        assert self.metrics["fe_time"] <= FEATURE_NFR_THRESHOLDS["fe_time_max"], \
            f"FE time exceeded NFR limit ({self.metrics['fe_time']}s > {FEATURE_NFR_THRESHOLDS['fe_time_max']}s)."

        assert self.metrics["total_features"] <= FEATURE_NFR_THRESHOLDS["max_features"], \
            f"Too many features created ({self.metrics['total_features']} > {FEATURE_NFR_THRESHOLDS['max_features']})."

        # -----------------------------------------------------------
        # 6. Save engineered dataset for downstream modules
        # -----------------------------------------------------------
        output_dir = "artifacts/feature_engineering/"
        os.makedirs(output_dir, exist_ok=True)

        save_path = output_dir + "engineered_dataset.csv"
        self.df.to_csv(save_path, index=False)

        print(f"[INFO] Engineered dataset saved to: {save_path}")
        print(f"[INFO] Total features: {self.metrics['total_features']}")

        return self.df, self.metrics
