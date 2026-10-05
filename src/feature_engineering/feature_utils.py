"""
feature_utils.py
----------------
Utility functions used by the FeatureEngineeringPipeline.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# -------------------------------------------------------------------
# One-Hot Encoding (High-Cardinality Safe)
# -------------------------------------------------------------------
def encode_categorical(df: pd.DataFrame, categorical_cols: list):
    """
    One-hot encode categorical columns, excluding high-cardinality columns.

    High-cardinality columns (e.g., student_id with 100k unique values)
    would explode the feature space and violate NFR constraints.

    IMPORTANT:
    - We DO NOT drop high-cardinality columns.
    - We simply DO NOT encode them.
    - They remain in the dataset for identification, fairness slicing,
      SHAP per-student explanations, and prediction reporting.
    """
    safe_cols = []

    for col in categorical_cols:
        unique_count = df[col].nunique()

        # Skip columns with too many categories
        if unique_count <= 50:
            safe_cols.append(col)
        else:
            print(f"[WARNING] Skipping high-cardinality column: {col} ({unique_count} unique values)")

    # Only encode safe categorical columns
    return pd.get_dummies(df, columns=safe_cols, drop_first=True)


# -------------------------------------------------------------------
# Numeric Scaling
# -------------------------------------------------------------------
def scale_numeric(df: pd.DataFrame, numeric_cols: list, method="standard"):
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


# -------------------------------------------------------------------
# Interaction Features
# -------------------------------------------------------------------
def create_interaction_features(df: pd.DataFrame, numeric_cols: list):
    interactions = {}

    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            interactions[f"{col1}_plus_{col2}"] = df[col1] + df[col2]
            interactions[f"{col1}_times_{col2}"] = df[col1] * df[col2]

    return pd.DataFrame(interactions)
