"""
feature_utils.py
----------------
Utility functions for feature engineering.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------
# Categorical Encoding
# -----------------------------------------------------------
def encode_categorical(df, categorical_cols):
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].fillna("unknown")
        df[col] = df[col].astype("category")
    return df


# -----------------------------------------------------------
# Numeric Scaling
# -----------------------------------------------------------
def scale_numeric(df, numeric_cols, scaling_method):
    """
    Scales numeric columns using StandardScaler.

    ID columns MUST NOT be scaled.
    """

    # Skip ID columns
    ID_COLUMNS = ["student_id", "school_id", "class_id"]
    numeric_cols = [col for col in numeric_cols if col not in ID_COLUMNS]

    if not numeric_cols:
        return df

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


# -----------------------------------------------------------
# Interaction Features (Optimized to avoid fragmentation)
# -----------------------------------------------------------
def create_interaction_features(df, numeric_cols):
    """
    Creates pairwise interaction features for numeric columns.
    ID columns are excluded.

    Optimized to avoid DataFrame fragmentation by building
    all interaction columns in a dictionary first, then
    creating a DataFrame once.
    """

    ID_COLUMNS = ["student_id", "school_id", "class_id"]
    numeric_cols = [col for col in numeric_cols if col not in ID_COLUMNS]

    interaction_dict = {}

    # Build all interaction columns in a dict (fast, no fragmentation)
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col_i = numeric_cols[i]
            col_j = numeric_cols[j]
            interaction_name = f"{col_i}_x_{col_j}"
            interaction_dict[interaction_name] = df[col_i] * df[col_j]

    # Create DataFrame once (no fragmentation)
    interaction_df = pd.DataFrame(interaction_dict, index=df.index)

    return interaction_df
