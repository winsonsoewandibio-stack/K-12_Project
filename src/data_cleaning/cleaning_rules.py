"""
cleaning_rules.py
-----------------
This module defines the universal rules and thresholds used by the
CleaningPipeline. These rules ensure consistency across datasets and
prepare the data for feature engineering.

The rules here DO NOT perform cleaning themselves; they only define
policy. The actual cleaning logic lives in cleaning_utils.py and
data_cleaning.py.
"""

# -------------------------------------------------------------------
# Missing Value Handling Rules
# -------------------------------------------------------------------
# Feature engineering requires NO missing values. We define type-based
# strategies so the pipeline can fill missing values consistently.
MISSING_VALUE_STRATEGIES = {
    "numeric": "median",       # Median is robust to outliers
    "categorical": "mode",     # Mode preserves category distribution
    "boolean": "mode",
}

# -------------------------------------------------------------------
# Numeric Anomaly Rules
# -------------------------------------------------------------------
# Negative values break scaling, log transforms, and ratios.
# We enforce a minimum threshold of 0 for all numeric columns.
NUMERIC_ANOMALY_RULES = {
    "min_value": 0,
    "max_value": 1e9,          # Arbitrary sanity upper bound
}

# -------------------------------------------------------------------
# Column Normalization Rules
# -------------------------------------------------------------------
# Feature engineering requires consistent column names across datasets.
COLUMN_NORMALIZATION_RULES = {
    "strip_whitespace": True,
    "lowercase_columns": True,
}

# -------------------------------------------------------------------
# Cleaning NFR Thresholds
# -------------------------------------------------------------------
# These thresholds ensure the cleaning pipeline meets performance
# and scalability requirements.
CLEANING_NFR_THRESHOLDS = {
    "cleaning_time_max": 5.0,      # seconds
    "max_records": 100000,         # scalability limit
}
