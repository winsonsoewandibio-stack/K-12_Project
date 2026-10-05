"""
feature_rules.py
----------------
Defines universal feature engineering rules and NFR thresholds.
"""

ENCODING_RULES = {
    "categorical_encoding": "onehot",
}

SCALING_RULES = {
    "numeric_scaling": "standard",
}

FEATURE_NFR_THRESHOLDS = {
    "fe_time_max": 30.0,   # FE time limit
    "max_features": 600,   # increased from 500 → 600
}
