"""
UTILITY FUNCTIONS FOR FEATURE ENGINEERING

These helpers DO NOT encode domain logic.
They only provide reusable mathematical operations.

This keeps feature_rules.py clean, readable, and domain‑focused.
"""

import pandas as pd
import numpy as np


def safe_divide(a, b):
    """
    Safe division:
    - Avoids ZeroDivisionError
    - Avoids NaN propagation
    - Returns 0 when denominator is zero or missing

    Used for:
        • study_efficiency_score
        • missingness_ratio
        • normalized risk calculations
    """
    try:
        if b is None or b == 0 or pd.isna(b):
            return 0
        return a / b
    except Exception:
        return 0


def bucketize(series, bins, labels):
    """
    Convert numeric values into categorical buckets.

    Used for:
        • risk tiers
        • study/sleep buckets
        • teacher‑friendly categories
    """
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


def interaction(a, b):
    """
    Create numeric interaction terms.

    Used for:
        • study × sleep
        • attendance × study

    SHAP interprets interaction terms extremely well.
    """
    return a * b


def normalize_to_unit(series, max_val):
    """
    Normalize numeric values to [0,1].

    Used for:
        • opportunity scores
        • normalized interactions
    """
    return series.astype(float) / float(max_val)


def boolean_flag(condition_series):
    """
    Convert boolean conditions into 0/1 flags.

    Used for:
        • low_study_flag
        • low_sleep_flag
        • high_missingness_flag

    Boolean flags are highly SHAP‑interpretable.
    """
    return condition_series.astype(int)


def entropy_from_binary_row(row):
    """
    Compute entropy of missingness indicators.

    High entropy = inconsistent missingness
    Low entropy = predictable missingness

    Used for:
        • missingness_entropy (data health feature)
    """
    vals = row.values
    total = len(vals)
    if total == 0:
        return 0.0
    p1 = np.sum(vals) / total
    p0 = 1 - p1
    eps = 1e-9
    return -(p0 * np.log2(p0 + eps) + p1 * np.log2(p1 + eps))
