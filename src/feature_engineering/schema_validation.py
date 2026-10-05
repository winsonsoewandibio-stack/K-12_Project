"""
schema_validation.py
--------------------
Validates feature engineering output.

Ensures:
- no duplicate columns
- no missing values
- safe for model training
"""

def validate_feature_schema(df):
    """Ensure no duplicate columns and no missing values."""
    assert df.columns.is_unique, "Duplicate columns found after FE."
    assert df.isna().sum().sum() == 0, "Missing values found after FE."
    return True
