"""
FINAL SCHEMA VALIDATION — ALIGNED WITH FEATURE RULES & CLEANING

MANDATORY FEATURES:
    Only features guaranteed to exist for ALL datasets.

CONDITIONAL FEATURES:
    Required ONLY if the CLEANED dataset contains the same trigger
    that feature_rules.py uses to create them.
"""

# ============================================================
# MANDATORY FEATURES — must exist for ALL datasets
# ============================================================

MANDATORY_FEATURES = {
    "attendance_risk_score",
    "exam_score_normalized",
    "performance_risk_score",
    "engagement_composite",
    "performance_composite",
    "internet_access_binary",
    "missingness_total",
    "missingness_ratio",
    "missingness_entropy",
    "missingness_max",
    "missingness_variance",
    "high_missingness_flag",
    "gender_missing_flag",
    "gender_neutral",
    "overall_risk_score",
}

# ============================================================
# CONDITIONAL FEATURES — required only if trigger exists
# ============================================================

CONDITIONAL_FEATURES = {
    # Study-based features (triggered by base column)
    "study_hours_per_day": [
        "study_consistency_score",
        "study_hours_rolling_mean_3",
        "study_hours_delta",
        "study_stability_flag",
        "study_hours_bucket",
        "study_category",
        "low_study_flag",
        "high_study_flag",
        "study_efficiency_score",
        "study_efficiency_tier",
        "study_opportunity_score",
    ],

    # Sleep-based features (triggered by base column)
    "sleep_hours": [
        "sleep_consistency_score",
        "sleep_hours_rolling_mean_3",
        "sleep_hours_delta",
        "sleep_stability_flag",
        "sleep_hours_bucket",
        "sleep_category",
        "low_sleep_flag",
        "high_sleep_flag",
        "sleep_opportunity_score",
    ],

    # Attendance-based features (triggered by base column)
    "attendance_percentage": [
        "attendance_rolling_mean_5",
        "attendance_delta",
        "attendance_category",
        "attendance_opportunity_score",
    ],

    # LOW-QUALITY FLAGS — triggered by *_missing columns
    "study_hours_per_day_missing": [
        "study_hours_per_day_low_quality_flag",
    ],
    "sleep_hours_missing": [
        "sleep_hours_low_quality_flag",
    ],
    "attendance_percentage_missing": [
        "attendance_percentage_low_quality_flag",
    ],

    # Interaction features
    ("study_hours_per_day", "sleep_hours"): [
        "study_x_sleep_interaction",
        "study_x_sleep_norm",
    ],
    ("attendance_normalized", "study_hours_per_day"): [
        "attendance_x_study_interaction",
        "attendance_x_study_norm",
    ],
}


def validate_feature_schema(df_engineered, dataset_name, cleaned_columns):
    """
    Validate mandatory + conditional engineered features.

    IMPORTANT:
        For low-quality flags, we follow feature_rules.py:
        they are required ONLY if <base>_missing exists in CLEANED data.
    """

    df_columns = set(df_engineered.columns)

    # 1. Mandatory features
    missing_mandatory = MANDATORY_FEATURES - df_columns
    if missing_mandatory:
        raise AssertionError(
            f"{dataset_name} missing mandatory engineered features: {missing_mandatory}"
        )

    # 2. Conditional features
    missing_conditional = set()

    for base, required_features in CONDITIONAL_FEATURES.items():

        # Single-column trigger
        if isinstance(base, str):
            if base in cleaned_columns:
                for feat in required_features:
                    if feat not in df_columns:
                        missing_conditional.add(feat)

        # Multi-column trigger (interaction)
        elif isinstance(base, tuple):
            if all(col in cleaned_columns for col in base):
                for feat in required_features:
                    if feat not in df_columns:
                        missing_conditional.add(feat)

    if missing_conditional:
        raise AssertionError(
            f"{dataset_name} missing conditional engineered features: {missing_conditional}"
        )

    return True

