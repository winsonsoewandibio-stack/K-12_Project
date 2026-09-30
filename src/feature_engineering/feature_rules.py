"""
FEATURE ENGINEERING RULES

This file defines WHAT features we create and WHY.

All features must be:
    • interpretable
    • SHAP‑friendly
    • fairness‑aligned
    • non‑leaky
    • aligned with CLEANED dataset schema
    • teacher‑friendly
"""

import pandas as pd
from src.feature_engineering.feature_utils import (
    safe_divide,
    bucketize,
    interaction,
    normalize_to_unit,
    boolean_flag,
    entropy_from_binary_row,
)

# ============================================================
# ENGAGEMENT FEATURES
# ============================================================

def create_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engagement features measure consistency and participation.
    These are core to early‑warning systems.

    IMPORTANT:
    Only create features if the CLEANED dataset contains the base column.
    """
    df = df.copy()

    # Attendance risk = inverse of normalized attendance
    if "attendance_normalized" in df.columns:
        df["attendance_risk_score"] = 1 - df["attendance_normalized"]

    # Study consistency (rolling std)
    if "study_hours_per_day" in df.columns:
        df["study_consistency_score"] = (
            df["study_hours_per_day"].rolling(3, min_periods=1).std().fillna(0)
        )

        df["study_hours_rolling_mean_3"] = (
            df["study_hours_per_day"].rolling(3, min_periods=1).mean()
        )

        df["study_hours_delta"] = df["study_hours_per_day"].diff().fillna(0)

        df["study_stability_flag"] = boolean_flag(
            df["study_consistency_score"] < 1.0
        )

    # Sleep consistency
    if "sleep_hours" in df.columns:
        df["sleep_consistency_score"] = (
            df["sleep_hours"].rolling(3, min_periods=1).std().fillna(0)
        )

        df["sleep_hours_rolling_mean_3"] = (
            df["sleep_hours"].rolling(3, min_periods=1).mean()
        )

        df["sleep_hours_delta"] = df["sleep_hours"].diff().fillna(0)

        df["sleep_stability_flag"] = boolean_flag(
            df["sleep_consistency_score"] < 1.0
        )

    # Attendance trend
    if "attendance_percentage" in df.columns:
        df["attendance_rolling_mean_5"] = (
            df["attendance_percentage"].rolling(5, min_periods=1).mean()
        )
        df["attendance_delta"] = df["attendance_percentage"].diff().fillna(0)

    return df


# ============================================================
# PERFORMANCE FEATURES
# ============================================================

def create_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performance features measure academic outcomes.
    """
    df = df.copy()

    # Normalize exam score
    if "final_exam_score" in df.columns:
        df["exam_score_normalized"] = df["final_exam_score"] / 100.0

    # Study efficiency = score per hour
    if "study_hours_per_day" in df.columns and "final_exam_score" in df.columns:
        df["study_efficiency_score"] = df.apply(
            lambda row: safe_divide(row["final_exam_score"], row["study_hours_per_day"]),
            axis=1,
        )

    # Performance risk = inverse of normalized score
    if "exam_score_normalized" in df.columns:
        df["performance_risk_score"] = 1 - df["exam_score_normalized"]

    # Risk tiers (categorical)
    if "attendance_risk_score" in df.columns:
        df["attendance_risk_tier"] = bucketize(
            df["attendance_risk_score"],
            bins=[0.0, 0.33, 0.66, 1.0],
            labels=["Low", "Medium", "High"],
        )

    if "performance_risk_score" in df.columns:
        df["performance_risk_tier"] = bucketize(
            df["performance_risk_score"],
            bins=[0.0, 0.33, 0.66, 1.0],
            labels=["Low", "Medium", "High"],
        )

    if "study_efficiency_score" in df.columns:
        df["study_efficiency_tier"] = bucketize(
            df["study_efficiency_score"],
            bins=[0, 20, 50, 100, 200],
            labels=["Inefficient", "Balanced", "Efficient", "Very Efficient"],
        )

    # Composite scores
    engagement_parts = []
    if "attendance_risk_score" in df.columns:
        engagement_parts.append(df["attendance_risk_score"])
    if "study_consistency_score" in df.columns:
        engagement_parts.append(df["study_consistency_score"])
    if "sleep_consistency_score" in df.columns:
        engagement_parts.append(df["sleep_consistency_score"])

    if engagement_parts:
        df["engagement_composite"] = sum(engagement_parts)

    performance_parts = []
    if "performance_risk_score" in df.columns:
        performance_parts.append(df["performance_risk_score"])
    if "study_efficiency_score" in df.columns:
        performance_parts.append(df["study_efficiency_score"])

    if performance_parts:
        df["performance_composite"] = sum(performance_parts)

    return df


# ============================================================
# BEHAVIORAL FEATURES
# ============================================================

def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Behavioral features capture habits and patterns.
    """
    df = df.copy()

    # Study buckets + teacher-friendly categories
    if "study_hours_per_day" in df.columns:
        df["study_hours_bucket"] = bucketize(
            df["study_hours_per_day"],
            bins=[0, 2, 4, 8, 24],
            labels=["Very Low", "Low", "Medium", "High"],
        )

        df["study_category"] = pd.cut(
            df["study_hours_per_day"],
            bins=[0, 2, 6, 24],
            labels=["Understudying", "Balanced", "Overstudying"],
        )

        df["low_study_flag"] = boolean_flag(df["study_hours_per_day"] < 2)
        df["high_study_flag"] = boolean_flag(df["study_hours_per_day"] > 6)

    # Sleep buckets + teacher-friendly categories
    if "sleep_hours" in df.columns:
        df["sleep_hours_bucket"] = bucketize(
            df["sleep_hours"],
            bins=[0, 4, 6, 8, 24],
            labels=["Very Low", "Low", "Medium", "High"],
        )

        df["sleep_category"] = pd.cut(
            df["sleep_hours"],
            bins=[0, 6, 9, 24],
            labels=["Sleep-Deprived", "Healthy Sleep", "Oversleeping"],
        )

        df["low_sleep_flag"] = boolean_flag(df["sleep_hours"] < 6)
        df["high_sleep_flag"] = boolean_flag(df["sleep_hours"] > 9)

    # Attendance category
    if "attendance_percentage" in df.columns:
        df["attendance_category"] = pd.cut(
            df["attendance_percentage"],
            bins=[0, 70, 90, 100],
            labels=["Poor", "Moderate", "Strong"],
        )

    # Internet access binary
    if "internet_access" in df.columns:
        df["internet_access_binary"] = (
            df["internet_access"].map({"Yes": 1, "No": 0}).fillna(0)
        )

    return df


# ============================================================
# MISSINGNESS & DATA HEALTH FEATURES
# ============================================================

def create_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missingness features measure data quality.

    IMPORTANT:
    Only create low‑quality flags if CLEANING created <col>_missing.
    """
    df = df.copy()

    missing_cols = [c for c in df.columns if c.endswith("_missing")]

    if missing_cols:
        df["missingness_total"] = df[missing_cols].sum(axis=1)
        df["missingness_ratio"] = df.apply(
            lambda row: safe_divide(row["missingness_total"], len(missing_cols)),
            axis=1,
        )

        df["missingness_entropy"] = df[missing_cols].apply(
            entropy_from_binary_row, axis=1
        )
        df["missingness_max"] = df[missing_cols].max(axis=1)
        df["missingness_variance"] = df[missing_cols].var(axis=1)

        df["high_missingness_flag"] = boolean_flag(df["missingness_ratio"] > 0.3)

        # Low-quality flags (conditional)
        for base in ["study_hours_per_day", "sleep_hours", "attendance_percentage"]:
            col = f"{base}_missing"
            if col in df.columns:
                df[f"{base}_low_quality_flag"] = df[col].astype(int)

    return df


# ============================================================
# INTERACTION FEATURES
# ============================================================

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interaction features capture relationships between behaviors.
    """
    df = df.copy()

    if "study_hours_per_day" in df.columns and "sleep_hours" in df.columns:
        df["study_x_sleep_interaction"] = interaction(
            df["study_hours_per_day"], df["sleep_hours"]
        )
        df["study_x_sleep_norm"] = interaction(
            normalize_to_unit(df["study_hours_per_day"], 24),
            normalize_to_unit(df["sleep_hours"], 24),
        )

    if "attendance_normalized" in df.columns and "study_hours_per_day" in df.columns:
        df["attendance_x_study_interaction"] = interaction(
            df["attendance_normalized"], df["study_hours_per_day"]
        )
        df["attendance_x_study_norm"] = interaction(
            df["attendance_normalized"],
            normalize_to_unit(df["study_hours_per_day"], 24),
        )

    return df


# ============================================================
# FAIRNESS & OPPORTUNITY FEATURES
# ============================================================

def create_fairness_and_opportunity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Opportunity features frame behavior positively.
    Fairness guardrails prevent bias.
    """
    df = df.copy()

    if "study_hours_per_day" in df.columns:
        df["study_opportunity_score"] = normalize_to_unit(df["study_hours_per_day"], 8)

    if "sleep_hours" in df.columns:
        df["sleep_opportunity_score"] = normalize_to_unit(df["sleep_hours"], 8)

    if "attendance_percentage" in df.columns:
        df["attendance_opportunity_score"] = normalize_to_unit(
            df["attendance_percentage"], 100
        )

    if "gender" in df.columns:
        df["gender_missing_flag"] = boolean_flag(df["gender"] == "Unknown")
        df["gender_neutral"] = "Student"

    return df


# ============================================================
# OVERALL RISK SCORE
# ============================================================

def create_overall_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overall risk score combines:
        • engagement composite
        • performance composite
        • missingness ratio
        • behavioral flags

    This is the MAIN SHAP target.
    """
    df = df.copy()

    components = []

    for col in [
        "engagement_composite",
        "performance_composite",
        "missingness_ratio",
        "high_missingness_flag",
        "low_study_flag",
        "low_sleep_flag",
    ]:
        if col in df.columns:
            components.append(df[col].astype(float))

    if components:
        df["overall_risk_score"] = sum(components)

    return df


# ============================================================
# FULL PIPELINE
# ============================================================

def apply_feature_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ALL feature engineering steps in order.
    """
    df = df.copy()

    df = create_engagement_features(df)
    df = create_performance_features(df)
    df = create_behavioral_features(df)
    df = create_missingness_features(df)
    df = create_interaction_features(df)
    df = create_fairness_and_opportunity_features(df)
    df = create_overall_risk_score(df)

    return df
