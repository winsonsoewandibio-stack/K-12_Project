"""
training_utils.py
-----------------
Utility functions for model training:
- feature selection
- train/test split
- model creation (regression or classification)
- NFR checks
"""

from typing import Tuple, Dict, Any
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.model_training.evaluation_metrics import compute_classification_metrics
from src.model_training.training_rules import TRAINING_NFR_THRESHOLDS


ID_COLUMNS = ["student_id", "school_id", "class_id"]
NON_FEATURE_COLUMNS = ID_COLUMNS + ["source_dataset"]


def prepare_features_and_target(df: pd.DataFrame, target_column: str):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    drop_cols = [col for col in NON_FEATURE_COLUMNS if col in df.columns]
    drop_cols.append(target_column)

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_column]

    X = X.select_dtypes(include=["number"])
    return X, y


def is_regression_target(target_column: str) -> bool:
    """
    Explicit rule:
    - UCI → g3 → regression
    - Exam → exam_score → regression
    - Habits → final_grade → classification
    """
    return target_column in ["g3", "exam_score"]


def split_train_test(X, y, target_column, test_size=0.2, random_state=42):
    """
    Regression → no stratification
    Classification → stratification
    """

    if is_regression_target(target_column):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def create_model(target_column: str):
    """
    Automatically choose classifier or regressor based on target type.
    """

    if is_regression_target(target_column):
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
    else:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )


def train_and_evaluate_model(df, target_column):
    X, y = prepare_features_and_target(df, target_column)

    num_features = X.shape[1]
    if num_features > TRAINING_NFR_THRESHOLDS["max_features"]:
        raise ValueError(f"Too many features: {num_features}")

    X_train, X_test, y_train, y_test = split_train_test(X, y, target_column)

    model = create_model(target_column)

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    training_time = end - start

    if training_time > TRAINING_NFR_THRESHOLDS["training_time_max"]:
        raise RuntimeError("Training time exceeded NFR limit")

    y_pred = model.predict(X_test)

    if is_regression_target(target_column):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5

        metrics = {
            "rmse": rmse,
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "training_time": training_time,
            "num_features": num_features,
        }
    else:
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        metrics["training_time"] = training_time
        metrics["num_features"] = num_features

    return model, metrics
