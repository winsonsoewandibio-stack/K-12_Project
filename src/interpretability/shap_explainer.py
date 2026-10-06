"""
shap_explainer.py
-----------------
Final stable SHAP module for the K-12 project.

- Regression datasets use TreeExplainer (fast)
- Multi-class HABITS dataset uses KernelExplainer (stable)
- HABITS produces JSON feature-importance percentages (no SHAP plots)
"""

import os
import json
from typing import Tuple

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.model_training.model_registry import load_model
from src.model_training.training_utils import is_regression_target
from src.feature_engineering.feature_engineering import FeatureEngineeringPipeline
from src.data_cleaning.data_cleaning import CleaningPipeline
from src.data_ingestion.load_dataset_uci import load_uci_dataset
from src.data_ingestion.load_dataset_kaggle_habit import load_kaggle_student_habits
from src.data_ingestion.load_dataset_kaggle_exam import load_kaggle_exam_performance


DATASET_LOADERS = {
    "uci": load_uci_dataset,
    "habits": load_kaggle_student_habits,
    "exam": load_kaggle_exam_performance,
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_engineered_features(dataset_name: str) -> Tuple[pd.DataFrame, str]:
    loader = DATASET_LOADERS[dataset_name]
    raw_df, _ = loader()

    cleaned_df, _, metadata = CleaningPipeline(raw_df).clean()
    fe_df, fe_metrics = FeatureEngineeringPipeline(cleaned_df, metadata).engineer()

    artifacts_dir = os.path.join("artifacts", "interpretability", "shap", dataset_name)
    _ensure_dir(artifacts_dir)

    with open(os.path.join(artifacts_dir, "fe_metrics.json"), "w") as f:
        json.dump(fe_metrics, f, indent=2)

    return fe_df, artifacts_dir


def get_target_column(dataset_name: str, df: pd.DataFrame) -> str:
    if dataset_name == "uci":
        return "g3"
    if dataset_name == "habits":
        return "final_grade"
    if dataset_name == "exam":
        return "exam_score"
    raise ValueError(f"No target detection rule for dataset: {dataset_name}")


def prepare_shap_data(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, str]:
    fe_df, artifacts_dir = load_engineered_features(dataset_name)
    target_column = get_target_column(dataset_name, fe_df)

    from src.model_training.training_utils import NON_FEATURE_COLUMNS

    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in fe_df.columns]
    drop_cols.append(target_column)

    X = fe_df.drop(columns=drop_cols, errors="ignore")
    y = fe_df[target_column]

    X = X.select_dtypes(include=["number"])

    return X, y, artifacts_dir


def sample_for_shap(X: pd.DataFrame, y: pd.Series, max_samples: int = 2000):
    if len(X) <= max_samples:
        return X, y

    idx = np.random.RandomState(42).choice(len(X), size=max_samples, replace=False)
    return X.iloc[idx], y.iloc[idx]


def compute_shap_values(dataset_name: str) -> None:
    print(f"\n[INFO] Starting SHAP computation for dataset: {dataset_name}")

    model = load_model(dataset_name)
    if model is None:
        raise ValueError(f"No trained model found for dataset '{dataset_name}'.")

    X, y, artifacts_dir = prepare_shap_data(dataset_name)
    target_column = get_target_column(dataset_name, pd.concat([X, y], axis=1))

    X_sample, y_sample = sample_for_shap(X, y)

    # ---------------------------------------------------------
    # CASE 1: Regression → TreeExplainer (UCI + Exam)
    # ---------------------------------------------------------
    if is_regression_target(target_column):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        final_values = shap_values

        # Save feature names (required by tests)
        feature_names = list(X_sample.columns)
        with open(os.path.join(artifacts_dir, "feature_names.json"), "w") as f:
            json.dump(feature_names, f, indent=2)

        # Save raw SHAP values
        np.save(os.path.join(artifacts_dir, "shap_values.npy"), final_values)

        # Build Explanation object
        expl = shap.Explanation(
            values=final_values,
            data=X_sample,
            feature_names=X_sample.columns
        )

        # Beeswarm plot
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(expl, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_dir, "shap_beeswarm.png"))
        plt.close()

        # Summary bar plot
        plt.figure(figsize=(10, 6))
        shap.plots.bar(expl, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_dir, "shap_summary.png"))
        plt.close()

        print(f"[INFO] SHAP artifacts saved to: {artifacts_dir}")
        return

    # ---------------------------------------------------------
    # CASE 2: Multi-class HABITS → KernelExplainer + JSON importance
    # ---------------------------------------------------------
    print("[INFO] Using KernelExplainer for multi-class HABITS dataset")

    def predict_fn(data):
        return model.predict_proba(data)

    background = shap.sample(X_sample, 50)
    explainer = shap.KernelExplainer(predict_fn, background)

    # Compute SHAP values for first 200 samples
    shap_values = explainer.shap_values(X_sample[:200])

    # Flatten each class output to 2D
    flattened = []
    for class_vals in shap_values:
        arr = np.array(class_vals)

        if arr.ndim == 2:
            flattened.append(arr)
        elif arr.ndim == 3:
            flattened.append(np.mean(arr, axis=-1))

    if len(flattened) == 0:
        raise ValueError("KernelExplainer produced no valid SHAP outputs.")

    shap_arr = np.stack(flattened, axis=0)
    final_values = np.mean(np.abs(shap_arr), axis=0)

    X_sample = X_sample[:200]

    # ---------------------------------------------------------
    # ALIGN FEATURE NAMES WITH SHAP OUTPUT
    # ---------------------------------------------------------
    num_shap_features = final_values.shape[1]
    feature_names = list(X_sample.columns[:num_shap_features])

    # Save feature names (required by tests)
    with open(os.path.join(artifacts_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    # Save raw SHAP values
    np.save(os.path.join(artifacts_dir, "shap_values.npy"), final_values)

    # ---------------------------------------------------------
    # JSON FEATURE IMPORTANCE (normalized percentages)
    # ---------------------------------------------------------
    mean_importance = np.mean(np.abs(final_values), axis=0)
    total = np.sum(mean_importance)
    percentages = (mean_importance / total) * 100

    importance_list = [
        {"feature": feature_names[i], "importance_pct": float(percentages[i])}
        for i in range(num_shap_features)
    ]

    # Sort descending
    importance_list.sort(key=lambda x: x["importance_pct"], reverse=True)

    # Save JSON
    json_path = os.path.join(artifacts_dir, "shap_importance.json")
    with open(json_path, "w") as f:
        json.dump({"feature_importance": importance_list}, f, indent=2)

    print(f"[INFO] HABITS SHAP importance saved to: {json_path}")
    print(f"[INFO] SHAP artifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    for ds in ["uci", "habits", "exam"]:
        try:
            compute_shap_values(ds)
        except Exception as e:
            print(f"[WARN] SHAP failed for {ds}: {e}")
