"""
model_registry.py
-----------------
Handles saving and loading trained models and their metadata.
"""

import os
import json
from typing import Dict, Any
import joblib


def ensure_registry_dir():
    registry_dir = "artifacts/model_registry/"
    os.makedirs(registry_dir, exist_ok=True)
    return registry_dir


def save_model(model, dataset_name: str, metrics: Dict[str, Any]):
    """
    Saves the trained model and its evaluation metrics.
    """

    registry_dir = ensure_registry_dir()

    model_path = os.path.join(registry_dir, f"{dataset_name}_model.joblib")
    meta_path = os.path.join(registry_dir, f"{dataset_name}_metrics.json")

    joblib.dump(model, model_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"[INFO] Model saved to: {model_path}")
    print(f"[INFO] Metrics saved to: {meta_path}")


def load_model(dataset_name: str):
    """
    Loads a trained model for a given dataset.
    """

    registry_dir = ensure_registry_dir()
    model_path = os.path.join(registry_dir, f"{dataset_name}_model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found for dataset: {dataset_name}")

    model = joblib.load(model_path)
    return model
