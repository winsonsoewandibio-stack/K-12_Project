"""
test_model_training.py
----------------------
Tests the model training module for each dataset individually.
Supports both regression and classification.
"""

import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_training.training_pipeline import run_training_pipeline
from src.model_training.model_registry import load_model
from src.model_training.training_rules import TRAINING_NFR_THRESHOLDS


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_training_pipeline_uci():
    print_section("MODEL TRAINING TEST — UCI DATASET")
    run_training_pipeline("uci")
    model = load_model("uci")
    assert model is not None

    metrics_path = "artifacts/model_registry/uci_metrics.json"
    assert os.path.exists(metrics_path)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Regression metrics must exist
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics


def test_training_pipeline_habits():
    print_section("MODEL TRAINING TEST — HABITS DATASET")
    run_training_pipeline("habits")
    model = load_model("habits")
    assert model is not None

    metrics_path = "artifacts/model_registry/habits_metrics.json"
    assert os.path.exists(metrics_path)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Classification metrics must exist
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics


def test_training_pipeline_exam():
    print_section("MODEL TRAINING TEST — EXAM DATASET")
    run_training_pipeline("exam")
    model = load_model("exam")
    assert model is not None

    metrics_path = "artifacts/model_registry/exam_metrics.json"
    assert os.path.exists(metrics_path)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Regression metrics must exist
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics


def test_training_nfr_compliance():
    print_section("NFR COMPLIANCE TEST — MODEL TRAINING")

    for ds in ["uci", "habits", "exam"]:
        metrics_path = f"artifacts/model_registry/{ds}_metrics.json"
        assert os.path.exists(metrics_path)

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        assert metrics["num_features"] <= TRAINING_NFR_THRESHOLDS["max_features"]
        assert metrics["training_time"] <= TRAINING_NFR_THRESHOLDS["training_time_max"]


if __name__ == "__main__":
    test_training_pipeline_uci()
    test_training_pipeline_habits()
    test_training_pipeline_exam()
    test_training_nfr_compliance()
