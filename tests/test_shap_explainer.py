import sys
import os

# Fix Python path so "src" can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.interpretability.shap_explainer import compute_shap_values


def print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70 + "\n")


def assert_file_exists(path: str):
    assert os.path.exists(path), f"Expected file not found: {path}"


def test_shap_uci():
    print_section("SHAP TEST — UCI DATASET")

    compute_shap_values("uci")

    base = "artifacts/interpretability/shap/uci"

    assert_file_exists(f"{base}/shap_beeswarm.png")
    assert_file_exists(f"{base}/shap_summary.png")
    assert_file_exists(f"{base}/fe_metrics.json")
    assert_file_exists(f"{base}/feature_names.json")
    assert_file_exists(f"{base}/shap_values.npy")


def test_shap_habits():
    print_section("SHAP TEST — HABITS DATASET")

    compute_shap_values("habits")

    base = "artifacts/interpretability/shap/habits"

    # HABITS uses JSON importance instead of SHAP plots
    assert_file_exists(f"{base}/shap_importance.json")

    assert_file_exists(f"{base}/fe_metrics.json")
    assert_file_exists(f"{base}/feature_names.json")
    assert_file_exists(f"{base}/shap_values.npy")


def test_shap_exam():
    print_section("SHAP TEST — EXAM DATASET")

    compute_shap_values("exam")

    base = "artifacts/interpretability/shap/exam"

    assert_file_exists(f"{base}/shap_beeswarm.png")
    assert_file_exists(f"{base}/shap_summary.png")
    assert_file_exists(f"{base}/fe_metrics.json")
    assert_file_exists(f"{base}/feature_names.json")
    assert_file_exists(f"{base}/shap_values.npy")


if __name__ == "__main__":
    test_shap_uci()
    test_shap_habits()
    test_shap_exam()
