import sys
import os

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fairness_audit.fairness_audit import FairnessAudit


def explain_metric(name, value, passed):
    """
    Teacher-friendly explanation for each fairness metric.
    """

    if name == "SPD":
        meaning = (
            "SPD checks whether one group receives more positive predictions than another. "
            "Large differences indicate potential favoritism or disadvantage."
        )
    elif name == "EOD":
        meaning = (
            "EOD checks whether the model correctly identifies positive cases equally across groups. "
            "Large differences indicate unfair treatment in true positives."
        )
    elif name == "DI":
        meaning = (
            "DI checks whether the unprivileged group receives positive outcomes at a similar rate. "
            "Values below 0.75 or above 1.30 indicate systemic bias."
        )

    status = "PASS" if passed else "FAIL"

    return (
        f"{name}: {value:.4f} ({status})\n"
        f"Interpretation: {meaning}\n"
        f"Teacher Meaning: {'No fairness concern detected.' if passed else 'Potential fairness violation detected.'}\n"
    )


def print_teacher_friendly_results(title, results):
    """
    Prints full FR/NFR-compliant fairness results with teacher-friendly explanations.
    """

    print(f"\n=== {title} ===\n")

    print(explain_metric("SPD", results["SPD"], results["SPD_pass"]))
    print(explain_metric("EOD", results["EOD"], results["EOD_pass"]))
    print(explain_metric("DI", results["DI"], results["DI_pass"]))

    overall_status = "PASS" if results["overall_pass"] else "FAIL"
    overall_meaning = (
        "The model meets fairness requirements across all metrics."
        if results["overall_pass"]
        else "The model fails fairness requirements. Teacher should review model behavior and consider corrective action."
    )

    print(f"Overall Fairness Decision: {overall_status}")
    print(f"Teacher Meaning: {overall_meaning}")
    print("\n====================================================\n")


def test_fairness_audit_fail_case():
    """
    FAIL CASE — intentionally biased predictions.
    """

    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 1]  # biased predictions
    gender = ["M", "F", "M", "F"]

    audit = FairnessAudit(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_attr=gender,
        privileged_group="M",
        unprivileged_group="F"
    )

    results = audit.run_audit()

    print_teacher_friendly_results("FAIL CASE RESULTS (FR/NFR + Teacher Interpretation)", results)


def test_fairness_audit_pass_case():
    """
    PASS CASE — fair predictions.
    """

    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 0]  # fair predictions
    gender = ["M", "F", "M", "F"]

    audit = FairnessAudit(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_attr=gender,
        privileged_group="M",
        unprivileged_group="F"
    )

    results = audit.run_audit()

    print_teacher_friendly_results("PASS CASE RESULTS (FR/NFR + Teacher Interpretation)", results)


# ---------------------------------------------------------
# Allow running with python directly
# ---------------------------------------------------------
if __name__ == "__main__":
    test_fairness_audit_fail_case()
    test_fairness_audit_pass_case()
