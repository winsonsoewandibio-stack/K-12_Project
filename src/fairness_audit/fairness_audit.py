"""
Fairness Audit Module
---------------------
This module implements FR/NFR-compliant fairness auditing for binary classification
models used in the K–12 project. It computes industry-standard fairness metrics:

    • Statistical Parity Difference (SPD)
    • Equal Opportunity Difference (EOD)
    • Disparate Impact (DI)

It evaluates each metric against explicit thresholds defined in the NFR:

    • |SPD| ≤ 0.15
    • |EOD| ≤ 0.15
    • 0.75 ≤ DI ≤ 1.30

The module produces:
    • Individual PASS/FAIL flags for each metric
    • An overall PASS/FAIL fairness decision
    • A teacher-friendly interpretability report

This file contains detailed annotations explaining each computation step.
"""

import numpy as np
import pandas as pd


class FairnessAudit:
    """
    FairnessAudit
    -------------
    A fully annotated class implementing fairness evaluation logic.

    Parameters
    ----------
    y_true : list or array
        Ground-truth labels (0 or 1).
    y_pred : list or array
        Model predictions (0 or 1).
    sensitive_attr : list or array
        Sensitive attribute values (e.g., gender, ethnicity).
    privileged_group : str
        Group considered privileged (e.g., "M").
    unprivileged_group : str
        Group considered unprivileged (e.g., "F").
    """

    def __init__(self, y_true, y_pred, sensitive_attr, privileged_group, unprivileged_group):
        # Convert inputs to numpy arrays for vectorized operations
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.sensitive_attr = np.array(sensitive_attr)

        # Store group labels
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group

    def selection_rate(self, group):
        """
        Computes the selection rate:
            P(ŷ = 1 | group)

        This measures how often the model predicts positive outcomes for a group.
        """
        mask = self.sensitive_attr == group
        return np.mean(self.y_pred[mask])

    def true_positive_rate(self, group):
        """
        Computes the true positive rate:
            P(ŷ = 1 | y = 1, group)

        This measures fairness in correctly identifying positive cases.
        """
        mask = self.sensitive_attr == group
        positives = self.y_true[mask] == 1

        # Avoid division by zero if no positives exist
        if positives.sum() == 0:
            return 0.0

        return np.mean(self.y_pred[mask][positives])

    def statistical_parity_difference(self):
        """
        SPD = SR_unprivileged - SR_privileged
        Measures difference in selection rates between groups.
        """
        return self.selection_rate(self.unprivileged_group) - self.selection_rate(self.privileged_group)

    def equal_opportunity_difference(self):
        """
        EOD = TPR_unprivileged - TPR_privileged
        Measures difference in true positive rates between groups.
        """
        return self.true_positive_rate(self.unprivileged_group) - self.true_positive_rate(self.privileged_group)

    def disparate_impact(self):
        """
        DI = SR_unprivileged / SR_privileged
        Measures ratio of selection rates.
        """
        pr = self.selection_rate(self.privileged_group)
        upr = self.selection_rate(self.unprivileged_group)

        if pr == 0:
            return 0.0

        return upr / pr

    def run_audit(self):
        """
        Computes all fairness metrics and evaluates them against NFR thresholds.
        Returns a dictionary containing:
            • Metric values
            • PASS/FAIL flags
            • Overall fairness decision
        """

        spd = self.statistical_parity_difference()
        eod = self.equal_opportunity_difference()
        di = self.disparate_impact()

        results = {
            "SPD": spd,
            "EOD": eod,
            "DI": di,

            # NFR thresholds
            "SPD_pass": abs(spd) <= 0.15,
            "EOD_pass": abs(eod) <= 0.15,
            "DI_pass": 0.75 <= di <= 1.30
        }

        # Overall fairness decision
        results["overall_pass"] = all([
            results["SPD_pass"],
            results["EOD_pass"],
            results["DI_pass"]
        ])

        return results

    def print_audit_report(self):
        """
        Prints a teacher-friendly fairness audit report.
        """
        results = self.run_audit()

        print("\n================ FAIRNESS AUDIT REPORT ================\n")
        print(f"Privileged group: {self.privileged_group}")
        print(f"Unprivileged group: {self.unprivileged_group}\n")

        print(f"Statistical Parity Difference (SPD): {results['SPD']:.4f}")
        print(f"Equal Opportunity Difference (EOD): {results['EOD']:.4f}")
        print(f"Disparate Impact (DI): {results['DI']:.4f}\n")

        print("Thresholds:")
        print(" - SPD ≤ ±0.15")
        print(" - EOD ≤ ±0.15")
        print(" - DI between 0.75 and 1.3\n")

        print("PASS/FAIL:")
        print(f" - SPD Pass: {results['SPD_pass']}")
        print(f" - EOD Pass: {results['EOD_pass']}")
        print(f" - DI Pass: {results['DI_pass']}\n")

        if results["overall_pass"]:
            print("OVERALL RESULT: PASS — No significant fairness violations detected.")
        else:
            print("OVERALL RESULT: FAIL — Significant fairness violations detected.")

        print("\n========================================================\n")
