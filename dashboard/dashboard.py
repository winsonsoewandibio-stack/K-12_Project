import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import shap

# -----------------------------
# Path setup so `src` is importable
# -----------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# -----------------------------
# Safe loaders
# -----------------------------
def safe_load_json(path):
    try:
        return json.load(open(path)) if os.path.exists(path) else None
    except Exception:
        return None

def safe_load_image(path):
    return path if os.path.exists(path) else None

def safe_load_npy(path):
    try:
        return np.load(path) if os.path.exists(path) else None
    except Exception:
        return None

def safe_load_model(path):
    try:
        return joblib.load(path) if os.path.exists(path) else None
    except Exception:
        return None

# -----------------------------
# EXAM-SPECIFIC INTERPRETATION
# -----------------------------
def interpret_exam_model_performance(metrics):
    rmse = metrics.get("rmse")
    mae = metrics.get("mae")
    r2 = metrics.get("r2")
    train_time = metrics.get("training_time")
    num_features = metrics.get("num_features")

    interpretation = []
    actions = []

    if rmse is not None:
        interpretation.append(
            f"RMSE = {rmse:.6f}. The average prediction error is extremely small—"
            f"less than 0.004 points on the exam score scale."
        )
        if rmse < 0.01:
            interpretation.append("This indicates near-perfect precision in predicting exam scores.")

    if mae is not None:
        interpretation.append(
            f"MAE = {mae:.6f}. On average, predictions differ from true scores by only ~0.0005 points."
        )
        interpretation.append("This level of accuracy is effectively indistinguishable from perfect scoring.")

    if r2 is not None:
        interpretation.append(
            f"R² = {r2:.6f}. The model explains **99.998%** of the variance in exam scores."
        )
        interpretation.append("Almost all variation in exam performance is captured by the features.")

    if train_time is not None:
        interpretation.append(
            f"Training time = {train_time:.2f} seconds. This reflects heavy feature engineering."
        )
        if train_time > 300:
            actions.append("Reduce interaction features to shorten training time and improve efficiency.")

    if num_features is not None:
        interpretation.append(
            f"The model uses {num_features} engineered features—a very high number."
        )
        if num_features > 250:
            actions.append("High feature count may reduce explainability. Consider simplifying feature engineering.")

    return interpretation, actions

def interpret_exam_feature_engineering(fe):
    interpretation = []
    actions = []

    fe_time = fe.get("fe_time")
    encoded = fe.get("encoded_columns")
    scaled = fe.get("scaled_columns")
    interactions = fe.get("interaction_features")
    total = fe.get("total_features")

    interpretation.append(
        f"The EXAM model uses **{total} total features**, including {encoded} encoded categorical "
        f"features and {scaled} scaled numeric features."
    )
    interpretation.append(
        f"Interaction features: {interactions}. This indicates heavy feature engineering to capture complex patterns."
    )

    if interactions > 200:
        actions.append("Reduce interaction features to improve explainability for teachers.")

    if fe_time is not None:
        interpretation.append(
            f"Feature engineering took {fe_time:.2f} seconds—acceptable but indicates notable complexity."
        )

    if total > 250:
        actions.append("Consider simplifying the feature set to improve transparency and reduce overfitting risk.")

    return interpretation, actions

def interpret_exam_shap_collapsed(df):
    findings = []
    actions = []

    top10 = df.head(10)

    for _, row in top10.iterrows():
        feature = row["Feature"]
        score = row["Importance"]

        findings.append(f"• **{feature}** strongly influences exam score predictions (importance = {score:.6f}).")

        f = feature.lower()
        if "previous_exam_score" in f:
            actions.append("• Students with lower previous exam scores may need targeted revision plans.")
        elif "previous_gpa" in f:
            actions.append("• GPA trends can guide long-term academic support strategies.")
        elif "attendance_percentage" in f or "attendance" in f:
            actions.append("• Low attendance should trigger early intervention and parent communication.")
        elif "assignment_completion_rate" in f:
            actions.append("• Encourage consistent assignment completion; it strongly predicts exam outcomes.")
        elif "study_hours" in f or "study_hours_per_day" in f:
            actions.append("• Promote structured study routines; study hours directly affect performance.")
        elif "self_study" in f:
            actions.append("• Students with low self-study hours may benefit from guided study plans.")
        elif "practice_tests_completed" in f:
            actions.append("• Increase practice test frequency for students scoring below expectations.")
        elif "stress_level" in f:
            actions.append("• High stress levels require counseling or workload adjustments.")
        elif "exam_anxiety" in f:
            actions.append("• Provide exam-anxiety coping workshops or relaxation techniques.")
        elif "time_management" in f:
            actions.append("• Teach time management strategies; poor time management reduces exam performance.")
        else:
            actions.append(f"• Monitor **{feature}**—it has meaningful impact on exam outcomes.")

    return findings, actions

# -----------------------------
# HABITS-SPECIFIC INTERPRETATION
# -----------------------------
def interpret_habits_model_performance(metrics):
    acc = metrics.get("accuracy")
    prec = metrics.get("precision")
    rec = metrics.get("recall")
    f1 = metrics.get("f1")
    cm = metrics.get("confusion_matrix")
    train_time = metrics.get("training_time")
    num_features = metrics.get("num_features")

    interpretation = []
    actions = []

    if acc is not None:
        interpretation.append(
            f"Accuracy = {acc:.3f}. The HABITS model correctly classifies about **98.5%** of students’ habit profiles."
        )
        if acc > 0.95:
            interpretation.append("This is a very reliable classifier for student habits.")

    if prec is not None and rec is not None and f1 is not None:
        interpretation.append(
            f"Precision = {prec:.3f}, Recall = {rec:.3f}, F1 = {f1:.3f}. "
            "The model balances false alarms and missed cases very well."
        )

    if cm is not None:
        total = sum(sum(row) for row in cm)
        misclassified = total - sum(cm[i][i] for i in range(len(cm)))
        interpretation.append(
            f"Out of {total} students, only {misclassified} are misclassified across all habit categories."
        )
        if misclassified > 0:
            actions.append("Review misclassified students manually, especially those near category boundaries.")

    if train_time is not None:
        interpretation.append(
            f"Training time = {train_time:.3f} seconds—very fast, suitable for frequent retraining."
        )

    if num_features is not None:
        interpretation.append(
            f"The HABITS model uses {num_features} features—a compact and explainable feature set."
        )

    return interpretation, actions

def interpret_habits_feature_engineering(fe):
    interpretation = []
    actions = []

    fe_time = fe.get("fe_time")
    encoded = fe.get("encoded_columns")
    scaled = fe.get("scaled_columns")
    interactions = fe.get("interaction_features")
    total = fe.get("total_features")

    interpretation.append(
        f"The HABITS model uses **{total} total features**, with {encoded} encoded categorical "
        f"and {scaled} scaled numeric features."
    )
    interpretation.append(
        f"Interaction features: {interactions}. This is a moderate level of feature engineering."
    )

    if fe_time is not None:
        interpretation.append(
            f"Feature engineering time = {fe_time:.3f} seconds—very efficient."
        )

    if interactions > 30:
        actions.append("Monitor interaction features to ensure they remain interpretable for teachers.")

    if total <= 50:
        actions.append("The compact feature set supports clear communication of habit drivers to teachers.")

    return interpretation, actions

def interpret_habits_shap_from_importance(shap_importance_list):
    findings = []
    actions = []

    for item in shap_importance_list:
        feature = item["feature"]
        pct = item["importance_pct"]

        findings.append(f"• **{feature}** contributes about {pct:.1f}% of the model’s decision power.")

        f = feature.lower()
        if "study_time" in f:
            actions.append("• Encourage consistent daily study routines; study time is the strongest habit driver.")
        elif "attendance" in f:
            actions.append("• Maintain high attendance; poor attendance strongly harms habit quality.")
        elif "sleep" in f:
            actions.append("• Promote healthy sleep schedules; sleep hours strongly affect learning effectiveness.")
        elif "previous_grade" in f:
            actions.append("• Use previous grades to identify students needing long-term habit support.")
        elif "final_exam_score" in f:
            actions.append("• Connect exam performance back to habits to motivate students to improve routines.")
        else:
            actions.append(f"• Monitor **{feature}** as a key habit-related factor.")

    return findings, actions

# -----------------------------
# UCI-SPECIFIC INTERPRETATION
# -----------------------------
def interpret_uci_model_performance(metrics):
    rmse = metrics.get("rmse")
    mae = metrics.get("mae")
    r2 = metrics.get("r2")
    train_time = metrics.get("training_time")
    num_features = metrics.get("num_features")

    interpretation = []
    actions = []

    if rmse is not None:
        interpretation.append(
            f"RMSE = {rmse:.3f}. On the 0–20 Portuguese grading scale, this means the typical error "
            f"is less than 0.04 grade points."
        )
        interpretation.append(
            "This is far below the boundary between pass (9.5) and fail, so predictions are very stable."
        )

    if mae is not None:
        interpretation.append(
            f"MAE = {mae:.3f}. On average, predicted final grades differ from true grades by about 0.01 points."
        )
        interpretation.append(
            "Such a small error means the model can reliably distinguish between performance bands "
            "like ‘Insuficiente’ and ‘Suficiente’."
        )

    if r2 is not None:
        interpretation.append(
            f"R² = {r2:.6f}. The model explains about **99.84%** of the variance in final grades."
        )
        interpretation.append(
            "Almost all variation in academic performance is captured by the features (study time, absences, "
            "alcohol use, etc.)."
        )

    if train_time is not None:
        interpretation.append(
            f"Training time = {train_time:.3f} seconds—fast enough for regular retraining as new data arrives."
        )

    if num_features is not None:
        interpretation.append(
            f"The UCI model uses {num_features} SHAP-explained features (including interactions)."
        )

    actions.append(
        "Use the model to identify students at risk of falling into ‘Insuficiente’ (3.5–9.4) or ‘Mau’ (0–3.4) "
        "bands and intervene early."
    )
    actions.append(
        "Pay special attention to students near the 9.5 threshold (minimum passing score), where small changes "
        "in habits can shift them between fail and pass."
    )

    return interpretation, actions

def interpret_uci_feature_engineering(fe):
    interpretation = []
    actions = []

    fe_time = fe.get("fe_time")
    encoded = fe.get("encoded_columns")
    scaled = fe.get("scaled_columns")
    interactions = fe.get("interaction_features")
    total = fe.get("total_features")

    interpretation.append(
        f"The UCI model uses **{total} total features**, with {encoded} encoded categorical "
        f"and {scaled} scaled numeric features."
    )
    interpretation.append(
        f"There are {interactions} interaction features (e.g., age × studytime, absences × g1), "
        "capturing combined effects of multiple factors on final grade."
    )

    if fe_time is not None:
        interpretation.append(
            f"Feature engineering time = {fe_time:.3f} seconds—reasonable given the high number of interactions."
        )

    if interactions > 150:
        actions.append(
            "Because interaction features are numerous, rely on collapsed SHAP importance to communicate only "
            "the most meaningful base features to teachers."
        )

    actions.append(
        "Highlight core features like studytime, failures, absences, alcohol use (dalc/walc), and prior grades (g1, g2) "
        "when discussing academic performance with teachers."
    )

    return interpretation, actions

def interpret_uci_shap_collapsed(df):
    findings = []
    actions = []

    top10 = df.head(10)

    for _, row in top10.iterrows():
        feature = row["Feature"]
        score = row["Importance"]

        findings.append(f"• **{feature}** is a major driver of final grade predictions (importance = {score:.6f}).")

        f = feature.lower()
        if "studytime" in f:
            actions.append("• Increase structured study time for students with low studytime; it strongly improves final grades.")
        elif "failures" in f:
            actions.append("• Students with previous failures need targeted remediation and close monitoring to avoid repeating.")
        elif "absences" in f:
            actions.append("• High absences are strongly linked to lower final grades; intervene early on attendance issues.")
        elif "dalc" in f or "walc" in f:
            actions.append(
                "• High daily or weekend alcohol consumption (dalc/walc) is associated with poorer academic performance; "
                "consider counseling and family engagement."
            )
        elif "g1" in f or "g2" in f:
            actions.append(
                "• Use earlier grades (G1, G2) as early warning signals; students with low G1/G2 are at risk of low G3."
            )
        elif "famrel" in f:
            actions.append(
                "• Weak family relationships (low famrel) may correlate with lower grades; consider involving parents "
                "in support plans."
            )
        elif "goout" in f:
            actions.append(
                "• Excessive going out (high goout) can reduce study time and grades; help students balance social life "
                "and school."
            )
        elif "health" in f:
            actions.append(
                "• Poor health scores may hinder learning; coordinate with counselors or health services where needed."
            )
        elif "attendance" in f:
            actions.append("• Maintain strong attendance; it is directly tied to staying in ‘Bom’ or ‘Muito Bom’ bands.")
        elif "exam_score" in f or "grade_level" in f:
            actions.append(
                "• Use exam scores and grade level to tailor expectations and support; students near critical thresholds "
                "need extra guidance."
            )
        else:
            actions.append(f"• Monitor **{feature}** as a meaningful contributor to final grade outcomes.")

    return findings, actions

# -----------------------------
# Feature lists (full lists from your original file)
# -----------------------------
EXAM_FEATURES = [
    "age",
    "previous_exam_score",
    "previous_gpa",
    "attendance_percentage",
    "assignment_completion_rate",
    "study_hours_per_day",
    "self_study_hours",
    "private_tuition",
    "online_learning_hours",
    "practice_tests_completed",
    "sleep_hours",
    "daily_screen_time",
    "physical_activity_hours",
    "stress_level",
    "internet_access",
    "online_course_hours",
    "exam_preparation_days",
    "questions_attempted",
    "questions_correct",
    "time_management_score",
    "exam_anxiety_level",
    "attendance",
    "age_x_previous_exam_score",
    "age_x_previous_gpa",
    "age_x_attendance_percentage",
    "age_x_assignment_completion_rate",
    "age_x_study_hours_per_day",
    "age_x_self_study_hours",
    "age_x_private_tuition",
    "age_x_online_learning_hours",
    "age_x_practice_tests_completed",
    "age_x_sleep_hours",
    "age_x_daily_screen_time",
    "age_x_physical_activity_hours",
    "age_x_stress_level",
    "age_x_internet_access",
    "age_x_online_course_hours",
    "age_x_exam_preparation_days",
    "age_x_questions_attempted",
    "age_x_questions_correct",
    "age_x_time_management_score",
    "age_x_exam_anxiety_level",
    "age_x_exam_score",
    "age_x_attendance",
    "previous_exam_score_x_previous_gpa",
    "previous_exam_score_x_attendance_percentage",
    "previous_exam_score_x_assignment_completion_rate",
    "previous_exam_score_x_study_hours_per_day",
    "previous_exam_score_x_self_study_hours",
    "previous_exam_score_x_private_tuition",
    "previous_exam_score_x_online_learning_hours",
    "previous_exam_score_x_practice_tests_completed",
    "previous_exam_score_x_sleep_hours",
    "previous_exam_score_x_daily_screen_time",
    "previous_exam_score_x_physical_activity_hours",
    "previous_exam_score_x_stress_level",
    "previous_exam_score_x_internet_access",
    "previous_exam_score_x_online_course_hours",
    "previous_exam_score_x_exam_preparation_days",
    "previous_exam_score_x_questions_attempted",
    "previous_exam_score_x_questions_correct",
    "previous_exam_score_x_time_management_score",
    "previous_exam_score_x_exam_anxiety_level",
    "previous_exam_score_x_exam_score",
    "previous_exam_score_x_attendance",
    "previous_gpa_x_attendance_percentage",
    "previous_gpa_x_assignment_completion_rate",
    "previous_gpa_x_study_hours_per_day",
    "previous_gpa_x_self_study_hours",
    "previous_gpa_x_private_tuition",
    "previous_gpa_x_online_learning_hours",
    "previous_gpa_x_practice_tests_completed",
    "previous_gpa_x_sleep_hours",
    "previous_gpa_x_daily_screen_time",
    "previous_gpa_x_physical_activity_hours",
    "previous_gpa_x_stress_level",
    "previous_gpa_x_internet_access",
    "previous_gpa_x_online_course_hours",
    "previous_gpa_x_exam_preparation_days",
    "previous_gpa_x_questions_attempted",
    "previous_gpa_x_questions_correct",
    "previous_gpa_x_time_management_score",
    "previous_gpa_x_exam_anxiety_level",
    "previous_gpa_x_exam_score",
    "previous_gpa_x_attendance",
    "attendance_percentage_x_assignment_completion_rate",
    "attendance_percentage_x_study_hours_per_day",
    "attendance_percentage_x_self_study_hours",
    "attendance_percentage_x_private_tuition",
    "attendance_percentage_x_online_learning_hours",
    "attendance_percentage_x_practice_tests_completed",
    "attendance_percentage_x_sleep_hours",
    "attendance_percentage_x_daily_screen_time",
    "attendance_percentage_x_physical_activity_hours",
    "attendance_percentage_x_stress_level",
    "attendance_percentage_x_internet_access",
    "attendance_percentage_x_online_course_hours",
    "attendance_percentage_x_exam_preparation_days",
    "attendance_percentage_x_questions_attempted",
    "attendance_percentage_x_questions_correct",
    "attendance_percentage_x_time_management_score",
    "attendance_percentage_x_exam_anxiety_level",
    "attendance_percentage_x_exam_score",
    "attendance_percentage_x_attendance",
    "assignment_completion_rate_x_study_hours_per_day",
    "assignment_completion_rate_x_self_study_hours",
    "assignment_completion_rate_x_private_tuition",
    "assignment_completion_rate_x_online_learning_hours",
    "assignment_completion_rate_x_practice_tests_completed",
    "assignment_completion_rate_x_sleep_hours",
    "assignment_completion_rate_x_daily_screen_time",
    "assignment_completion_rate_x_physical_activity_hours",
    "assignment_completion_rate_x_stress_level",
    "assignment_completion_rate_x_internet_access",
    "assignment_completion_rate_x_online_course_hours",
    "assignment_completion_rate_x_exam_preparation_days",
    "assignment_completion_rate_x_questions_attempted",
    "assignment_completion_rate_x_questions_correct",
    "assignment_completion_rate_x_time_management_score",
    "assignment_completion_rate_x_exam_anxiety_level",
    "assignment_completion_rate_x_exam_score",
    "assignment_completion_rate_x_attendance",
    "study_hours_per_day_x_self_study_hours",
    "study_hours_per_day_x_private_tuition",
    "study_hours_per_day_x_online_learning_hours",
    "study_hours_per_day_x_practice_tests_completed",
    "study_hours_per_day_x_sleep_hours",
    "study_hours_per_day_x_daily_screen_time",
    "study_hours_per_day_x_physical_activity_hours",
    "study_hours_per_day_x_stress_level",
    "study_hours_per_day_x_internet_access",
    "study_hours_per_day_x_online_course_hours",
    "study_hours_per_day_x_exam_preparation_days",
    "study_hours_per_day_x_questions_attempted",
    "study_hours_per_day_x_questions_correct",
    "study_hours_per_day_x_time_management_score",
    "study_hours_per_day_x_exam_anxiety_level",
    "study_hours_per_day_x_exam_score",
    "study_hours_per_day_x_attendance",
    "self_study_hours_x_private_tuition",
    "self_study_hours_x_online_learning_hours",
    "self_study_hours_x_practice_tests_completed",
    "self_study_hours_x_sleep_hours",
    "self_study_hours_x_daily_screen_time",
    "self_study_hours_x_physical_activity_hours",
    "self_study_hours_x_stress_level",
    "self_study_hours_x_internet_access",
    "self_study_hours_x_online_course_hours",
    "self_study_hours_x_exam_preparation_days",
    "self_study_hours_x_questions_attempted",
    "self_study_hours_x_questions_correct",
    "self_study_hours_x_time_management_score",
    "self_study_hours_x_exam_anxiety_level",
    "self_study_hours_x_exam_score",
    "self_study_hours_x_attendance",
    "private_tuition_x_online_learning_hours",
    "private_tuition_x_practice_tests_completed",
    "private_tuition_x_sleep_hours",
    "private_tuition_x_daily_screen_time",
    "private_tuition_x_physical_activity_hours",
    "private_tuition_x_stress_level",
    "private_tuition_x_internet_access",
    "private_tuition_x_online_course_hours",
    "private_tuition_x_exam_preparation_days",
    "private_tuition_x_questions_attempted",
    "private_tuition_x_questions_correct",
    "private_tuition_x_time_management_score",
    "private_tuition_x_exam_anxiety_level",
    "private_tuition_x_exam_score",
    "private_tuition_x_attendance",
    "online_learning_hours_x_practice_tests_completed",
    "online_learning_hours_x_sleep_hours",
    "online_learning_hours_x_daily_screen_time",
    "online_learning_hours_x_physical_activity_hours",
    "online_learning_hours_x_stress_level",
    "online_learning_hours_x_internet_access",
    "online_learning_hours_x_online_course_hours",
    "online_learning_hours_x_exam_preparation_days",
    "online_learning_hours_x_questions_attempted",
    "online_learning_hours_x_questions_correct",
    "online_learning_hours_x_time_management_score",
    "online_learning_hours_x_exam_anxiety_level",
    "online_learning_hours_x_exam_score",
    "online_learning_hours_x_attendance",
    "practice_tests_completed_x_sleep_hours",
    "practice_tests_completed_x_daily_screen_time",
    "practice_tests_completed_x_physical_activity_hours",
    "practice_tests_completed_x_stress_level",
    "practice_tests_completed_x_internet_access",
    "practice_tests_completed_x_online_course_hours",
    "practice_tests_completed_x_exam_preparation_days",
    "practice_tests_completed_x_questions_attempted",
    "practice_tests_completed_x_questions_correct",
    "practice_tests_completed_x_time_management_score",
    "practice_tests_completed_x_exam_anxiety_level",
    "practice_tests_completed_x_exam_score",
    "practice_tests_completed_x_attendance",
    "sleep_hours_x_daily_screen_time",
    "sleep_hours_x_physical_activity_hours",
    "sleep_hours_x_stress_level",
    "sleep_hours_x_internet_access",
    "sleep_hours_x_online_course_hours",
    "sleep_hours_x_exam_preparation_days",
    "sleep_hours_x_questions_attempted",
    "sleep_hours_x_questions_correct",
    "sleep_hours_x_time_management_score",
    "sleep_hours_x_exam_anxiety_level",
    "sleep_hours_x_exam_score",
    "sleep_hours_x_attendance",
    "daily_screen_time_x_physical_activity_hours",
    "daily_screen_time_x_stress_level",
    "daily_screen_time_x_internet_access",
    "daily_screen_time_x_online_course_hours",
    "daily_screen_time_x_exam_preparation_days",
    "daily_screen_time_x_questions_attempted",
    "daily_screen_time_x_questions_correct",
    "daily_screen_time_x_time_management_score",
    "daily_screen_time_x_exam_anxiety_level",
    "daily_screen_time_x_exam_score",
    "daily_screen_time_x_attendance",
    "physical_activity_hours_x_stress_level",
    "physical_activity_hours_x_internet_access",
    "physical_activity_hours_x_online_course_hours",
    "physical_activity_hours_x_exam_preparation_days",
    "physical_activity_hours_x_questions_attempted",
    "physical_activity_hours_x_questions_correct",
    "physical_activity_hours_x_time_management_score",
    "physical_activity_hours_x_exam_anxiety_level",
    "physical_activity_hours_x_exam_score",
    "physical_activity_hours_x_attendance",
    "stress_level_x_internet_access",
    "stress_level_x_online_course_hours",
    "stress_level_x_exam_preparation_days",
    "stress_level_x_questions_attempted",
    "stress_level_x_questions_correct",
    "stress_level_x_time_management_score",
    "stress_level_x_exam_anxiety_level",
    "stress_level_x_exam_score",
    "stress_level_x_attendance",
    "internet_access_x_online_course_hours",
    "internet_access_x_exam_preparation_days",
    "internet_access_x_questions_attempted",
    "internet_access_x_questions_correct",
    "internet_access_x_time_management_score",
    "internet_access_x_exam_anxiety_level",
    "internet_access_x_exam_score",
    "internet_access_x_attendance",
    "online_course_hours_x_exam_preparation_days",
    "online_course_hours_x_questions_attempted",
    "online_course_hours_x_questions_correct",
    "online_course_hours_x_time_management_score",
    "online_course_hours_x_exam_anxiety_level",
    "online_course_hours_x_exam_score",
    "online_course_hours_x_attendance",
    "exam_preparation_days_x_questions_attempted",
    "exam_preparation_days_x_questions_correct",
    "exam_preparation_days_x_time_management_score",
    "exam_preparation_days_x_exam_anxiety_level",
    "exam_preparation_days_x_exam_score",
    "exam_preparation_days_x_attendance",
    "questions_attempted_x_questions_correct",
    "questions_attempted_x_time_management_score",
    "questions_attempted_x_exam_anxiety_level",
    "questions_attempted_x_exam_score",
    "questions_attempted_x_attendance",
    "questions_correct_x_time_management_score",
    "questions_correct_x_exam_anxiety_level",
    "questions_correct_x_exam_score",
    "questions_correct_x_attendance",
    "time_management_score_x_exam_anxiety_level",
    "time_management_score_x_exam_score",
    "time_management_score_x_attendance",
    "exam_anxiety_level_x_exam_score",
    "exam_anxiety_level_x_attendance",
    "exam_score_x_attendance"
]

HABITS_FEATURES = [
    "study_time_hours",
    "attendance_percent",
    "sleep_hours",
    "previous_grade",
    "final_exam_score"
]

UCI_FEATURES = [
    "age","medu","fedu","traveltime","studytime","failures","famrel","freetime","goout",
    "dalc","walc","health","absences","g1","g2","grade_level","attendance","exam_score",
    "age_x_medu","age_x_fedu","age_x_traveltime","age_x_studytime","age_x_failures",
    "age_x_famrel","age_x_freetime","age_x_goout","age_x_dalc","age_x_walc","age_x_health",
    "age_x_absences","age_x_g1","age_x_g2","age_x_g3","age_x_grade_level","age_x_attendance",
    "age_x_exam_score","medu_x_fedu","medu_x_traveltime","medu_x_studytime","medu_x_failures",
    "medu_x_famrel","medu_x_freetime","medu_x_goout","medu_x_dalc","medu_x_walc","medu_x_health",
    "medu_x_absences","medu_x_g1","medu_x_g2","medu_x_g3","medu_x_grade_level","medu_x_attendance",
    "medu_x_exam_score","fedu_x_traveltime","fedu_x_studytime","fedu_x_failures","fedu_x_famrel",
    "fedu_x_freetime","fedu_x_goout","fedu_x_dalc","fedu_x_walc","fedu_x_health","fedu_x_absences",
    "fedu_x_g1","fedu_x_g2","fedu_x_g3","fedu_x_grade_level","fedu_x_attendance","fedu_x_exam_score",
    "traveltime_x_studytime","traveltime_x_failures","traveltime_x_famrel","traveltime_x_freetime",
    "traveltime_x_goout","traveltime_x_dalc","traveltime_x_walc","traveltime_x_health",
    "traveltime_x_absences","traveltime_x_g1","traveltime_x_g2","traveltime_x_g3",
    "traveltime_x_grade_level","traveltime_x_attendance","traveltime_x_exam_score",
    "studytime_x_failures","studytime_x_famrel","studytime_x_freetime","studytime_x_goout",
    "studytime_x_dalc","studytime_x_walc","studytime_x_health","studytime_x_absences",
    "studytime_x_g1","studytime_x_g2","studytime_x_g3","studytime_x_grade_level",
    "studytime_x_attendance","studytime_x_exam_score","failures_x_famrel","failures_x_freetime",
    "failures_x_goout","failures_x_dalc","failures_x_walc","failures_x_health","failures_x_absences",
    "failures_x_g1","failures_x_g2","failures_x_g3","failures_x_grade_level","failures_x_attendance",
    "failures_x_exam_score","famrel_x_freetime","famrel_x_goout","famrel_x_dalc","famrel_x_walc",
    "famrel_x_health","famrel_x_absences","famrel_x_g1","famrel_x_g2","famrel_x_g3",
    "famrel_x_grade_level","famrel_x_attendance","famrel_x_exam_score","freetime_x_goout",
    "freetime_x_dalc","freetime_x_walc","freetime_x_health","freetime_x_absences","freetime_x_g1",
    "freetime_x_g2","freetime_x_g3","freetime_x_grade_level","freetime_x_attendance",
    "freetime_x_exam_score","goout_x_dalc","goout_x_walc","goout_x_health","goout_x_absences",
    "goout_x_g1","goout_x_g2","goout_x_g3","goout_x_grade_level","goout_x_attendance",
    "goout_x_exam_score","dalc_x_walc","dalc_x_health","dalc_x_absences","dalc_x_g1","dalc_x_g2",
    "dalc_x_g3","dalc_x_grade_level","dalc_x_attendance","dalc_x_exam_score","walc_x_health",
    "walc_x_absences","walc_x_g1","walc_x_g2","walc_x_g3","walc_x_grade_level","walc_x_attendance",
    "walc_x_exam_score","health_x_absences","health_x_g1","health_x_g2","health_x_g3",
    "health_x_grade_level","health_x_attendance","health_x_exam_score","absences_x_g1",
    "absences_x_g2","absences_x_g3","absences_x_grade_level","absences_x_attendance",
    "absences_x_exam_score","g1_x_g2","g1_x_g3","g1_x_grade_level","g1_x_attendance",
    "g1_x_exam_score","g2_x_g3","g2_x_grade_level","g2_x_attendance","g2_x_exam_score",
    "g3_x_grade_level","g3_x_attendance","g3_x_exam_score","grade_level_x_attendance",
    "grade_level_x_exam_score","attendance_x_exam_score"
]

# -----------------------------
# Single-row FE functions
# -----------------------------
def fe_single_exam_student(raw):
    base = raw.copy()
    features = {}

    for k in base:
        if k in EXAM_FEATURES:
            features[k] = base[k]

    for name in EXAM_FEATURES:
        if "_x_" in name:
            a, b = name.split("_x_")
            val_a = base.get(a, 0.0)
            val_b = base.get(b, 0.0)
            features[name] = val_a * val_b

    for name in EXAM_FEATURES:
        if name not in features:
            features[name] = 0.0

    return pd.DataFrame([features])

def fe_single_habits_student(raw):
    return pd.DataFrame([[
        raw["study_time_hours"],
        raw["attendance_percent"],
        raw["sleep_hours"],
        raw["previous_grade"],
        raw["final_exam_score"],
    ]], columns=HABITS_FEATURES)

def fe_single_uci_student(raw):
    base = raw.copy()
    base["g3"] = 0.0

    features = {}
    for k in base:
        if k in UCI_FEATURES:
            features[k] = base[k]

    for name in UCI_FEATURES:
        if "_x_" in name:
            a, b = name.split("_x_")
            val_a = base.get(a, 0.0)
            val_b = base.get(b, 0.0)
            features[name] = val_a * val_b

    for name in UCI_FEATURES:
        if name not in features:
            features[name] = 0.0

    return pd.DataFrame([features])

# -----------------------------
# Load artifacts
# -----------------------------
def load_artifacts(dataset_name):
    base_path = "artifacts"

    model_path = os.path.join(base_path, "model_registry", f"{dataset_name}_model.joblib")
    metrics_path = os.path.join(base_path, "model_registry", f"{dataset_name}_metrics.json")

    shap_base = os.path.join(base_path, "interpretability", "shap", dataset_name)
    shap_values_path = os.path.join(shap_base, "shap_values.npy")
    shap_summary_img = os.path.join(shap_base, "shap_summary.png")
    shap_beeswarm_img = os.path.join(shap_base, "shap_beeswarm.png")
    feature_names_path = os.path.join(shap_base, "feature_names.json")
    fe_metrics_path = os.path.join(shap_base, "fe_metrics.json")
    shap_importance_path = os.path.join(shap_base, "shap_importance.json")

    return {
        "model": safe_load_model(model_path),
        "metrics": safe_load_json(metrics_path),
        "shap_values": safe_load_npy(shap_values_path),
        "feature_names": safe_load_json(feature_names_path),
        "shap_summary_img": safe_load_image(shap_summary_img),
        "shap_beeswarm_img": safe_load_image(shap_beeswarm_img),
        "fe_metrics": safe_load_json(fe_metrics_path),
        "shap_importance": safe_load_json(shap_importance_path),
    }

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("K–12 ML Dashboard")

dataset_choice = st.sidebar.selectbox(
    "Choose dataset:",
    ["exam", "habits", "uci"]
)

artifacts = load_artifacts(dataset_choice)

page = st.sidebar.radio(
    "Select Dashboard Page",
    [
        "Model Performance",
        "SHAP Interpretability",
        "Feature Engineering Summary",
        "Feature Importance",
        "Student Intervention Checker",
    ],
)

# -----------------------------
# Page 1 — Model Performance
# -----------------------------
if page == "Model Performance":
    st.title(f"Model Performance — {dataset_choice.upper()}")

    metrics = artifacts["metrics"]

    if metrics is None:
        st.error("No metrics.json found for this dataset.")
    else:
        st.subheader("Raw Metrics")
        st.json(metrics)

        if dataset_choice == "exam":
            interpretation, actions = interpret_exam_model_performance(metrics)
            st.subheader("Teacher Interpretation (EXAM Dataset)")
        elif dataset_choice == "habits":
            interpretation, actions = interpret_habits_model_performance(metrics)
            st.subheader("Teacher Interpretation (HABITS Dataset)")
        elif dataset_choice == "uci":
            interpretation, actions = interpret_uci_model_performance(metrics)
            st.subheader("Teacher Interpretation (UCI Final Grade Dataset)")
        else:
            interpretation, actions = [], []
            st.subheader("Teacher Interpretation")
            st.write("Generic interpretation not yet configured for this dataset.")

        for line in interpretation:
            st.write(line)

        st.subheader("Recommended Actions for Teachers")
        for line in actions:
            st.write(line)

# -----------------------------
# Page 2 — SHAP Interpretability
# -----------------------------
elif page == "SHAP Interpretability":
    st.title(f"SHAP Interpretability — {dataset_choice.upper()}")

    shap_summary = artifacts["shap_summary_img"]
    shap_beeswarm = artifacts["shap_beeswarm_img"]

    st.subheader("SHAP Summary Plot")
    if shap_summary:
        st.image(shap_summary)
    else:
        st.info("No SHAP summary plot available for this dataset.")
        st.write(
            "• The model still provides SHAP values even without a summary PNG.\n"
            "• You can rely on the Feature Importance page for detailed insights."
        )

    st.subheader("SHAP Beeswarm Plot")
    if shap_beeswarm:
        st.image(shap_beeswarm)
    else:
        st.info("No SHAP beeswarm plot available for this dataset.")
        st.write(
            "• Beeswarm plots visualize how each feature pushes predictions up or down.\n"
            "• Since this PNG is missing, refer to the Feature Importance page for ranked SHAP contributions."
        )

    if dataset_choice == "exam":
        st.subheader("Teacher Interpretation (EXAM Dataset)")
        st.write(
            "• SHAP plots show how each feature pushes exam score predictions up or down.\n"
            "• Red = higher feature values; Blue = lower feature values.\n"
            "• The most influential features (e.g., previous exam score, GPA, attendance, study habits) appear at the top."
        )
        st.subheader("Recommended Actions for Teachers")
        st.write(
            "• Focus on the top SHAP features such as previous exam score, GPA, attendance, and study habits.\n"
            "• Use SHAP insights to personalize academic support and intervention plans."
        )
    elif dataset_choice == "habits":
        st.subheader("Teacher Interpretation (HABITS Dataset)")
        st.write(
            "• SHAP values show how each habit (study time, attendance, sleep, previous grade, exam score)\n"
            "  influences the predicted habit profile.\n"
            "• Study time, attendance, and sleep hours are typically the strongest drivers."
        )
        st.subheader("Recommended Actions for Teachers")
        st.write(
            "• Encourage consistent daily study routines.\n"
            "• Maintain high attendance.\n"
            "• Promote healthy sleep schedules.\n"
            "• Use previous grades to identify students needing long-term habit support.\n"
            "• Connect exam performance back to habits to motivate students."
        )
    elif dataset_choice == "uci":
        st.subheader("Teacher Interpretation (UCI Final Grade Dataset)")
        st.write(
            "• SHAP values show how each factor (studytime, failures, absences, alcohol use, family relations, etc.)\n"
            "  influences the predicted final grade (G3) on the 0–20 Portuguese scale.\n"
            "• Positive SHAP values push students toward higher bands (e.g., ‘Bom’, ‘Muito Bom’, ‘Excelente’).\n"
            "• Negative SHAP values push students toward lower bands (e.g., ‘Insuficiente’, ‘Mau’)."
        )
        st.subheader("Recommended Actions for Teachers")
        st.write(
            "• Use SHAP insights to identify which factors most strongly push a student toward failing bands.\n"
            "• Prioritize interventions on studytime, absences, failures, and alcohol use where SHAP shows strong influence.\n"
            "• Pay special attention to students near the 9.5 passing threshold."
        )
    else:
        st.subheader("Teacher Interpretation")
        st.write("SHAP interpretation not yet customized for this dataset.")

# -----------------------------
# Page 3 — Feature Engineering Summary
# -----------------------------
elif page == "Feature Engineering Summary":
    st.title(f"Feature Engineering Summary — {dataset_choice.upper()}")

    fe = artifacts["fe_metrics"]

    if fe is None:
        st.error("fe_metrics.json not found for this dataset.")
    else:
        st.subheader("Raw Feature Engineering Metrics")
        st.json(fe)

        if dataset_choice == "exam":
            interpretation, actions = interpret_exam_feature_engineering(fe)
            st.subheader("Teacher Interpretation (EXAM Dataset)")
        elif dataset_choice == "habits":
            interpretation, actions = interpret_habits_feature_engineering(fe)
            st.subheader("Teacher Interpretation (HABITS Dataset)")
        elif dataset_choice == "uci":
            interpretation, actions = interpret_uci_feature_engineering(fe)
            st.subheader("Teacher Interpretation (UCI Final Grade Dataset)")
        else:
            interpretation, actions = [], []
            st.subheader("Teacher Interpretation")
            st.write("Feature engineering interpretation not yet configured for this dataset.")

        for line in interpretation:
            st.write(line)

        st.subheader("Recommended Actions for Teachers")
        for line in actions:
            st.write(line)

# -----------------------------
# Page 4 — Feature Importance
# -----------------------------
elif page == "Feature Importance":
    st.title(f"Feature Importance — {dataset_choice.upper()}")

    shap_values = artifacts["shap_values"]
    names = artifacts["feature_names"]
    shap_importance = artifacts["shap_importance"]

    if dataset_choice == "exam":
        if shap_values is None or names is None:
            st.error("shap_values.npy or feature_names.json not found for EXAM dataset.")
        else:
            st.subheader("Raw Feature Names (EXAM)")
            st.json(names)

            raw_importance = np.mean(np.abs(shap_values), axis=0)
            collapsed = {}

            for feature, imp in zip(names, raw_importance):
                if "_x_" in feature:
                    base = feature.split("_x_")[0]
                else:
                    base = feature
                collapsed[base] = collapsed.get(base, 0) + imp

            df = pd.DataFrame({
                "Feature": list(collapsed.keys()),
                "Importance": list(collapsed.values()),
            }).sort_values("Importance", ascending=False)

            top10 = df.head(10)
            st.subheader("Top 10 Meaningful Features (EXAM, Collapsed)")
            st.dataframe(top10)

            findings, actions = interpret_exam_shap_collapsed(df)

            st.subheader("Top 10 Findings")
            for f in findings:
                st.write(f)

            st.subheader("Recommended Actions for Teachers")
            for a in actions:
                st.write(a)

    elif dataset_choice == "habits":
        if shap_importance is None or "feature_importance" not in shap_importance:
            st.error("shap_importance.json with 'feature_importance' not found for HABITS dataset.")
        else:
            fi_list = shap_importance["feature_importance"]

            st.subheader("Raw SHAP Importance (HABITS)")
            st.json(fi_list)

            df = pd.DataFrame(fi_list).rename(columns={"importance_pct": "ImportancePct"})
            st.subheader("Feature Importance (HABITS, Percentage Contribution)")
            st.dataframe(df)

            findings, actions = interpret_habits_shap_from_importance(fi_list)

            st.subheader("Findings")
            for f in findings:
                st.write(f)

            st.subheader("Recommended Actions for Teachers")
            for a in actions:
                st.write(a)

    elif dataset_choice == "uci":
        if shap_values is None or names is None:
            st.error("shap_values.npy or feature_names.json not found for UCI dataset.")
        else:
            st.subheader("Raw Feature Names (UCI)")
            st.json(names)

            raw_importance = np.mean(np.abs(shap_values), axis=0)
            collapsed = {}

            for feature, imp in zip(names, raw_importance):
                if "_x_" in feature:
                    base = feature.split("_x_")[0]
                else:
                    base = feature
                collapsed[base] = collapsed.get(base, 0) + imp

            df = pd.DataFrame({
                "Feature": list(collapsed.keys()),
                "Importance": list(collapsed.values()),
            }).sort_values("Importance", ascending=False)

            top10 = df.head(10)
            st.subheader("Top 10 Meaningful Features (UCI Final Grade, Collapsed)")
            st.dataframe(top10)

            findings, actions = interpret_uci_shap_collapsed(df)

            st.subheader("Top 10 Findings")
            for f in findings:
                st.write(f)

            st.subheader("Recommended Actions for Teachers")
            for a in actions:
                st.write(a)
    else:
        st.write("Feature importance not yet configured for this dataset.")

# -----------------------------
# Page 5 — Student Intervention Checker
# -----------------------------
elif page == "Student Intervention Checker":
    st.title("Student Intervention Checker")
    st.write("This tool uses the ML model internally to generate teacher recommendations.")

    model = artifacts["model"]

    # ============================================================
    # EXAM — ML-powered recommendations (no prediction shown)
    # ============================================================
    if dataset_choice == "exam":
        st.header("EXAM Student Input")

        age = st.number_input("Age", 5, 25, 16)
        previous_exam_score = st.number_input("Previous Exam Score (0–100)", 0.0, 100.0, 70.0)
        previous_gpa = st.number_input("Previous GPA (0.0–4.0)", 0.0, 4.0, 3.0)
        attendance_percentage = st.number_input("Attendance Percentage (0–100)", 0.0, 100.0, 85.0)
        assignment_completion_rate = st.number_input("Assignment Completion Rate (0–100)", 0.0, 100.0, 80.0)
        study_hours_per_day = st.number_input("Study Hours per Day", 0.0, 24.0, 2.0)
        self_study_hours = st.number_input("Self Study Hours", 0.0, 24.0, 1.0)
        private_tuition = st.number_input("Private Tuition Hours", 0.0, 40.0, 0.0)
        online_learning_hours = st.number_input("Online Learning Hours", 0.0, 40.0, 2.0)
        practice_tests_completed = st.number_input("Practice Tests Completed", 0, 50, 3)
        sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
        daily_screen_time = st.number_input("Daily Screen Time", 0.0, 24.0, 4.0)
        physical_activity_hours = st.number_input("Physical Activity Hours", 0.0, 40.0, 2.0)
        stress_level = st.number_input("Stress Level (1–10)", 1, 10, 5)
        internet_access = st.selectbox("Internet Access", ["No", "Yes"], index=1)
        online_course_hours = st.number_input("Online Course Hours", 0.0, 40.0, 1.0)
        exam_preparation_days = st.number_input("Exam Preparation Days", 0, 60, 10)
        questions_attempted = st.number_input("Questions Attempted", 0, 500, 100)
        questions_correct = st.number_input("Questions Correct", 0, 500, 80)
        time_management_score = st.number_input("Time Management Score (1–10)", 1, 10, 6)
        exam_anxiety_level = st.number_input("Exam Anxiety Level (1–10)", 1, 10, 4)
        attendance = st.number_input("Attendance", 0.0, 100.0, 90.0)

        raw_exam = {
            "age": age,
            "previous_exam_score": previous_exam_score,
            "previous_gpa": previous_gpa,
            "attendance_percentage": attendance_percentage,
            "assignment_completion_rate": assignment_completion_rate,
            "study_hours_per_day": study_hours_per_day,
            "self_study_hours": self_study_hours,
            "private_tuition": private_tuition,
            "online_learning_hours": online_learning_hours,
            "practice_tests_completed": practice_tests_completed,
            "sleep_hours": sleep_hours,
            "daily_screen_time": daily_screen_time,
            "physical_activity_hours": physical_activity_hours,
            "stress_level": stress_level,
            "internet_access": 1.0 if internet_access == "Yes" else 0.0,
            "online_course_hours": online_course_hours,
            "exam_preparation_days": exam_preparation_days,
            "questions_attempted": questions_attempted,
            "questions_correct": questions_correct,
            "time_management_score": time_management_score,
            "exam_anxiety_level": exam_anxiety_level,
            "attendance": attendance
        }

        if st.button("Generate Teacher Recommendations (EXAM)"):

            df_engineered = fe_single_exam_student(raw_exam)

            # ML prediction (not shown)
            _ = model.predict(df_engineered)

            # SHAP explanation
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(df_engineered)
            shap_df = pd.DataFrame({
                "Feature": df_engineered.columns,
                "SHAP": shap_vals.flatten()
            }).sort_values("SHAP")

            st.subheader("Teacher Recommendations")

            recommendations = set()  # <-- remove duplicates automatically

            negative_features = shap_df.head(10)

            for _, row in negative_features.iterrows():
                feature = row["Feature"]
                shap_value = row["SHAP"]

                if shap_value < 0:
                    if "attendance" in feature:
                        recommendations.add("Improve attendance through parent communication.")
                    if "study_hours" in feature:
                        recommendations.add("Increase daily study time with structured routines.")
                    if "assignment_completion" in feature:
                        recommendations.add("Support assignment completion habits.")
                    if "stress" in feature:
                        recommendations.add("Provide stress‑management support.")
                    if "anxiety" in feature:
                        recommendations.add("Offer exam‑anxiety coping strategies.")
                    if "previous_exam_score" in feature:
                        recommendations.add("Provide targeted revision and remediation.")

            if len(recommendations) == 0:
                recommendations.add("Student appears on track. Maintain current support strategies.")

            for r in sorted(recommendations):
                st.write("• " + r)

    # ============================================================
    # HABITS — no prediction, SHAP-driven recommendations
    # ============================================================
    elif dataset_choice == "habits":
        st.header("HABITS Student Input")

        study_time_hours = st.number_input("Study Time (hours per day)", 0.0, 24.0, 2.0)
        attendance_percent = st.number_input("Attendance (%)", 0.0, 100.0, 90.0)
        sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
        previous_grade = st.number_input("Previous Grade (%)", 0.0, 100.0, 75.0)

        if st.button("Generate Teacher Recommendations (HABITS)"):

            st.warning("The HABITS model does not support prediction. Only habit-based recommendations are available.")

            st.subheader("Teacher Recommendations")

            recommendations = set()

            shap_importance = artifacts["shap_importance"]["feature_importance"]

            for item in shap_importance:
                feature = item["feature"]

                if feature == "study_time_hours" and study_time_hours < 2:
                    recommendations.add("Encourage more consistent daily study routines.")
                if feature == "attendance_percent" and attendance_percent < 80:
                    recommendations.add("Improve attendance consistency.")
                if feature == "sleep_hours" and sleep_hours < 6:
                    recommendations.add("Promote healthier sleep routines.")
                if feature == "previous_grade" and previous_grade < 60:
                    recommendations.add("Provide long‑term academic coaching.")

            if len(recommendations) == 0:
                recommendations.add("Student habits appear healthy.")

            for r in sorted(recommendations):
                st.write("• " + r)

            st.subheader("SHAP Importance (Which habits matter most)")
            st.dataframe(pd.DataFrame(shap_importance))

    # ============================================================
    # UCI — ML-powered recommendations (no prediction shown)
    # ============================================================
    elif dataset_choice == "uci":
        st.header("UCI Student Input")

        age = st.number_input("Age", 10, 25, 16)
        medu = st.number_input("Mother's Education (0–4)", 0, 4, 2)
        fedu = st.number_input("Father's Education (0–4)", 0, 4, 2)
        traveltime = st.number_input("Travel Time (1–4)", 1, 4, 1)
        studytime = st.number_input("Study Time (1–4)", 1, 4, 2)
        failures = st.number_input("Past Failures (0–3)", 0, 3, 0)
        famrel = st.number_input("Family Relationship (1–5)", 1, 5, 4)
        freetime = st.number_input("Free Time (1–5)", 1, 5, 3)
        goout = st.number_input("Going Out (1–5)", 1, 5, 3)
        dalc = st.number_input("Workday Alcohol (1–5)", 1, 5, 1)
        walc = st.number_input("Weekend Alcohol (1–5)", 1, 5, 2)
        health = st.number_input("Health (1–5)", 1, 5, 3)
        absences = st.number_input("Absences (0–93)", 0, 93, 5)
        g1 = st.number_input("G1 (0–20)", 0, 20, 10)
        g2 = st.number_input("G2 (0–20)", 0, 20, 11)
        grade_level = st.number_input("Grade Level", 7, 13, 10)
        attendance = st.number_input("Attendance (%)", 0.0, 100.0, 90.0)

        raw_uci = {
            "age": age,
            "medu": medu,
            "fedu": fedu,
            "traveltime": traveltime,
            "studytime": studytime,
            "failures": failures,
            "famrel": famrel,
            "freetime": freetime,
            "goout": goout,
            "dalc": dalc,
            "walc": walc,
            "health": health,
            "absences": absences,
            "g1": g1,
            "g2": g2,
            "grade_level": grade_level,
            "attendance": attendance,
            "exam_score": 0
        }

        if st.button("Generate Teacher Recommendations (UCI)"):

            df_engineered = fe_single_uci_student(raw_uci)

            # ML prediction (not shown)
            _ = model.predict(df_engineered)

            # SHAP explanation
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(df_engineered)
            shap_df = pd.DataFrame({
                "Feature": df_engineered.columns,
                "SHAP": shap_vals.flatten()
            }).sort_values("SHAP")

            st.subheader("Teacher Recommendations")

            recommendations = set()

            negative_features = shap_df.head(10)

            for _, row in negative_features.iterrows():
                feature = row["Feature"]
                shap_value = row["SHAP"]

                if shap_value < 0:
                    if "studytime" in feature:
                        recommendations.add("Increase structured study time.")
                    if "failures" in feature:
                        recommendations.add("Provide remediation for students with past failures.")
                    if "absences" in feature:
                        recommendations.add("Reduce absences through early intervention.")
                    if "dalc" in feature or "walc" in feature:
                        recommendations.add("Address alcohol use through counseling.")
                    if "g1" in feature or "g2" in feature:
                        recommendations.add("Use earlier grades as early warning signals.")

            if len(recommendations) == 0:
                recommendations.add("Student appears on track.")

            for r in sorted(recommendations):
                st.write("• " + r)
