import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os

# -----------------------------
# Safe loaders
# -----------------------------
def safe_load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None

def safe_load_image(path):
    return path if os.path.exists(path) else None

def safe_load_npy(path):
    return np.load(path) if os.path.exists(path) else None

def safe_load_model(path):
    return joblib.load(path) if os.path.exists(path) else None


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
            f"RMSE = {rmse:.6f}. The model’s average prediction error is extremely small — "
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
        interpretation.append("This is an exceptionally strong model — almost no unexplained variance.")

    if train_time is not None:
        interpretation.append(
            f"Training time = {train_time:.2f} seconds. This reflects heavy feature engineering."
        )
        if train_time > 300:
            actions.append("Reduce interaction features to shorten training time and improve efficiency.")

    if num_features is not None:
        interpretation.append(
            f"The model uses {num_features} engineered features — a very high number."
        )
        if num_features > 250:
            actions.append(
                "High feature count may reduce explainability. Consider simplifying feature engineering."
            )

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
        f"Interaction features: {interactions}. This indicates heavy feature engineering."
    )

    if interactions > 200:
        actions.append("Reduce interaction features to improve explainability for teachers.")

    if fe_time > 1.0:
        interpretation.append(
            f"Feature engineering took {fe_time:.2f} seconds — acceptable but indicates complexity."
        )

    if total > 250:
        actions.append(
            "Consider simplifying the feature set to improve transparency and reduce overfitting risk."
        )

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
        elif "attendance_percentage" in f:
            actions.append("• Low attendance should trigger early intervention and parent communication.")
        elif "assignment_completion_rate" in f:
            actions.append("• Encourage consistent assignment completion; it strongly predicts exam outcomes.")
        elif "study_hours" in f:
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
            actions.append(f"• Monitor **{feature}** — it has meaningful impact on exam outcomes.")

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
            actions.append(
                "Review misclassified students manually, especially those near category boundaries."
            )

    if train_time is not None:
        interpretation.append(
            f"Training time = {train_time:.3f} seconds — very fast, suitable for frequent retraining."
        )

    if num_features is not None:
        interpretation.append(
            f"The HABITS model uses {num_features} features — a compact and explainable feature set."
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
            f"Feature engineering time = {fe_time:.3f} seconds — very efficient."
        )

    if interactions > 30:
        actions.append(
            "Monitor interaction features to ensure they remain interpretable for teachers."
        )

    if total <= 50:
        actions.append(
            "The compact feature set supports clear communication of habit drivers to teachers."
        )

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
# UCI-SPECIFIC INTERPRETATION (final_grade, 0–20 Portuguese scale)
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
            f"Training time = {train_time:.3f} seconds — fast enough for regular retraining as new data arrives."
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
            f"Feature engineering time = {fe_time:.3f} seconds — reasonable given the high number of interactions."
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
            actions.append(
                "• Increase structured study time for students with low studytime; it strongly improves final grades."
            )
        elif "failures" in f:
            actions.append(
                "• Students with previous failures need targeted remediation and close monitoring to avoid repeating."
            )
        elif "absences" in f:
            actions.append(
                "• High absences are strongly linked to lower final grades; intervene early on attendance issues."
            )
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
            actions.append(
                "• Maintain strong attendance; it is directly tied to staying in ‘Bom’ or ‘Muito Bom’ bands."
            )
        elif "exam_score" in f or "grade_level" in f:
            actions.append(
                "• Use exam scores and grade level to tailor expectations and support; students near critical thresholds "
                "need extra guidance."
            )
        else:
            actions.append(f"• Monitor **{feature}** as a meaningful contributor to final grade outcomes.")

    return findings, actions


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
        "Feature Importance"
    ]
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
        st.write("""
        • The model still provides SHAP values even without a summary PNG.  
        • You can rely on the Feature Importance page for detailed insights.  
        """)

    st.subheader("SHAP Beeswarm Plot")
    if shap_beeswarm:
        st.image(shap_beeswarm)
    else:
        st.info("No SHAP beeswarm plot available for this dataset.")
        st.write("""
        • Beeswarm plots visualize how each feature pushes predictions up or down.  
        • Since this PNG is missing, refer to the Feature Importance page for ranked SHAP contributions.  
        """)

    if dataset_choice == "exam":
        st.subheader("Teacher Interpretation (EXAM Dataset)")
        st.write("""
        • SHAP plots show how each feature pushes exam score predictions up or down.  
        • Red = higher feature values; Blue = lower feature values.  
        • The most influential features (e.g., previous exam score, GPA, attendance, study habits) appear at the top.  
        """)

        st.subheader("Recommended Actions for Teachers")
        st.write("""
        • Focus on the top SHAP features such as previous exam score, GPA, attendance, and study habits.  
        • Use SHAP insights to personalize academic support and intervention plans.  
        """)

    elif dataset_choice == "habits":
        st.subheader("Teacher Interpretation (HABITS Dataset)")
        st.write("""
        • SHAP values show how each habit (study time, attendance, sleep, previous grade, exam score)  
          influences the predicted habit category.  
        • Even without PNGs, SHAP values still quantify how strongly each habit affects classification.  
        • Study time, attendance, and sleep hours are typically the strongest drivers.  
        """)

        st.subheader("Recommended Actions for Teachers")
        st.write("""
        • Encourage consistent daily study routines — SHAP shows study time is a major habit driver.  
        • Maintain high attendance; poor attendance strongly harms habit quality.  
        • Promote healthy sleep schedules; sleep hours strongly affect learning effectiveness.  
        • Use previous grades to identify students needing long-term habit support.  
        • Connect exam performance back to habits to motivate students to improve routines.  
        """)

    elif dataset_choice == "uci":
        st.subheader("Teacher Interpretation (UCI Final Grade Dataset)")
        st.write("""
        • SHAP values show how each factor (studytime, failures, absences, alcohol use, family relations, etc.)  
          influences the predicted final grade (G3) on the 0–20 Portuguese scale.  
        • Positive SHAP values push students toward higher bands (e.g., ‘Bom’, ‘Muito Bom’, ‘Excelente’).  
        • Negative SHAP values push students toward lower bands (e.g., ‘Insuficiente’, ‘Mau’).  
        """)

        st.subheader("Recommended Actions for Teachers")
        st.write("""
        • Use SHAP insights to identify which factors most strongly push a student toward failing bands.  
        • Prioritize interventions on studytime, absences, failures, and alcohol use where SHAP shows strong influence.  
        • Pay special attention to students near the 9.5 passing threshold, where small habit changes can shift bands.  
        """)

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
                "Importance": list(collapsed.values())
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
                "Importance": list(collapsed.values())
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
