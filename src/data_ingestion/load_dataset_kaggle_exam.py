# src/data_ingestion/load_dataset_kaggle_exam.py

import pandas as pd
import kagglehub
import os
import shutil
from src.data_ingestion.common_scheme import UNIFIED_SCHEMA, convert_letter_grade


def load_kaggle_exam_performance():
    # Mapping notes:
    # study_hours_per_day → study_hours_per_day
    # attendance_rate → attendance_rate
    # previous_exam_score → previous_exam_score
    # final_exam_score → final_exam_score
    # performance_grade → final_grade_letter

    downloaded = kagglehub.dataset_download(
        "mobeenfatimah/student-exam-performance-and-success-dataset"
    )

    project_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "kaggle",
                     "mobeenfatimah", "student-exam-performance-and-success-dataset")
    )
    if not os.path.exists(project_dir):
        shutil.copytree(downloaded, project_dir)

    csv_files = [f for f in os.listdir(project_dir) if f.endswith(".csv")]
    df = pd.read_csv(os.path.join(project_dir, csv_files[0]))

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    mapped = {col: None for col in UNIFIED_SCHEMA}

    mapped["student_id"] = df["student_id"]
    mapped["age"] = df["age"]
    mapped["gender"] = df["gender"]
    mapped["education_level"] = df["education_level"]
    mapped["school_type"] = df["school_type"]
    mapped["family_income"] = df["family_income"]
    mapped["parental_education"] = df["parent_education"]
    mapped["urban_rural"] = df["urban_rural"]

    mapped["previous_exam_score"] = df["previous_exam_score"]
    mapped["previous_gpa"] = df["previous_gpa"]

    mapped["attendance_percentage"] = df["attendance_percentage"]
    mapped["assignment_completion_rate"] = df["assignment_completion_rate"]
    mapped["class_participation"] = df["class_participation"]

    mapped["study_hours_per_day"] = df["study_hours_per_day"]
    mapped["self_study_hours"] = df["self_study_hours"]
    mapped["private_tuition"] = df["private_tuition"]
    mapped["online_learning_hours"] = df["online_learning_hours"]
    mapped["study_consistency"] = df["study_consistency"]
    mapped["study_environment"] = df["study_environment"]
    mapped["study_method"] = df["study_method"]
    mapped["revision_frequency"] = df["revision_frequency"]
    mapped["practice_tests_completed"] = df["practice_tests_completed"]
    mapped["notes_quality"] = df["notes_quality"]

    mapped["sleep_hours"] = df["sleep_hours"]
    mapped["sleep_quality"] = df["sleep_quality"]
    mapped["daily_screen_time"] = df["daily_screen_time"]
    mapped["physical_activity_hours"] = df["physical_activity_hours"]
    mapped["break_frequency"] = df["break_frequency"]
    mapped["stress_level"] = df["stress_level"]
    mapped["motivation_level"] = df["motivation_level"]

    mapped["internet_access"] = df["internet_access"]
    mapped["device_availability"] = df["device_availability"]
    mapped["educational_app_usage"] = df["educational_app_usage"]
    mapped["online_course_hours"] = df["online_course_hours"]

    mapped["exam_preparation_days"] = df["exam_preparation_days"]
    mapped["questions_attempted"] = df["questions_attempted"]
    mapped["questions_correct"] = df["questions_correct"]
    mapped["time_management_score"] = df["time_management_score"]
    mapped["exam_anxiety_level"] = df["exam_anxiety_level"]

    mapped["final_exam_score"] = df["exam_score"]
    mapped["final_grade_letter"] = df["performance_grade"]
    mapped["final_grade_numeric"] = df["performance_grade"].apply(convert_letter_grade)
    mapped["pass_status"] = df["pass_status"]
    mapped["performance_level"] = df["performance_level"]

    mapped["source_dataset"] = "kaggle_exam_performance"

    return pd.DataFrame(mapped)
