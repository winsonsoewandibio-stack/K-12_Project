# src/data_ingestion/load_dataset_kaggle_habits.py

import pandas as pd
import kagglehub
import os
import shutil
from src.data_ingestion.common_scheme import UNIFIED_SCHEMA, convert_letter_grade


def load_kaggle_student_habits():
    # Mapping notes:
    # study_time_hours → study_time_hours
    # attendance_percent → attendance_percentage
    # previous_grade → previous_grade_letter
    # final_exam_score → final_exam_score
    # final_grade → final_grade_letter

    downloaded = kagglehub.dataset_download(
        "harshadapatil31/student-performance-and-study-habits-dataset"
    )

    project_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "kaggle",
                     "harshadapatil31", "student-performance-and-study-habits-dataset")
    )
    if not os.path.exists(project_dir):
        shutil.copytree(downloaded, project_dir)

    csv_files = [f for f in os.listdir(project_dir) if f.endswith(".csv")]
    df = pd.read_csv(os.path.join(project_dir, csv_files[0]))

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    mapped = {col: None for col in UNIFIED_SCHEMA}

    mapped["student_id"] = df["student_id"]
    mapped["gender"] = df["gender"]
    mapped["study_time_hours"] = df["study_time_hours"]
    mapped["attendance_percentage"] = df["attendance_percent"]
    mapped["sleep_hours"] = df["sleep_hours"]
    mapped["parental_education"] = df["parental_education"]
    mapped["internet_access"] = df["internet_access"]
    mapped["extracurricular_activities"] = df["extracurricular_activities"]
    mapped["part_time_job"] = df["part_time_job"]
    mapped["previous_grade_letter"] = df["previous_grade"]
    mapped["previous_grade_numeric"] = df["previous_grade"].apply(convert_letter_grade)
    mapped["final_exam_score"] = df["final_exam_score"]
    mapped["final_grade_letter"] = df["final_grade"]
    mapped["final_grade_numeric"] = df["final_grade"].apply(convert_letter_grade)

    mapped["source_dataset"] = "kaggle_study_habits"

    return pd.DataFrame(mapped)
