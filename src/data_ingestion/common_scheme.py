# src/data_ingestion/common_schema.py

# common_scheme.py
# Unified schema for UCI, Kaggle Study Habits, and Kaggle Exam Performance datasets.

UNIFIED_SCHEMA = [
    # Identity & Demographics
    "student_id", "age", "gender", "education_level", "school_type",
    "urban_rural", "family_income", "address_type", "family_size",
    "parent_status", "guardian", "source_dataset",

    # Family Background
    "parental_education", "mother_education", "father_education",
    "mother_job", "father_job", "family_relationship_quality",
    "family_support", "school_support", "paid_classes", "nursery",
    "higher_education_aspiration", "romantic_relationship",

    # Academic History & Performance
    "previous_exam_score", "previous_gpa", "previous_grade_letter",
    "previous_grade_numeric", "G1", "G2", "G3",
    "final_exam_score", "final_grade_letter", "final_grade_numeric",
    "performance_grade", "performance_level", "pass_status",
    "exam_difficulty",

    # Attendance & Engagement
    "attendance_percentage", "attendance_rate", "absences",
    "assignment_completion_rate", "class_participation",
    "extracurricular_activities", "part_time_job", "internet_access",
    "device_availability", "educational_app_usage", "online_course_hours",

    # Study Behavior & Learning Habits
    "study_hours_per_day", "self_study_hours", "study_time_hours",
    "weekly_study_time", "study_consistency", "study_environment",
    "study_method", "revision_frequency", "practice_tests_completed",
    "notes_quality", "online_learning_hours", "traveltime", "failures",

    # Well‑Being & Lifestyle
    "sleep_hours", "sleep_quality", "daily_screen_time",
    "physical_activity_hours", "break_frequency", "stress_level",
    "motivation_level", "health", "alcohol_use_workday",
    "alcohol_use_weekend", "free_time", "go_out",

    # Exam Behavior & Analytics
    "exam_preparation_days", "questions_attempted", "questions_correct",
    "time_management_score", "exam_anxiety_level",

    # Derived Fields
    "grade_level", "attendance_normalized", "study_efficiency_score",
    "engagement_score", "risk_score", "performance_category",
]

GRADE_MAP = {
    "A": 90, "B": 80, "C": 70, "D": 60, "F": 50
}

def convert_letter_grade(letter):
    if letter is None:
        return None

    # If numeric, return numeric bucket
    if isinstance(letter, (int, float)):
        if letter >= 90:
            return 90
        elif letter >= 80:
            return 80
        elif letter >= 70:
            return 70
        elif letter >= 60:
            return 60
        else:
            return 50

    # If string letter grade
    return GRADE_MAP.get(letter.upper(), None)
