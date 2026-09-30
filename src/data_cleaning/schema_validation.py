"""
Schema validation for cleaned datasets.

Ensures each dataset contains the expected columns after cleaning.
This prevents silent ingestion changes from breaking downstream steps.
"""

EXPECTED_UCI_COLUMNS = {
    "gender",
    "age",
    "study_hours_per_day",
    "sleep_hours",
    "attendance_percentage",
    "final_exam_score",
}

EXPECTED_KAGGLE_HABITS_COLUMNS = {
    "gender",
    "age",
    "study_hours_per_day",
    "sleep_hours",
    "internet_access",
    "attendance_percentage",
}

EXPECTED_KAGGLE_EXAM_COLUMNS = {
    "gender",
    "age",
    "study_hours_per_day",
    "sleep_hours",
    "attendance_percentage",
    "final_exam_score",
}


def validate_schema(df, expected_columns, dataset_name):
    """
    Validate that the cleaned dataset contains the expected columns.
    Missingness indicators are not required in the expected schema.
    """
    df_columns = set(df.columns)
    missing = expected_columns - df_columns

    if missing:
        raise AssertionError(
            f"{dataset_name} is missing required columns: {missing}"
        )

    return True
