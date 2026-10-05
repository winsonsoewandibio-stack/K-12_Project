"""
common_scheme.py
----------------
This module defines the universal schema rules, SIS-required fields,
canonical column mappings, supported ingestion formats, and NFR thresholds.

All ingestion loaders import from this file to ensure consistency.
"""

# -------------------------------------------------------------------
# SIS REQUIRED FIELDS
# -------------------------------------------------------------------
# These fields MUST exist after ingestion. They are added on top of
# the dataset (non-destructive) and never replace or remove original
# columns.
SIS_REQUIRED_FIELDS = [
    "student_id",
    "grade_level",
    "attendance",
    "exam_score",
]

# -------------------------------------------------------------------
# SUPPORTED FILE FORMATS
# -------------------------------------------------------------------
# The ingestion pipeline uses this list to validate input file types.
# It is imported by ingestion.py, so it MUST exist here.
SUPPORTED_FORMATS = ["csv", "json"]

# -------------------------------------------------------------------
# CANONICAL COLUMN MAPPING
# -------------------------------------------------------------------
# This mapping is intentionally minimal. It only normalizes columns
# that appear with inconsistent naming across datasets.
CANONICAL_COLUMNS = {
    # UCI dataset canonical fields
    "G1": "G1",
    "G3": "G3",
    "absences": "absences",

    # Kaggle exam dataset canonical fields
    "attendance_percentage": "attendance_percentage",
    "education_level": "education_level",
    "exam_score": "exam_score",

    # Kaggle habits dataset canonical fields
    "attendance_percent": "attendance_percent",
    "previous_grade": "previous_grade",
    "final_exam_score": "final_exam_score",
}

# -------------------------------------------------------------------
# NFR THRESHOLDS
# -------------------------------------------------------------------
# These thresholds define acceptable ingestion performance.
# They are used in the ingestion test suite.
NFR_THRESHOLDS = {
    "ingestion_time_max": 5.0,     # seconds
    "max_records": 100000,         # scalability limit
}
