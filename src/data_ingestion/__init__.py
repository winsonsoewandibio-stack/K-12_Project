"""
__init__.py
-----------
This file exposes the public API of the data_ingestion module.

IMPORTANT:
- Only import objects that actually exist.
- FEATURE_DEPENDENCIES and SCHEMA_RULES were removed because
  they belong to feature engineering, not ingestion.
"""

from .ingestion import DataIngestionPipeline

# Canonical schema + SIS fields + NFR thresholds
from .common_scheme import (
    CANONICAL_COLUMNS,
    SIS_REQUIRED_FIELDS,
    SUPPORTED_FORMATS,
    NFR_THRESHOLDS,
)

# Dataset loaders
from .load_dataset_uci import load_uci_dataset
from .load_dataset_kaggle_habit import load_kaggle_student_habits
from .load_dataset_kaggle_exam import load_kaggle_exam_performance
