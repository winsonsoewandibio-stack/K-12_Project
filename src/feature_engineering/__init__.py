"""
FEATURE ENGINEERING PACKAGE

This package contains all modules responsible for transforming CLEANED data
(from src.data_cleaning) into FAIR, INTERPRETABLE, SHAP‑READY engineered features.

The modules inside this folder follow a strict separation of concerns:

- feature_utils.py
      Low‑level helper functions (math, normalization, bucketing).
      No domain logic here — only reusable utilities.

- feature_rules.py
      The HEART of feature engineering.
      Defines WHAT features are created and WHY.
      All features are aligned with:
          • fairness constraints
          • SHAP interpretability
          • K‑12 early‑warning domain logic
          • the actual cleaned dataset schema

- feature_engineering.py
      Pipeline orchestrator.
      Loads CLEANED datasets → applies feature rules → validates schema → saves output.

- schema_feature_validation.py
      Ensures engineered datasets contain ALL required features.
      Mandatory features must always exist.
      Conditional features only required if the base column exists in cleaned data.
"""
