"""
__init__.py
-----------
Public API for the feature engineering module.
"""

from .feature_engineering import FeatureEngineeringPipeline
from .feature_rules import ENCODING_RULES, SCALING_RULES, FEATURE_NFR_THRESHOLDS
from .feature_utils import encode_categorical, scale_numeric, create_interaction_features
from .schema_validation import validate_feature_schema
