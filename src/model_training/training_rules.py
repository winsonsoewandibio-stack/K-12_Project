"""
training_rules.py
-----------------
Non-Functional Requirements (NFRs) and training rules
for the model training module.
"""

TRAINING_NFR_THRESHOLDS = {
    "training_time_max": 1000,   # increased from 120 → 1000 due to dataset "exam" size
    "min_accuracy": 0.70,
    "max_features": 300,
}
