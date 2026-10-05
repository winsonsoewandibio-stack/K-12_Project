"""
ingestion.py
------------
This module implements the DataIngestionPipeline, responsible for:

1. Loading raw CSV/JSON files.
2. Measuring ingestion performance (NFR).
3. Detecting anomalies (missing values, negative numbers).
4. Encrypting the ingested dataset (security NFR).
5. Returning both the cleaned DataFrame and ingestion metrics.

The ingestion pipeline NEVER modifies schema — it only loads and
validates. SIS mapping happens in the dataset loaders.
"""

import pandas as pd
import time
from cryptography.fernet import Fernet

from src.data_ingestion.common_scheme import SIS_REQUIRED_FIELDS, SUPPORTED_FORMATS, NFR_THRESHOLDS

# Supported formats for ingestion
SUPPORTED_FORMATS = ["csv", "json"]


class DataIngestionPipeline:
    """
    DataIngestionPipeline
    ---------------------
    This class enforces ingestion NFRs:
    - Performance (ingestion_time)
    - Reliability (failure detection)
    - Scalability (record_count)
    - Security (AES encryption)
    - Schema validation (SIS fields added later)

    The pipeline loads the dataset and produces a metrics dictionary.
    """

    def __init__(self, path: str):
        self.path = path

        # Initialize metrics dictionary
        self.metrics = {
            "ingestion_time": None,
            "failure": False,
            "record_count": None,
            "missing_sis_fields": [],
            "anomaly_detected": False,
            "encrypted": False,
        }

        # Generate encryption key for security NFR
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def _detect_format(self):
        """Detect file format based on extension."""
        if self.path.endswith(".csv"):
            return "csv"
        if self.path.endswith(".json"):
            return "json"
        raise ValueError(f"Unsupported file format: {self.path}")

    def load(self):
        """
        Load the dataset, measure ingestion time, detect anomalies,
        encrypt the dataset, and return the DataFrame.
        """

        start_time = time.time()

        try:
            file_format = self._detect_format()

            if file_format == "csv":
                df = pd.read_csv(self.path)
            else:
                df = pd.read_json(self.path)

        except Exception:
            self.metrics["failure"] = True
            raise

        end_time = time.time()
        self.metrics["ingestion_time"] = end_time - start_time

        # Record count (scalability NFR)
        self.metrics["record_count"] = len(df)

        # Detect anomalies (missing values or negative numbers)
        numeric_cols = df.select_dtypes(include=["number"])
        self.metrics["anomaly_detected"] = (
            df.isna().sum().sum() > 0 or (numeric_cols < 0).any().any()
        )

        # Encrypt dataset (security NFR)
        encrypted_bytes = self.cipher.encrypt(df.to_csv(index=False).encode())
        with open("data/encrypted_ingestion.bin", "wb") as f:
            f.write(encrypted_bytes)

        self.metrics["encrypted"] = True

        return df
