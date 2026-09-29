# src/data_ingestion/load_dataset_uci.py

import pandas as pd
import kagglehub
import os
import shutil
from src.data_ingestion.common_scheme import UNIFIED_SCHEMA, convert_letter_grade


def load_uci_student_performance():
    # Mapping notes:
    # sex → gender
    # address → address_type
    # famsize → family_size
    # Pstatus → parent_status
    # Medu/Fedu → mother_education / father_education
    # Mjob/Fjob → mother_job / father_job
    # studytime → weekly_study_time
    # Dalc/Walc → alcohol_use_workday / alcohol_use_weekend
    # G1/G2/G3 → previous + final exam scores

    project_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "uci", "student_performance")
    )
    os.makedirs(project_dir, exist_ok=True)

    file_path = os.path.join(project_dir, "student_performance.csv")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=320)
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        df.to_csv(file_path, index=False)

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    mapped = {col: None for col in UNIFIED_SCHEMA}

    mapped["student_id"] = df.index + 1
    mapped["age"] = df["age"]
    mapped["gender"] = df["sex"]
    mapped["address_type"] = df["address"]
    mapped["family_size"] = df["famsize"]
    mapped["parent_status"] = df["pstatus"]
    mapped["mother_education"] = df["medu"]
    mapped["father_education"] = df["fedu"]
    mapped["mother_job"] = df["mjob"]
    mapped["father_job"] = df["fjob"]
    mapped["study_environment"] = df["reason"]
    mapped["guardian"] = df["guardian"]
    mapped["traveltime"] = df["traveltime"]
    mapped["weekly_study_time"] = df["studytime"]
    mapped["failures"] = df["failures"]
    mapped["school_support"] = df["schoolsup"]
    mapped["family_support"] = df["famsup"]
    mapped["paid_classes"] = df["paid"]
    mapped["extracurricular_activities"] = df["activities"]
    mapped["nursery"] = df["nursery"]
    mapped["higher_education_aspiration"] = df["higher"]
    mapped["internet_access"] = df["internet"]
    mapped["romantic_relationship"] = df["romantic"]
    mapped["family_relationship_quality"] = df["famrel"]
    mapped["free_time"] = df["freetime"]
    mapped["go_out"] = df["goout"]
    mapped["alcohol_use_workday"] = df["dalc"]
    mapped["alcohol_use_weekend"] = df["walc"]
    mapped["health"] = df["health"]
    mapped["absences"] = df["absences"]

    mapped["G1"] = df["g1"]
    mapped["G2"] = df["g2"]
    mapped["G3"] = df["g3"]
    mapped["final_exam_score"] = df["g3"]

    mapped["source_dataset"] = "uci_student_performance"

    return pd.DataFrame(mapped)

