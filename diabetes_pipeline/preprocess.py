from __future__ import annotations

import pandas as pd
from imblearn.under_sampling import NearMiss

FEATURE_COLUMNS = [
    "Diabetes_binary",
    "HighBP",
    "HighChol",
    "BMI",
    "Stroke",
    "HeartDiseaseorAttack",
    "GenHlth",
    "PhysHlth",
    "DiffWalk",
    "Age",
    "PhysActivity",
    "Education",
    "Income",
]


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and cast BMI to int to match notebook flow."""
    cleaned = df.drop_duplicates().copy()
    if "BMI" in cleaned.columns:
        cleaned["BMI"] = cleaned["BMI"].astype(int)
    return cleaned


def get_percentage_table(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Return row-wise percentage table for feature vs Diabetes_binary."""
    result = pd.crosstab(data[column_name], data["Diabetes_binary"], normalize="index")
    return result * 100


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep model columns used in notebook."""
    return df[FEATURE_COLUMNS].copy()


def balance_with_nearmiss(
    data: pd.DataFrame, target: str = "Diabetes_binary", n_neighbors: int = 10
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply NearMiss undersampling."""
    x = data.drop(target, axis=1)
    y = data[target]
    nm = NearMiss(version=1, n_neighbors=n_neighbors)
    x_sm, y_sm = nm.fit_resample(x, y)
    return x_sm, y_sm

