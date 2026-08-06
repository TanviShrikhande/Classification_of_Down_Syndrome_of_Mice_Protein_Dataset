from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


DATASET_FILENAME = "Data_Cortex_Nuclear.csv"


def load_dataset(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load the mice protein dataset and return a processed dataframe."""
    candidates = []
    if data_path is not None:
        candidates.append(Path(data_path))
    candidates.append(Path(DATASET_FILENAME))
    candidates.append(Path(__file__).resolve().parent.parent / DATASET_FILENAME)

    path = None
    for candidate in candidates:
        if candidate.exists():
            path = candidate
            break

    if path is None:
        try:
            from ucimlrepo import fetch_ucirepo

            dataset = fetch_ucirepo(id=342)
            df = dataset.data.features.copy()
            df["class"] = dataset.data.targets.iloc[:, 0].astype(str)
            path = Path(__file__).resolve().parent.parent / DATASET_FILENAME
            df.to_csv(path, index=False)
            path = path
        except Exception as exc:
            raise FileNotFoundError(f"Dataset not found. Tried: {', '.join(str(c) for c in candidates)}") from exc
    else:
        df = pd.read_csv(path)

    df = fill_missing_values(df)
    df = drop_unused_columns(df)
    df = encode_target(df)
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with column means for numeric columns."""
    df = df.copy()
    for column in df.columns[df.isnull().any(axis=0)]:
        df[column] = df[column].fillna(df[column].mean())
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop metadata columns and keep the protein-expression features plus the target."""
    df = df.copy()
    columns_to_drop = [col for col in ["MouseID", "Genotype", "Treatment", "Behavior"] if col in df.columns]
    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the target column to numeric values for model training while preserving the original labels."""
    df = df.copy()
    if "class" in df.columns:
        df.rename(columns={"class": "class_label"}, inplace=True)

    if "class_label" in df.columns:
        if "mice_class" not in df.columns:
            from sklearn import preprocessing

            label_encoder = preprocessing.LabelEncoder()
            df["mice_class"] = label_encoder.fit_transform(df["class_label"])
            df["class_label"] = df["class_label"].astype(str)
        else:
            df["class_label"] = df["mice_class"].astype(str)

    return df


def split_features_target(df: pd.DataFrame, target_column: str = "mice_class") -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target label."""
    feature_columns = [col for col in df.columns if col != target_column and col != "class_label"]
    X = df[feature_columns]
    y = df[target_column]
    if y.dtype == "object":
        from sklearn import preprocessing

        encoder = preprocessing.LabelEncoder()
        y = pd.Series(encoder.fit_transform(y), index=y.index)
    return X, y


def create_group_datasets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create the three group-specific datasets from the notebook."""
    df = df.copy()

    if "class_label" in df.columns:
        class_labels = df["class_label"].astype(str)
    elif "mice_class" in df.columns:
        class_labels = df["mice_class"].astype(str)
    else:
        raise KeyError("Expected either 'class_label' or 'mice_class' in the dataframe")

    groups = {
        "normal_learning": df[class_labels.isin(["c-CS-m", "c-CS-s", "c-SC-m", "c-SC-s"])],
        "trisomy_success_vs_failure": df[class_labels.isin(["t-CS-m", "t-CS-s"])],
        "normal_vs_trisomy_failure": df[class_labels.isin(["t-CS-s", "c-CS-m", "c-CS-s"])],
    }
    return groups
