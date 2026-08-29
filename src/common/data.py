from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

from src.common.config import (
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
)


def parse_labels(value: object) -> list[str]:
    if value is None or (
        isinstance(value, float) and np.isnan(value)
    ):
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    parsed = json.loads(str(value))

    if not isinstance(parsed, list):
        raise ValueError(f"cwe_ids non è una lista JSON: {value!r}")

    return [str(item).strip() for item in parsed if str(item).strip()]


def clean_description(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset non trovato: {dataset_path}")

    df = pd.read_csv(dataset_path, low_memory=False)

    required_columns = {"cve_id", "description", "cwe_ids"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Colonne mancanti: {sorted(missing)}")

    df = df.copy()
    df["labels"] = df["cwe_ids"].apply(parse_labels)
    df["clean_description"] = df["description"].apply(clean_description)

    df = df[
        df["clean_description"].ne("")
        & df["labels"].map(bool)
    ].reset_index(drop=True)

    return df


def get_all_classes(df: pd.DataFrame) -> list[str]:
    return sorted(
        {
            label
            for labels in df["labels"]
            for label in labels
        }
    )


def split_dataset(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Rappresentazione binaria multilabel usata solo per lo splitting
    classes = get_all_classes(df)

    mlb = MultiLabelBinarizer(classes=classes)
    y = mlb.fit_transform(df["labels"])

    # 1. Train+Validation / Test
    test_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=random_state,
    )

    train_val_idx, test_idx = next(
        test_splitter.split(np.zeros(len(df)), y)
    )

    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    y_train_val = y[train_val_idx]

    # 2. Train / Validation
    relative_validation_size = VALIDATION_SIZE / (1.0 - TEST_SIZE)

    validation_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=relative_validation_size,
        random_state=random_state,
    )

    train_idx, validation_idx = next(
        validation_splitter.split(
            np.zeros(len(train_val_df)),
            y_train_val,
        )
    )

    train_df = train_val_df.iloc[train_idx]
    validation_df = train_val_df.iloc[validation_idx]

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_targets(
    classes: list[str],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[MultiLabelBinarizer, np.ndarray, np.ndarray, np.ndarray]:
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])

    y_train = mlb.transform(train_df["labels"])
    y_validation = mlb.transform(validation_df["labels"])
    y_test = mlb.transform(test_df["labels"])

    missing_train_classes = np.asarray(classes)[y_train.sum(axis=0) == 0]

    if len(missing_train_classes):
        raise ValueError(
            "Classi assenti dal training split: "
            + ", ".join(missing_train_classes)
        )

    return mlb, y_train, y_validation, y_test
