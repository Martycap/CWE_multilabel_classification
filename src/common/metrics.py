from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


@dataclass
class Metrics:
    threshold: float
    micro_f1: float
    macro_f1: float
    weighted_f1: float
    micro_precision: float
    micro_recall: float
    hamming_loss: float
    subset_accuracy: float


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Metrics:
    y_pred = (y_score >= threshold).astype(np.int8)

    return Metrics(
        threshold=float(threshold),
        micro_f1=float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        macro_f1=float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        weighted_f1=float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        micro_precision=float(
            precision_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        micro_recall=float(
            recall_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        hamming_loss=float(hamming_loss(y_true, y_pred)),
        subset_accuracy=float(np.mean(np.all(y_true == y_pred, axis=1))),
    )


def select_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    best_threshold = float(thresholds[0])
    best_macro_f1 = -1.0
    history: list[dict[str, float]] = []

    for threshold in thresholds:
        metrics = compute_metrics(y_true, y_score, float(threshold))
        history.append(asdict(metrics))

        if metrics.macro_f1 > best_macro_f1:
            best_macro_f1 = metrics.macro_f1
            best_threshold = float(threshold)

    return best_threshold, history