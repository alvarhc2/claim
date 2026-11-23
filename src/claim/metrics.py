"""Scores used in the manuscript.

Sensitivity is the recall of the positive class and specificity is the recall
of the negative class. AUC uses the positive-class probability. A split that
contains only one class has an undefined AUC.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def classification(labels: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    predicted = np.asarray(predicted).astype(int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive = labels == 1
    negative = ~positive
    true_positive = int(np.sum(positive & (predicted == 1)))
    false_negative = int(np.sum(positive & (predicted == 0)))
    true_negative = int(np.sum(negative & (predicted == 0)))
    false_positive = int(np.sum(negative & (predicted == 1)))
    total = max(len(labels), 1)
    sensitivity = _ratio(true_positive, true_positive + false_negative)
    precision = _ratio(true_positive, true_positive + false_positive)
    return {
        "accuracy": (true_positive + true_negative) / total,
        "sensitivity": sensitivity,
        "specificity": _ratio(true_negative, true_negative + false_positive),
        "precision": precision,
        "f1": _ratio(2 * precision * sensitivity, precision + sensitivity),
        "auc": _auc(labels, probabilities),
    }


def mean_std(rows: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    summary = {}
    for key in ("accuracy", "sensitivity", "specificity", "precision", "f1", "auc"):
        values = np.array([row[key] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            summary[key] = (float("nan"), float("nan"))
        else:
            summary[key] = (float(finite.mean()), float(finite.std(ddof=0)))
    return summary


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))
