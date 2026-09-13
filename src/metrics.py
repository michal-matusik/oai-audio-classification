"""Metric used by the official task."""

from sklearn.metrics import balanced_accuracy_score


def balanced_accuracy(target, prediction) -> float:
    return float(balanced_accuracy_score(target, prediction))


def task_score(metric: float) -> int:
    return round(max(0.0, min(1.0, (metric - 0.60) / (0.99 - 0.60))) * 100)
