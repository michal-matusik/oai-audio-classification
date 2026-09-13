"""Fit on official training data and evaluate on official validation data."""

import argparse

from .dataset import load_split
from .metrics import balanced_accuracy, task_score
from .train import train


def evaluate(train_path: str, validation_path: str) -> dict[str, float | int]:
    model = train(train_path)
    signals, labels, _ = load_split(validation_path)
    metric = balanced_accuracy(labels, model.predict(signals))
    return {"balanced_accuracy": metric, "estimated_task_score": task_score(metric), "samples": len(labels)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_npz")
    parser.add_argument("validation_npz")
    args = parser.parse_args()
    print(evaluate(args.train_npz, args.validation_npz))
