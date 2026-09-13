"""Train and serialize the reconstructed classifier."""

import argparse
from pathlib import Path

import joblib

from .dataset import load_split
from .model import AudioClassifier


def train(train_path: str) -> AudioClassifier:
    signals, labels, sample_rate = load_split(train_path)
    return AudioClassifier(sample_rate).fit(signals, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_npz")
    parser.add_argument("--output", default="artifacts/model.joblib")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(train(args.train_npz), args.output)
