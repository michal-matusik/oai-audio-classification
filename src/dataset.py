"""Official NPZ dataset loading."""

from pathlib import Path

import numpy as np


def load_split(path: str | Path) -> tuple[list[np.ndarray], np.ndarray, int]:
    payload = np.load(path, allow_pickle=True)
    signals = [np.asarray(signal, dtype=np.float32) for signal in payload["signals"]]
    return signals, payload["labels"].astype(np.int64), int(payload["sr"])
