import numpy as np

def log_spectrogram(waveform: np.ndarray, n_fft: int = 256, hop: int = 128) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=float)
    if len(waveform) < n_fft: waveform = np.pad(waveform, (0, n_fft - len(waveform)))
    frames = np.lib.stride_tricks.sliding_window_view(waveform, n_fft)[::hop]
    window = np.hanning(n_fft)
    power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2
    return np.log(power + 1e-10)

def standardize(features: np.ndarray) -> np.ndarray:
    return (features - features.mean()) / (features.std() + 1e-8)
