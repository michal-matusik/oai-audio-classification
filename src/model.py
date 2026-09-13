"""Mel-spectral summary features with an RBF support-vector classifier."""

import numpy as np
import torch
import torchaudio
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class MelFeatureExtractor:
    def __init__(self, sample_rate: int = 16_000, n_mels: int = 64):
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    @torch.no_grad()
    def transform(self, signals: list[np.ndarray]) -> np.ndarray:
        rows = []
        for signal in signals:
            waveform = torch.as_tensor(signal, dtype=torch.float32)
            spectrum = self.to_db(self.mel(waveform) + 1e-10)
            chunks = torch.tensor_split(spectrum, 4, dim=1)
            stats = torch.stack([waveform.mean(), waveform.std(), waveform.abs().max(), (waveform[1:] * waveform[:-1] < 0).float().mean()])
            rows.append(torch.cat([spectrum.mean(1), spectrum.std(1), *(chunk.mean(1) for chunk in chunks), stats]).numpy())
        return np.stack(rows)


class AudioClassifier:
    def __init__(self, sample_rate: int = 16_000):
        self.features = MelFeatureExtractor(sample_rate=sample_rate)
        self.classifier = make_pipeline(StandardScaler(), SVC(C=1.0))

    def fit(self, signals: list[np.ndarray], labels: np.ndarray):
        self.classifier.fit(self.features.transform(signals), labels)
        return self

    def predict(self, signals: list[np.ndarray]) -> np.ndarray:
        return self.classifier.predict(self.features.transform(signals))
