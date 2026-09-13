# Reconstructed Audio Classification - Polish AI Olympiad III

This repository is a reconstructed reference solution for the first-stage Audio Classification task in the Polish Artificial Intelligence Olympiad.


The pipeline converts waveforms into mel-spectral summary features and classifies them with a standardized radial-basis support-vector model.
It is deterministic, CPU-friendly, and has no hidden notebook state.

![Audio-classification task illustration](assets/task-audio.png)

*Task illustration embedded in the official Polish AI Olympiad III notebook.*

## Method

Each waveform is converted to a 64-band log-mel spectrogram (`n_fft=1024`, `hop_length=512`,
`AmplitudeToDB`), which approximates the ear's roughly-logarithmic frequency resolution and
compresses dynamic range the way loudness perception does. Rather than feeding the full
spectrogram to a CNN, it is reduced to a fixed-length descriptor: per-band mean and standard
deviation over the whole clip (overall spectral shape and its variability), per-band means over
4 equal temporal chunks (a coarse loudness/spectral envelope over time), and four raw-waveform
statistics — mean, standard deviation, peak absolute amplitude, and zero-crossing rate. The last
is a classical proxy for voiced-vs-unvoiced/noise-like content (high for breathy, aperiodic
whispering; lower and more structured for voiced screaming or normal speech), and is the single
most task-specific feature in the vector.

This fixed-length, hand-crafted descriptor is classified with a standardized RBF-kernel SVM
(`gamma='scale'`, `C=1.0`): the Gaussian kernel implicitly maps the feature vector into an
infinite-dimensional space where the three classes become linearly separable, without needing to
learn that mapping from data the way a CNN would. Choosing a kernel method over a
spectrogram-input CNN (which the task description otherwise gestures toward) is a deliberate
bias-variance call: with only 2,400 training clips, a compact engineered feature space plus a
kernel classifier is far less prone to overfitting than a convolutional model learning its own
filters from scratch, and here it reaches the metric's ceiling directly (measured balanced
accuracy 1.0000). The score (`src/metrics.py: task_score`) linearly rescales balanced accuracy
from 0.60 to 0.99 across 0-100 points, a demanding curve this pipeline saturates.

## Quick start

```bash
python scripts/download_data.py
python -m src.evaluate data/train.npz data/val.npz
python -m unittest discover -s tests -v
```

## Validation

The model reaches 1.0000 balanced accuracy on all 200 recordings in the official public validation split.
The official conversion maps this to an estimated 100/100 validation score.
The classifier is fitted only on the 2,400 official training recordings.
No hidden-test or leaderboard score is claimed.
See `SOLUTION.md` for implementation and runtime details.

## Provenance

The original Polish task notebook is retained as `notebooks/original_submission.ipynb`.
`docs/task_en.md` is an English task summary.
