# Reconstructed Audio Classification - Polish AI Olympiad III

This repository is a reconstructed reference solution for the first-stage Audio Classification task in the Polish Artificial Intelligence Olympiad.
It is not the author's original competition submission.

The pipeline converts waveforms into mel-spectral summary features and classifies them with a standardized radial-basis support-vector model.
It is deterministic, CPU-friendly, and has no hidden notebook state.

![Audio-classification task illustration](assets/task-audio.png)

*Task illustration embedded in the official Polish AI Olympiad III notebook.*

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
