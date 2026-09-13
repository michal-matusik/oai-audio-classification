# Solution notes

This repository contains a retrospective reconstruction and not the original competition submission.

Each waveform is converted to a 64-band mel spectrogram in decibels.
The feature vector combines per-band means and standard deviations, means over four temporal sections, and four waveform statistics.
A standardized radial-basis support-vector classifier separates normal speech, screaming, and whispering.

The official training split contains 2,400 balanced recordings.
The official validation split contains 200 recordings.
Fitting uses only the training split and takes under five seconds on this Apple Silicon CPU, including feature extraction and validation.

The measured official public-validation balanced accuracy is 1.0000.
The notebook's score conversion maps that result to 100/100.
No hidden-test or leaderboard score is claimed.
