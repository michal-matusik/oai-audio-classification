import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from dataset import log_spectrogram, standardize

class AudioSmokeTest(unittest.TestCase):
    def test_log_spectrogram(self):
        wave = np.sin(2 * np.pi * 440 * np.arange(2048) / 16000)
        features = standardize(log_spectrogram(wave))
        self.assertEqual(features.shape[1], 129); self.assertTrue(np.isfinite(features).all())

if __name__ == '__main__': unittest.main()
