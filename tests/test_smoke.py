import unittest

import numpy as np

from src.model import MelFeatureExtractor


class AudioSmokeTest(unittest.TestCase):
    def test_feature_shape_and_finiteness(self):
        waveform = np.sin(2 * np.pi * 440 * np.arange(4096) / 16000).astype(np.float32)
        features = MelFeatureExtractor().transform([waveform])
        self.assertEqual(features.shape, (1, 388))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
