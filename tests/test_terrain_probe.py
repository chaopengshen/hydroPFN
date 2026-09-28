"""Check held-out label separation and gauge alignment for the terrain probe."""
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from basin_terrain_probe import crossfit_baseline, load_terrain, ridge_predict, signatures


class TerrainProbeTests(unittest.TestCase):
    def test_outer_labels_cannot_affect_training_residuals(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(40, 3))
        y = rng.normal(size=(40, 2))
        groups = np.repeat(np.arange(4), 10)
        tr = np.flatnonzero(groups != 0)
        residual = crossfit_baseline(x, y, groups, tr, 100.)
        modified = y.copy()
        modified[groups == 0] = 999999
        np.testing.assert_equal(residual, crossfit_baseline(x, modified, groups, tr, 100.))
        # Within an inner validation region, its own labels can only change
        # its residual by that same change, never the baseline prediction.
        modified = y.copy()
        modified[groups == 1] += 100
        second = crossfit_baseline(x, modified, groups, tr, 100.)
        np.testing.assert_allclose(second[groups[tr] == 1] - residual[groups[tr] == 1], 100.)

    def test_query_batch_cannot_change_normalization(self):
        rng = np.random.default_rng(9)
        x, y = rng.normal(size=(30, 5)), rng.normal(size=(30, 2))
        test = rng.normal(size=(1, 5))
        first = ridge_predict(x, y, test, 100.)
        expanded = np.vstack([test, np.full((1, 5), 1e9)])
        np.testing.assert_allclose(first, ridge_predict(x, y, expanded, 100.)[:1])

    def test_gauges_are_joined_and_failed_coverage_remains_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "terrain.npz"
            np.savez(path, site_id=np.array([20, 10, 30]), feats=np.array([[2.], [1.], [3.]]),
                     ok=np.array([True, True, False]), feature_names=np.array(["elevation"]))
            x, names = load_terrain(path, ["00000010", "00000020", "00000030", "00000040"])
            np.testing.assert_equal(x[:2, 0], [1., 2.])
            self.assertTrue(np.isnan(x[2:]).all())
            self.assertEqual(names, ["elevation"])

    def test_signature_units_have_expected_log_scaling(self):
        q = (2 + np.sin(np.arange(800) / 20))[None]
        y, doubled = signatures(q), signatures(q * 2)
        np.testing.assert_allclose(doubled[0, :3] - y[0, :3], np.log(2))
        np.testing.assert_allclose(doubled[0, 3:], y[0, 3:])


if __name__ == "__main__":
    unittest.main()
