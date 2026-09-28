"""Physical-unit terrain checks; run in the GIS demenv on suntzu."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from basin_dem_features import terrain_descriptors


class TerrainGeometryTests(unittest.TestCase):
    def test_planar_slope_uses_physical_spacing(self):
        y, x = np.mgrid[:40, :50]
        z = 100 + .1 * (30 * x) + .2 * (60 * y)
        mask = np.ones_like(z, bool)
        features = terrain_descriptors(z, mask, 30., 60.)
        self.assertAlmostEqual(features[9], np.hypot(.1, .2), places=10)
        self.assertAlmostEqual(features[10], 0., places=10)
        doubled = terrain_descriptors(2 * z, mask, 30., 60.)
        self.assertAlmostEqual(doubled[9], 2 * features[9], places=10)
        self.assertAlmostEqual(doubled[2], 2 * features[2], places=10)

    def test_basin_statistics_exclude_outside_cells(self):
        y, x = np.mgrid[:40, :50]
        z = (x + y).astype(float)
        mask = x < 25
        before = terrain_descriptors(z, mask, 30., 30.)
        z[:, 25:] += 10000
        after = terrain_descriptors(z, mask, 30., 30.)
        np.testing.assert_equal(before[:9], after[:9])
        np.testing.assert_equal(before[13:], after[13:])


if __name__ == "__main__":
    unittest.main()
