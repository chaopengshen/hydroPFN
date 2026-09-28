"""Paired observation views, causal inputs, and averaged-gradient training."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from hydropfn.data.observations import training_di_views, training_di_features
import camels531_daily_di as driver


class PairedTrainingTests(unittest.TestCase):
    def test_daily_is_retained_and_masked_view_matches_original_sampler(self):
        q = np.arange(8 * 730, dtype=np.float32).reshape(8, 730)
        q[:, 50] = np.nan
        original = q.copy()
        views = training_di_views(q, "paired", np.random.default_rng(7))
        self.assertEqual([v[0] for v in views], ["daily", "mixed"])
        np.testing.assert_equal(views[0][1], training_di_features(q, "daily", np.random.default_rng(7)))
        np.testing.assert_equal(views[1][1], training_di_features(q, "mixed", np.random.default_rng(7)))
        np.testing.assert_equal(q, original)

    def test_both_views_preserve_causality(self):
        q = np.ones((12, 730), np.float32)
        changed = q.copy()
        changed[:, 400:] = 99999
        first = training_di_views(q, "paired", np.random.default_rng(9))
        second = training_di_views(changed, "paired", np.random.default_rng(9))
        for (_, a), (_, b) in zip(first, second):
            np.testing.assert_equal(a[:, :401], b[:, :401])

    def fit_one(self, mode, second="daily"):
        rng = np.random.default_rng(14)
        forc = rng.normal(size=(3, 750, 2)).astype(np.float32)
        attrs = rng.normal(size=(3, 2)).astype(np.float32)
        q = rng.normal(size=(3, 750)).astype(np.float32)
        args = SimpleNamespace(hidden=8, dropout=0., lr=1., steps=1, epochs=1,
                               batch=2, mode=mode, paired_second=second, loss="nse")
        torch.manual_seed(5)
        previous_device = driver.DEVICE
        driver.DEVICE = "cpu"
        try:
            return driver.fit(forc, attrs, q, np.arange(3), {"train": slice(0, 750)},
                              np.ones(3, np.float32), args,
                              np.random.default_rng(4), np.random.default_rng(0))
        finally:
            driver.DEVICE = previous_device

    def test_duplicate_daily_mean_equals_one_daily_update_without_dropout(self):
        single, double = self.fit_one("daily"), self.fit_one("paired")
        for key, value in single.state_dict().items():
            torch.testing.assert_close(value, double.state_dict()[key], atol=1e-7, rtol=1e-6)
        self.assertEqual(single.training_budget["optimizer_updates"], 1)
        self.assertEqual(double.training_budget["optimizer_updates"], 1)
        self.assertEqual(double.training_budget["sampled_window_views"], 4)

    def test_masked_view_contributes_to_update(self):
        daily, mixed = self.fit_one("paired"), self.fit_one("paired", "mixed")
        difference = sum((daily.state_dict()[k] - v).abs().sum().item()
                         for k, v in mixed.state_dict().items())
        self.assertGreater(difference, 1e-5)


if __name__ == "__main__":
    unittest.main()
