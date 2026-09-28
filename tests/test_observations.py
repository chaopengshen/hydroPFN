"""Observation availability, target leakage and recurrent causality checks."""
import unittest
import sys
from pathlib import Path
import numpy as np
import torch
from hydropfn.data.observations import (lagged_observations,
                                       training_di_features, evaluation_di_features)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from camels531_lstm import RegionalLSTM


class ObservationTests(unittest.TestCase):
    def test_every_input_precedes_its_prediction(self):
        q = np.arange(20, dtype=np.float32)[None]
        for delay in (1, 2, 7):
            f = lagged_observations(q, delay=delay)
            np.testing.assert_equal(f[0, delay:, 0], q[0, :-delay])
            np.testing.assert_equal(f[0, :delay, 3], 0.)
            altered = q.copy()
            altered[:, 10:] += 100000
            np.testing.assert_equal(f[:, :11], lagged_observations(altered, delay=delay)[:, :11])

    def test_missing_is_not_zero_and_old_values_expire(self):
        q = np.array([[0., np.nan, np.nan, 4., 5.]])
        f = lagged_observations(q, max_age=2)
        self.assertEqual(f[0, 1, 0], 0.)
        self.assertEqual(f[0, 1, 3], 1.)
        self.assertEqual(f[0, 3, 3], 0.)
        self.assertEqual(f[0, 4, 0], 4.)

    def test_campaign_stops_before_scoring_and_labels_cannot_change_it(self):
        q = np.arange(730, dtype=np.float32)[None]
        f = evaluation_di_features(q, "campaign30")
        changed = q.copy()
        changed[:, 365:] *= 1000
        np.testing.assert_equal(f, evaluation_di_features(changed, "campaign30"))
        self.assertEqual(f[0, 365, 0], 364.)
        self.assertEqual(f[0, 365, 1], 1.)
        self.assertEqual(f[0, 366, 1], 0.)

    def test_mixture_is_reproducible_and_does_not_use_target_values_for_masks(self):
        q = np.arange(32 * 730, dtype=np.float32).reshape(32, 730)
        first = training_di_features(q, "mixed", np.random.default_rng(9))
        second = training_di_features(q * 2, "mixed", np.random.default_rng(9))
        np.testing.assert_equal(first[..., 1:], second[..., 1:])

    def test_recurrent_output_cannot_read_today_or_future_discharge(self):
        torch.manual_seed(3)
        net = RegionalLSTM(6, 3, hidden=8, dropout=0.).eval()
        q = np.arange(30, dtype=np.float32)[None]
        changed = q.copy()
        changed[:, 15:] += 1000
        forcing = np.ones((1, 30, 2), np.float32)
        attrs = torch.zeros(1, 3)
        with torch.no_grad():
            a = net(torch.tensor(np.concatenate([forcing, lagged_observations(q)], -1)), attrs)
            b = net(torch.tensor(np.concatenate([forcing, lagged_observations(changed)], -1)), attrs)
        torch.testing.assert_close(a[:, :16], b[:, :16], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
