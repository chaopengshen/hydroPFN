"""Scientific invariants for the objective ablation; runnable without pytest."""
import unittest

import numpy as np
import torch

from hydropfn.train.objectives import basin_normalized_mse, training_variances


class ObjectiveTests(unittest.TestCase):
    def test_no_test_period_or_heldout_basin_statistics(self):
        q = np.array([[1., 3., 100., 300.], [8., 8., 999., 888.],
                      [4., 6., 20., 40.]])
        tr = np.array([0, 1])
        v = training_variances(q, tr, slice(0, 2), 2., 0.1)
        np.testing.assert_allclose(v[:2], [0.25, 0.0025])
        self.assertTrue(np.isnan(v[2]))
        q[:, 2:] *= 1000
        q[2] *= 1000
        np.testing.assert_equal(v, training_variances(q, tr, slice(0, 2), 2., 0.1))

    def test_affine_target_units_do_not_change_nse_surrogate(self):
        y = torch.tensor([[1., 2., 3.], [2., 4., 6.]])
        p = y + torch.tensor([[0.2], [0.4]])
        v = y.var(dim=1, correction=0)
        mask = torch.ones_like(y)
        loss = basin_normalized_mse(p, y, mask, v)
        for scale, offset in [(7., -4.), (0.03, 10.)]:
            other = basin_normalized_mse(p * scale + offset, y * scale + offset,
                                        mask, v * scale ** 2)
            torch.testing.assert_close(loss, other, atol=1e-6, rtol=1e-4)

    def test_missing_days_do_not_reweight_basins_or_leak_gradients(self):
        y = torch.tensor([[1., 2., float("nan"), 4.], [1., 2., 3., 4.]])
        p = torch.tensor([[2., 3., 10000., 999.], [3., 4., 5., 6.]],
                         requires_grad=True)
        valid = torch.tensor([[1., 1., 1., 0.], [1., 1., 1., 1.]])
        loss = basin_normalized_mse(p, y, valid, torch.ones(2))
        self.assertAlmostEqual(loss.item(), 2.5)
        loss.backward()
        torch.testing.assert_close(p.grad[0, 2:], torch.zeros(2))
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_empty_basin_excluded_and_unknown_variance_rejected(self):
        y = torch.zeros(2, 4)
        p = torch.ones_like(y)
        mask = torch.tensor([[0., 0., 0., 0.], [1., 1., 1., 1.]])
        self.assertEqual(basin_normalized_mse(p, y, mask, torch.ones(2)).item(), 1.)
        with self.assertRaises(ValueError):
            basin_normalized_mse(p, y, mask, torch.tensor([1., float("nan")]))


if __name__ == "__main__":
    unittest.main()
