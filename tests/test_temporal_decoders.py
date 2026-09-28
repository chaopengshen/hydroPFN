"""Causality, history, context, legacy compatibility and an end-to-end smoke."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch as mock_patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from hydropfn.models.temporal_decoders import (DailySequenceDecoder,
                                             PatchReconstructionDecoder, lift_patch_features)
from hydropfn.models.site_encoder import SiteEncoder
from hydropfn.models.connector import PUBModel
from hydropfn.data.decoder_windows import evaluation_windows, training_starts
from camels531_lstm import RegionalLSTM
import camels531_decoders as driver


class DecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_future_perturbation_and_gradients(self):
        for kind in ("lstm", "tcn", "pointwise"):
            net = DailySequenceDecoder(2, 3, kind, latent_dim=4, hidden=8,
                                      tcn_width=8, tcn_blocks=2, dropout=0.).eval()
            x, a, z = torch.randn(2, 48, 2), torch.randn(2, 3), torch.randn(2, 48, 4)
            x.requires_grad_(); z.requires_grad_()
            p = net(x, a, z)
            changed_x, changed_z = x.detach().clone(), z.detach().clone()
            changed_x[:, 25:] += 100
            changed_z[:, 25:] -= 100
            torch.testing.assert_close(p[:, :25], net(changed_x, a, changed_z)[:, :25])
            p[:, 24].sum().backward()
            self.assertEqual(float(x.grad[:, 25:].abs().sum()), 0.)
            self.assertEqual(float(z.grad[:, 25:].abs().sum()), 0.)
            self.assertGreater(float(z.grad[:, :25].abs().sum()), 0.)
            if kind != "pointwise":
                self.assertGreater(float(x.grad[:, 20:24].abs().sum()), 0.)

    def test_raw_lstm_matches_existing_reference(self):
        ref = RegionalLSTM(2, 3, hidden=8, dropout=0.).eval()
        net = DailySequenceDecoder(2, 3, "lstm", hidden=8, dropout=0.).eval()
        net.core.load_state_dict(ref.lstm.state_dict())
        net.readout[1].load_state_dict(ref.head.state_dict())
        x, a = torch.randn(2, 40, 2), torch.randn(2, 3)
        torch.testing.assert_close(ref(x, a), net(x, a), rtol=0, atol=0)

    def test_tcn_receptive_field_and_overlap_equivalence(self):
        net = DailySequenceDecoder(2, 3, "tcn", tcn_width=8, tcn_blocks=2, dropout=0.).eval()
        self.assertEqual(net.receptive_field, 13)
        x, a = torch.randn(2, 100, 2), torch.randn(2, 3)
        full = net(x, a)
        output = torch.zeros_like(full[:, 20:])
        counts = np.zeros(80)
        for start, lo, hi in evaluation_windows(20, 100, 40, 12):
            got = net(x[:, start:start+40], a)
            output[:, lo-20:hi-20] = got[:, lo-start:hi-start]
            counts[lo-20:hi-20] += 1
        np.testing.assert_equal(counts, np.ones(80))
        torch.testing.assert_close(full[:, 20:], output, atol=1e-6, rtol=1e-5)
        altered = x.clone(); altered[:, :60] += 99
        torch.testing.assert_close(full[:, 72:], net(altered, a)[:, 72:])

    def test_window_bounds_and_exactly_once_scoring(self):
        for stop in (1057, 1112, 1400):
            counts = np.zeros(stop-700)
            for start, lo, hi in evaluation_windows(700, stop, 512, 256):
                self.assertGreaterEqual(lo-start, 256)
                self.assertLessEqual(start+512, stop)
                counts[lo-700:hi-700] += 1
            np.testing.assert_equal(counts, np.ones_like(counts))
        starts = training_starts(7, 1100, 512, 64)
        self.assertEqual(starts[0], 7)
        self.assertEqual(starts[-1]+512, 1100)
        self.assertEqual(len(starts), len(set(starts)))
        with self.assertRaises(ValueError):
            evaluation_windows(100, 600, 512, 256)

    def test_patch_lifting_and_encoder_query_mask(self):
        z = torch.arange(6.).reshape(1, 3, 2)
        daily = lift_patch_features(z, 4)
        torch.testing.assert_close(daily[:, 4:8], z[:, 1:2].expand(-1, 4, -1))
        forcing = np.ones((2, 20, 2), np.float32)
        forcing[0, 2, 0] = np.nan
        batch = driver.encoder_batch(forcing, np.ones((2, 3), np.float32), 0, 16, 4, "cpu")
        self.assertEqual(float(batch["series"][..., -1, :].abs().sum()), 0.)
        self.assertEqual(float(batch["vis"][..., -1].sum()), 0.)
        self.assertTrue(batch["valid"][..., -1].bool().all())
        self.assertEqual(float(batch["valid"][0, 0, 0, 0]), 0.)
        self.assertTrue(torch.isfinite(batch["series"]).all())

    def test_daily_sampler_bounds_and_paired_rng(self):
        m = dict(train_start=7, train_stop=63, length=32)
        starts = training_starts(7, 63, 32, 8)
        fixed_rng, daily_rng = np.random.default_rng(9), np.random.default_rng(9)
        observed = set()
        for _ in range(2000):
            np.testing.assert_array_equal(fixed_rng.choice(20, 5, replace=False),
                                          daily_rng.choice(20, 5, replace=False))
            fixed, wi, u = driver.sample_start(fixed_rng, starts, m, 'fixed')
            daily, unused, v = driver.sample_start(daily_rng, starts, m, 'daily')
            self.assertEqual(u, v)
            self.assertEqual(fixed, starts[wi])
            self.assertIsNone(unused)
            self.assertTrue(7 <= daily <= 31)
            observed.add(daily)
        self.assertEqual(observed, set(range(7,32)))
        self.assertEqual({s % 16 for s in observed}, set(range(16)))

    def test_online_crop_matches_cache_and_keeps_encoder_frozen(self):
        encoder = PUBModel(SiteEncoder(3, 3, patch=4, d=16, depth=1, heads=4,
                                      k_summary=2, d_ffd=32),
                           d=16, depth=1, heads=4, geo=True, time_aligned=True)
        encoder.eval().requires_grad_(False)
        rng = np.random.default_rng(4)
        data = dict(forcing=rng.normal(size=(3,64,2)).astype(np.float32),
                    attrs=rng.normal(size=(3,3)).astype(np.float32))
        data['forcing'][0,11,0] = np.nan
        m = dict(train_start=0, train_stop=64, length=32, patch=4)
        b = np.arange(3); start = 7
        full = driver.encoder_batch(data['forcing'],data['attrs'],start,32,4,'cpu')
        cropped = driver.encoder_batch(data['forcing'][:,start:start+32],
                                       data['attrs'],0,32,4,'cpu',time_offset=start)
        for key in full:
            torch.testing.assert_close(full[key], cropped[key], rtol=0, atol=0)
        before = driver.weights_sha256(encoder)
        state = torch.get_rng_state().clone()
        _, _, online, _ = driver.online_inputs(encoder, data, b, start, m, 2, 'cpu')
        expected = lift_patch_features(encoder(full,return_hidden=True)[:,:,-1],4)
        torch.testing.assert_close(online, expected, rtol=1e-5, atol=1e-5)
        self.assertFalse(online.requires_grad)
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(before, driver.weights_sha256(encoder))
        with self.assertRaises(ValueError):
            driver.online_inputs(encoder,data,b,33,m,2,'cpu')

    def test_hidden_extraction_preserves_original_head(self):
        net = PUBModel(SiteEncoder(3, 3, patch=4, d=16, depth=1, heads=4,
                                  k_summary=2, d_ffd=32), d=16, depth=1, heads=4,
                       geo=True, time_aligned=True).eval()
        batch = driver.encoder_batch(np.ones((2, 16, 2), np.float32),
                                     np.ones((2, 3), np.float32), 0, 16, 4, "cpu")
        old = net(batch)
        hidden = net(batch, return_hidden=True)
        torch.testing.assert_close(net.head(hidden), old, rtol=0, atol=0)
        patch_head = PatchReconstructionDecoder(16, patch=4).eval()
        patch_head.head.load_state_dict(net.head.state_dict())
        daily = lift_patch_features(hidden[:, :, -1], 4)
        got = patch_head(torch.zeros(2, 16, 2), torch.zeros(2, 3), daily)
        torch.testing.assert_close(got, old[:, :, -1].flatten(1), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            patch_head(torch.zeros(2, 15, 2), torch.zeros(2, 3), daily[:, :15])

    def test_training_and_scoring_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); cache = root / "cache"; cache.mkdir()
            rng = np.random.default_rng(2)
            n, t, length, patch = 3, 120, 32, 4
            starts = training_starts(0, 80, length, 8)
            ev = evaluation_windows(80, 120, length, 16)
            forcing = rng.normal(size=(n, t, 2)).astype(np.float32)
            attrs = rng.normal(size=(n, 3)).astype(np.float32)
            q = rng.normal(size=(n, t)).astype(np.float32)
            np.savez(cache / "data.npz", forcing=forcing, attrs=attrs, q_raw=q,
                     q_mean=0., q_std=1., variance=np.ones(n, np.float32),
                     gage=np.array(['a','b','c']), train_starts=starts, eval_windows=ev)
            for label, wins in [('train', starts), ('eval', ev)]:
                np.save(cache / f'{label}_features.npy',
                        rng.normal(size=(len(wins), n, length//patch, 4)).astype(np.float32))
            np.save(cache / 'targ.npy', q[:, 80:])
            m = dict(length=length, patch=patch, warmup=16, latent_dim=4, n_basins=n,
                     score_days=40, train_start=0, train_stop=80, score_start=80,
                     information='synthetic smoke',
                     files_sha256={p.name:driver.sha(p) for p in cache.iterdir()})
            (cache / 'manifest.json').write_text(json.dumps(m))
            prev_device, prev_logs = driver.DEVICE, driver.LOGS
            driver.DEVICE, driver.LOGS = 'cpu', root / 'logs'
            try:
                for kind in ('lstm','tcn','pointwise','patch'):
                    a = SimpleNamespace(cache=str(cache), tag=kind, seed=0, decoder=kind,
                            inputs='latent' if kind == 'patch' else 'hybrid',
                            hidden=8, tcn_width=8, tcn_blocks=2,
                            dropout=0.1, lr=1., epochs=2, steps=2, batch=2, eval_chunk=2,
                            checkpoint_epochs='1,2', evaluate_checkpoints=True)
                    with contextlib.redirect_stdout(io.StringIO()):
                        driver.train(a)
                    out = driver.LOGS / 'camels531' / kind
                    self.assertTrue(np.isfinite(np.load(out/'pred.npy')).all())
                    record = json.loads((out/'run.json').read_text())
                    self.assertEqual(record['optimizer_updates'],4)
                    scores = json.loads((out/'checkpoint_scores.json').read_text())['scores']
                    self.assertEqual([v['epoch'] for v in scores], [1,2])
                    self.assertEqual(scores[-1]['median_nse'], record['median_nse'])
                    np.testing.assert_array_equal(np.load(out/'checkpoint_basin_nse.npz')['gage'],
                                                  np.array(['a','b','c']))
                    self.assertTrue(np.isfinite(np.load(out/'pred_epoch1.npy')).all())
                    # Saving and later evaluating milestones must not perturb
                    # the final training trajectory (including dropout).
                    a.tag = kind + '_no_milestones'
                    a.checkpoint_epochs, a.evaluate_checkpoints = '', False
                    with contextlib.redirect_stdout(io.StringIO()):
                        driver.train(a)
                    other = driver.LOGS / 'camels531' / a.tag
                    np.testing.assert_array_equal(np.load(out/'pred.npy'), np.load(other/'pred.npy'))
                # Exercise the complete daily path with a small real frozen
                # encoder. Its 4-channel query features match this fixture.
                encoder = PUBModel(SiteEncoder(3,3,patch=4,d=4,depth=1,heads=1,
                                               k_summary=2,d_ffd=8),
                                   d=4,depth=1,heads=1,geo=True,time_aligned=True)
                encoder.eval().requires_grad_(False)
                initial = driver.weights_sha256(encoder)
                a.tag, a.decoder, a.inputs = 'daily_smoke','lstm','hybrid'
                a.sampling, a.source, a.encoder_chunk = 'daily','mock_source',2
                with mock_patch.object(driver,'load_frozen_encoder',return_value=encoder):
                    with contextlib.redirect_stdout(io.StringIO()):
                        driver.train(a)
                daily = driver.LOGS / 'camels531' / a.tag
                r = json.loads((daily/'run.json').read_text())
                self.assertTrue(r['encoder_unchanged'])
                self.assertEqual(initial,driver.weights_sha256(encoder))
                self.assertTrue(np.isfinite(np.load(daily/'pred.npy')).all())
                a.tag, a.sampling = 'paired_fixed_smoke','fixed'
                a.feature_mode = 'online'
                with mock_patch.object(driver,'load_frozen_encoder',return_value=encoder):
                    with contextlib.redirect_stdout(io.StringIO()):
                        driver.train(a)
                fixed = json.loads((driver.LOGS/'camels531'/a.tag/'run.json').read_text())
                self.assertTrue(fixed['encoder_unchanged'])
                self.assertEqual(r['config']['initial_weights_sha256'],
                                 fixed['config']['initial_weights_sha256'])
                self.assertEqual(r['paired_sample_stream_sha256'],fixed['paired_sample_stream_sha256'])
            finally:
                driver.DEVICE, driver.LOGS = prev_device, prev_logs


if __name__ == '__main__':
    unittest.main()
