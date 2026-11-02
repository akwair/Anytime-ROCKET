"""Protocol regressions; run with python -m unittest discover -s tests -v."""
import importlib.util
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import RidgeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import anytime as at


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BaselineTests(unittest.TestCase):
    def modules(self):
        self.assertTrue((ROOT / 'src/baselines.py').exists(), 'baseline runner not implemented')
        import baselines as b
        import baseline_pruning as bp
        return b, bp

    def test_cost_counts_shared_units_and_dilations(self):
        b, _ = self.modules()
        params = (np.array([1, 2]), np.array([2, 3]), np.zeros(420))
        U = at.units_of(params, 32)
        self.assertEqual(b.subset_conv_cost(U, np.arange(420), 32), at.conv_cost(U, np.arange(168), 32))
        self.assertEqual(b.subset_conv_cost(U, [0, 1, 2, 168], 32),
                         2*32 + 16*32*2 + 3*32*3 + 2*32 + 24 + 16)

    def test_tiny_calibration_is_uncertified_full_fallback(self):
        b, _ = self.modules()
        Pc = np.zeros((3, 3), dtype=int)
        for method in ('ltt', 'crc'):
            result = b.choose_candidate(Pc, .01, method)
            self.assertEqual(result['index'], 0)
            self.assertFalse(result['certified'])
            self.assertEqual(result['status'], 'known_safe_full_fallback')

    def test_split_exactly_matches_v2(self):
        b, _ = self.modules()
        idx = np.random.default_rng(4).permutation(113)
        parts = b.split_indices(113, 40, 4)
        for actual, expected in zip(parts, (idx[:40], idx[40:76], idx[76:])):
            np.testing.assert_array_equal(actual, expected)

    def test_pocket_matches_official_core_both_matrix_shapes(self):
        _, bp = self.modules()
        official = load_file('official_pocket', ROOT / 'third_party/pocket/MiniROCKET/PROCKET_pruner.py')
        for shape in ((18, 31), (31, 18)):
            rng = np.random.default_rng(60)
            X = rng.normal(size=shape)
            X, _, _ = bp.center_l2(X)
            y = np.arange(shape[0]) % 3
            reference = official.PROCKETPruner(3, y, X, y, k=1., remain_num=6, epoch=50, if_print=False)
            reference.fit(X, y)
            actual = bp.pocket_core(X, y, remain=6, k=1., epochs=50)
            np.testing.assert_allclose(actual['W'], reference.W, rtol=1e-8, atol=1e-9)
            np.testing.assert_allclose(actual['theta'], reference.Theta, rtol=1e-8, atol=1e-9)
            np.testing.assert_array_equal(actual['mask'], np.setdiff1d(np.arange(shape[1]), reference.Theta_zero_index))

    def test_pocket_zero_columns_and_noncontiguous_labels(self):
        _, bp = self.modules()
        X = np.zeros((12, 17))
        for labels in ([2, 8]*6, [2, 8, 19]*4):
            r = bp.pocket_core(X, np.array(labels), remain=3, k=1., epochs=5)
            self.assertTrue(np.isfinite(r['W']).all())
            self.assertEqual(len(r['mask']), 0)
            np.testing.assert_array_equal(r['classes'], np.unique(labels))

    def test_sfd_matches_official_without_resurrecting_features(self):
        _, bp = self.modules()
        package = types.ModuleType('detach_rocket')
        package.__path__ = [str(ROOT / 'third_party/detach_rocket/detach_rocket')]
        with patch.dict(sys.modules, {'detach_rocket': package}):
            official = load_file('official_sfd', ROOT / 'third_party/detach_rocket/detach_rocket/sfd.py')
        rng = np.random.default_rng(52)
        X, y = rng.normal(size=(48, 24)), np.arange(48) % 3
        counts = np.unique((24 * .95 ** np.arange(12)).astype(int))[::-1]
        clf = RidgeClassifier(alpha=2.).fit(X, y)
        _, _, _, importance = official.feature_detachment(clf, X, y, num_steps=12)
        actual = bp.detach_masks(X, y, counts, alpha=2.)
        for count, weights in zip(counts, importance):
            np.testing.assert_array_equal(actual[int(count)], np.flatnonzero(weights))
        zero = bp.detach_masks(np.zeros((24, 20)), np.arange(24) % 2, [16, 8, 2], alpha=1.)
        self.assertTrue(set(zero[2]) <= set(zero[8]) <= set(zero[16]))

    def test_predictions_use_fitted_class_labels(self):
        b, _ = self.modules()
        rng = np.random.default_rng(2)
        X = rng.normal(size=(20, 6))
        for labels in ([3, 19]*10, [3, 9, 19, 3]*5, [19]*20):
            model = b.fit_head(X, np.array(labels))
            pred = b.head_predict(model, X)
            self.assertTrue(set(pred) <= set(labels))

    def test_training_never_receives_calibration_or_test_rows(self):
        b, bp = self.modules()
        rng = np.random.default_rng(1)
        X = rng.normal(size=(48, 16))
        y = np.resize([3, 19, 47], 48)
        fit, cal, test = b.split_indices(48, 24, 0)
        y[cal] = 999
        y[test] = 888
        original_fit, original_head = at.fit_params, b.fit_head
        original_pocket, original_detach = b.pocket_masks, b.detach_masks
        seen = []

        def checked_transform(rows, num_features=10000, seed=0):
            np.testing.assert_array_equal(rows, X[fit])
            seen.append('transform')
            return original_fit(rows, num_features, seed)

        def checked_head(rows, labels):
            self.assertEqual(len(rows), len(fit))
            np.testing.assert_array_equal(labels, y[fit])
            seen.append('head')
            return original_head(rows, labels)

        def checked_selector(original):
            def call(rows, labels, *args, **kwargs):
                self.assertEqual(len(rows), len(fit))
                np.testing.assert_array_equal(labels, y[fit])
                seen.append('selector')
                return original(rows, labels, *args, **kwargs)
            return call

        with tempfile.TemporaryDirectory(dir=ROOT) as out, patch.object(b, 'OUT', out), \
                patch.object(b, 'REFERENCE_DIR', None), patch.object(b, 'NUM_FEATURES', 252), \
                patch.object(b, 'load_ucr', return_value=(X[:24], y[:24], X[24:], y[24:])), \
                patch.object(at, 'fit_params', side_effect=checked_transform), \
                patch.object(b, 'pocket_masks', side_effect=checked_selector(original_pocket)), \
                patch.object(b, 'detach_masks', side_effect=checked_selector(original_detach)), \
                patch.object(b, 'fit_head', side_effect=checked_head):
            result = b.run('Synthetic', 0)
            self.assertIn('pocket_minirocket', result['families'])
            self.assertTrue(seen)
            with np.load(Path(out) / 'Synthetic_s0.npz') as z:
                for name, family in result['families'].items():
                    Pc = z[name + '_Pcal']
                    np.testing.assert_array_equal(Pc[:, 0], z['full_Pcal'])
                    if name != 'strat_imp_adaptive':
                        self.assertTrue(np.all(np.diff(z[name + '_ideal_cost']) <= 0))
                    self.assertEqual(family['ltt']['0.01']['index'], 0)
            self.assertEqual(result, b.run('Synthetic', 0))
            result_path = Path(out) / 'Synthetic_s0.json'
            npz_path = result_path.with_suffix('.npz')
            self.assertIsNone(b.reusable(result_path, npz_path, 'stale_protocol_hash', 'Synthetic', 0))
            with npz_path.open('ab') as f:
                f.write(b'corrupt')
            self.assertIsNone(b.reusable(result_path, npz_path, result['protocol_hash'], 'Synthetic', 0))


class BaselineSummaryTests(unittest.TestCase):
    def write_fixture(self, directory, stem='Example_s0', dataset='Example', packages=None,
                      reference_status='exact_v2_arrays_verified', internal_hash='fixture-protocol',
                      broken_anchor=False):
        path = Path(directory) / (stem + '.json')
        full_cal, full_test = np.array([3, 19]), np.array([19, 3, 19])
        candidate_cal = full_cal[:, None].copy()
        if broken_anchor:
            candidate_cal[0, 0] = 19
        family = 'strat_imp_adaptive'
        np.savez_compressed(path.with_suffix('.npz'), protocol_hash=np.array(internal_hash),
                            classes=np.array([3, 19]), y_test=full_test, full_Pcal=full_cal,
                            full_Ptest=full_test, **{family+'_Pcal': candidate_cal,
                            family+'_Ptest': full_test[:, None], family+'_ideal_cost': np.array([1.]),
                            family+'_full_transform_cost': np.array([1.])})
        metric = dict(cost=1., full_transform_cost=1., acc=1., disagree=0., certified=False,
                      test_disagreements=0, index=0)
        result = dict(schema='split-baselines-v1', dataset=dataset, seed=0,
                      protocol_hash='fixture-protocol', n_cal=2, n_test=3, full_acc=1.,
                      classes=[3, 19], settings={'eps': [.01]},
                      packages=packages or {'numpy': '1.26.4', 'scipy': '1.15.3'},
                      reference_check={'status': reference_status},
                      npz_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest(),
                      families={family: {'candidates': [{}], 'ltt': {'0.01': metric},
                                         'crc': {'0.01': metric}}})
        path.write_text(json.dumps(result), encoding='utf-8')
        return path

    def test_summary_accepts_verified_fixture(self):
        from summarize_baselines import summarize
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            self.write_fixture(directory)
            result = summarize(directory, seeds=(0,))
            self.assertEqual(result['n_runs'], 1)
            self.assertEqual(result['datasets_with_all_seeds'], 1)

    def test_summary_rejects_corrupt_npz(self):
        from summarize_baselines import summarize
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_fixture(directory)
            with path.with_suffix('.npz').open('ab') as f:
                f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'NPZ hash'):
                summarize(directory, seeds=(0,))

    def test_summary_rejects_mixed_packages(self):
        from summarize_baselines import summarize
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            self.write_fixture(directory)
            self.write_fixture(directory, stem='Other_s0', dataset='Other', packages={'numpy': '2.0'})
            with self.assertRaisesRegex(ValueError, 'package versions'):
                summarize(directory, seeds=(0,))

    def test_summary_rejects_unverified_reference(self):
        from summarize_baselines import summarize
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            self.write_fixture(directory, reference_status='regenerated_external_npz_absent')
            with self.assertRaisesRegex(ValueError, 'exact v2'):
                summarize(directory, seeds=(0,))

    def test_summary_rejects_duplicate_dataset_seed(self):
        from summarize_baselines import summarize
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            self.write_fixture(directory)
            self.write_fixture(directory, stem='duplicate')
            with self.assertRaisesRegex(ValueError, 'Duplicate dataset/seed'):
                summarize(directory, seeds=(0,))

    def test_summary_rejects_internal_protocol_mismatch(self):
        from summarize_baselines import summarize
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            self.write_fixture(directory, internal_hash='different-protocol')
            with self.assertRaisesRegex(ValueError, 'internal protocol'):
                summarize(directory, seeds=(0,))

    def test_summary_rejects_different_full_anchor(self):
        from summarize_baselines import summarize
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            self.write_fixture(directory, broken_anchor=True)
            with self.assertRaisesRegex(ValueError, 'full anchor'):
                summarize(directory, seeds=(0,))


if __name__ == '__main__':
    unittest.main()
