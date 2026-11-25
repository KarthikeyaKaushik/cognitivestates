import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src import decoding
from src.decoding import (InsufficientSamplesError, fit_projection,
                          pairwise_test_accuracy, state_projection_analysis,
                          state_samples)


class HeldoutDecodingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(14)
        self.y_train = np.repeat([10, 20, 30], 30)
        self.y_test = np.repeat([10, 20, 30], 10)
        self.train = rng.normal(size=(90, 5))
        self.test = rng.normal(size=(30, 5)) + 4
        self.train[:, 0] += self.y_train / 10
        self.test[:, 0] += self.y_test / 10

    def test_all_methods_fit_training_only_and_transform_test(self):
        expected_train, expected_y, expected_test, _ = state_samples(
            self.train, self.y_train, self.test, self.y_test, seed=2)
        with patch.object(decoding, 'fit_projection', wraps=fit_projection) as fit, \
                patch.object(decoding, 'pairwise_test_accuracy',
                             wraps=pairwise_test_accuracy) as score:
            projections, accuracies, targets = state_projection_analysis(
                self.train, self.y_train, self.test, self.y_test, seed=2)
        self.assertEqual(set(projections), {'LDA', 'PCA', 'ICA', 'FA'})
        self.assertEqual(score.call_count, 4)
        for call in fit.call_args_list:
            method, X, labels = call.args
            np.testing.assert_array_equal(X, expected_train)
            np.testing.assert_array_equal(labels, expected_y)
            manual = fit_projection(method, expected_train, expected_y)
            np.testing.assert_allclose(
                projections[method], manual.transform(expected_test), atol=1e-10)
            self.assertTrue(0 <= accuracies[method] <= 1)
        self.assertEqual(len(targets), 30)

    def test_perfect_training_separation_is_not_reported_as_perfect_test(self):
        train = np.array([[-3.], [-2.], [2.], [3.]])
        labels = np.array([0, 0, 1, 1])
        self.assertEqual(pairwise_test_accuracy(train, labels, train, 1 - labels), 0.)

    def test_scaler_is_fitted_only_on_training_and_scores_match_manual_svm(self):
        scaler = StandardScaler()
        with patch.object(decoding, 'StandardScaler', return_value=scaler):
            actual = pairwise_test_accuracy(
                self.train, self.y_train, self.test, self.y_test)
        np.testing.assert_allclose(scaler.mean_, self.train.mean(axis=0))
        scores = []
        for first, second in ((10, 20), (10, 30), (20, 30)):
            mask_train = np.isin(self.y_train, (first, second))
            mask_test = np.isin(self.y_test, (first, second))
            clf = LinearSVC(C=1e4, tol=1e-3, dual=False, max_iter=5000)
            clf.fit(scaler.transform(self.train)[mask_train], self.y_train[mask_train])
            scores.append(clf.score(scaler.transform(self.test)[mask_test],
                                    self.y_test[mask_test]))
        self.assertAlmostEqual(actual, np.mean(scores))

    def test_shuffled_control_stays_held_out(self):
        with patch.object(decoding, 'fit_projection', wraps=fit_projection) as fit:
            projections, scores, target = state_projection_analysis(
                self.train, self.y_train, self.test, self.y_test,
                seed=1, shuffled=True)
        self.assertEqual(list(projections), ['LDA'])
        self.assertEqual(list(scores), ['LDA'])
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(len(fit.call_args.args[1]), len(self.train))
        self.assertEqual(len(target), len(self.test))

    def test_scatter_filter_is_applied_to_both_splits_before_training(self):
        with patch.object(decoding, 'fit_projection', wraps=fit_projection) as fit:
            projections, scores, target = state_projection_analysis(
                self.train, self.y_train, self.test, self.y_test,
                keep_states={10, 30}, score=False)
        self.assertIsNone(scores)
        self.assertEqual(set(target), {10, 30})
        self.assertEqual(set(fit.call_args_list[0].args[2]), {10, 30})
        self.assertEqual(len(projections['PCA']), 20)

    def test_missing_states_are_reported_and_both_splits_balanced(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            train, labels, test, test_labels = state_samples(
                self.train, self.y_train, self.test[:20], self.y_test[:20],
                expected_states=[10, 20, 30, 40])
        self.assertIn('[30, 40]', output.getvalue())
        np.testing.assert_array_equal(np.unique(labels, return_counts=True)[1], [30, 30, 30])
        np.testing.assert_array_equal(np.unique(test_labels, return_counts=True)[1], [10, 10])
        self.assertEqual((len(train), len(test)), (90, 20))

    def test_missing_test_state_does_not_change_fitted_training_projections(self):
        with patch.object(decoding, 'fit_projection', wraps=fit_projection) as fit, \
                contextlib.redirect_stdout(io.StringIO()):
            state_projection_analysis(
                self.train, self.y_train, self.test, self.y_test, seed=7)
            state_projection_analysis(
                self.train, self.y_train, self.test[:20], self.y_test[:20], seed=7)
        for before, after in zip(fit.call_args_list[:4], fit.call_args_list[4:]):
            np.testing.assert_array_equal(before.args[1], after.args[1])
            np.testing.assert_array_equal(before.args[2], after.args[2])

    def test_test_states_not_in_training_raise(self):
        labels = self.y_test.copy()
        labels[labels == 30] = 99
        with self.assertRaises(InsufficientSamplesError):
            pairwise_test_accuracy(self.train, self.y_train, self.test, labels)


if __name__ == '__main__':
    unittest.main()
