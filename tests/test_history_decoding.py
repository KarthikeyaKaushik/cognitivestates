import contextlib
import io
import unittest
from itertools import product
from unittest.mock import patch

import numpy as np

from src import history_decoding
from src.decoding import fit_projection, pairwise_test_accuracy
from src.history_decoding import history_decoding_scores, history_samples


class HistoryDecodingTests(unittest.TestCase):
    def setUp(self):
        self.inputs = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
        self.hidden = np.arange(24).reshape(2, 4, 3).astype(float)

    def test_windows_end_at_current_input_without_crossing_sequences(self):
        predictor, labels = history_samples(self.hidden, self.inputs, 2)
        np.testing.assert_array_equal(predictor, self.hidden[:, 1:, :].reshape(-1, 3))
        self.assertEqual(labels.tolist(), [0, 1, 0, 1, 0, 1])

    def test_k_one_labels_are_current_inputs(self):
        predictor, labels = history_samples(self.hidden, self.inputs, 1)
        np.testing.assert_array_equal(predictor, self.hidden.reshape(-1, 3))
        np.testing.assert_array_equal(labels, self.inputs.reshape(-1))

    def test_k_three_discards_incomplete_prefixes(self):
        predictor, labels = history_samples(self.hidden, self.inputs, 3)
        np.testing.assert_array_equal(predictor, self.hidden[:, 2:, :].reshape(-1, 3))
        self.assertEqual(labels.tolist(), [0, 1, 1, 0])

    def test_alignment_and_window_validation(self):
        for k in (0, 5):
            with self.assertRaisesRegex(ValueError, 'retained timesteps'):
                history_samples(self.hidden, self.inputs, k)
        with self.assertRaisesRegex(ValueError, 'aligned'):
            history_samples(self.hidden[:, :-1], self.inputs, 1)

    def samples(self):
        inputs = np.array(list(product(range(2), repeat=3)) * 16)
        rng = np.random.default_rng(7)
        states = np.cumsum(inputs, axis=1) % 3
        hidden = rng.normal(size=(*inputs.shape, 5))
        hidden[:, :, 0] += 3 * inputs
        hidden[:, :, 1] += states
        return hidden, inputs, states

    def test_lda_uses_histories_but_classifiers_use_original_state_labels(self):
        train, inputs, states = self.samples()
        test = train + np.random.default_rng(8).normal(size=train.shape) * 0.1
        with patch.object(history_decoding, 'fit_projection',
                          wraps=fit_projection) as fit, \
                patch.object(history_decoding, 'pairwise_test_accuracy',
                             wraps=pairwise_test_accuracy) as score:
            rows = history_decoding_scores(train, inputs, states, test, inputs, states, 3)
        self.assertEqual([r['n_history_classes'] for r in rows], [2, 4, 8])
        self.assertEqual([r['n_components'] for r in rows], [1, 2, 2])
        self.assertEqual([r['n_state_classes'] for r in rows], [3, 3, 3])
        for k, (fit_call, score_call) in enumerate(
                zip(fit.call_args_list, score.call_args_list), start=1):
            self.assertEqual(len(np.unique(fit_call.args[2])), 2 ** k)
            for label_array in (score_call.args[1], score_call.args[3]):
                np.testing.assert_array_equal(np.unique(label_array), [0, 1, 2])
            _, train_history = history_samples(train, inputs, k)
            expected_counts = np.unique(train_history, return_counts=True)[1]
            self.assertEqual(len(fit_call.args[1]),
                             len(expected_counts) * expected_counts.min())

    def test_test_data_and_labels_do_not_change_fitted_history_lda(self):
        train, inputs, states = self.samples()
        recorded = []

        def capture(method, X, y):
            projection = fit_projection(method, X, y)
            recorded.append((X.copy(), y.copy(), projection.scalings_.copy()))
            return projection

        with patch.object(history_decoding, 'fit_projection', side_effect=capture):
            history_decoding_scores(train, inputs, states, train, inputs, states, 4)
            history_decoding_scores(train, inputs, states, train * 10 + 50,
                                    1 - inputs, np.roll(states, 1, axis=0), 4)
        for original, changed in zip(recorded[:3], recorded[3:]):
            for before, after in zip(original, changed):
                np.testing.assert_array_equal(before, after)

    def test_missing_prefix_states_are_printed_and_not_history_targets(self):
        train, inputs, states = self.samples()
        states = states.copy()
        states[:, 0] = 9
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rows = history_decoding_scores(train, inputs, states, train, inputs, states, 0)
        self.assertIn('excluding IDs [9]', output.getvalue())
        self.assertEqual([r['n_state_classes'] for r in rows], [4, 3, 3])

    def test_sparse_training_histories_are_explicitly_unavailable(self):
        with self.assertWarnsRegex(RuntimeWarning, 'state decoding unavailable'):
            rows = history_decoding_scores(
                self.hidden[:1], self.inputs[:1], self.inputs[:1],
                self.hidden[1:], self.inputs[1:], self.inputs[1:], 0)
        self.assertEqual(rows[1]['status'], 'insufficient_samples')
        self.assertTrue(np.isnan(rows[1]['accuracy']))
        self.assertEqual(rows[2]['status'], 'insufficient_samples')
        self.assertIn('training samples', rows[2]['reason'])

    def test_sparse_test_histories_do_not_prevent_training_lda(self):
        train, inputs, states = self.samples()
        # Test histories may have one observation: only TRAIN histories fit LDA.
        rows = history_decoding_scores(
            train, inputs, states, train[:8], inputs[:8], states[:8], 0)
        self.assertEqual(rows[2]['status'], 'ok')
        self.assertGreater(rows[2]['n_state_classes'], 1)

    def test_max_k_two_does_not_construct_or_fit_three_input_histories(self):
        train, inputs, states = self.samples()
        with patch.object(history_decoding, 'history_samples',
                          wraps=history_samples) as samples:
            rows = history_decoding_scores(
                train, inputs, states, train, inputs, states, 0, max_k=2)
        self.assertEqual([row['k'] for row in rows], [1, 2])
        self.assertEqual([call.args[2] for call in samples.call_args_list], [1, 1, 2, 2])

    def test_invalid_max_k_is_rejected(self):
        train, inputs, states = self.samples()
        for max_k in (0, 4):
            with self.assertRaisesRegex(ValueError, 'max_k'):
                history_decoding_scores(
                    train, inputs, states, train, inputs, states, 0, max_k=max_k)


if __name__ == '__main__':
    unittest.main()
