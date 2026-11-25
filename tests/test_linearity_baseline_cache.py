import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib

matplotlib.use('Agg')

import pandas as pd

from scripts import fig_supp_linearitybaseline as analysis


class LinearityCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        directory = patch.object(analysis, 'RESULTS_DIR', self.root)
        directory.start()
        self.addCleanup(directory.stop)
        self.diagram = SimpleNamespace(
            states={0, 1}, X={'a'}, state_labels={0: 'x', 1: 'y'},
            statetrans={(0, 'a'): 1, (1, 'a'): 0},
            outputtrans={(0, 'a'): 'n', (1, 'a'): 'x'})
        self.pairs = [dict(labels=('x', 'y'), classification='immediate')]
        self.output = self.root / 'linearity_baseline_limitedstack.csv'
        self.trainer = patch.object(analysis, 'train_and_score', side_effect=self.train)
        self.mock_train = self.trainer.start()
        self.addCleanup(self.trainer.stop)

    def train(self, task_name, diagram, pairs, seeds, epochs, batch_size):
        return pd.DataFrame([
            dict(seed=seed, training=condition, state_1='x', state_2='y',
                 type='immediate', accuracy=0.75)
            for seed in seeds for condition in ('untrained', 'trained')
        ], columns=analysis.COLUMNS)

    def run_cached(self, seeds=range(2), epochs=10, recompute=False, output=None):
        return analysis.cached_results(
            'limitedstack', self.diagram, self.pairs, seeds, epochs, 128,
            output or self.output, recompute=recompute)

    def test_completed_run_does_not_train_again(self):
        first = self.run_cached()
        self.assertEqual(self.mock_train.call_count, 2)
        self.mock_train.reset_mock()
        second = self.run_cached()
        self.mock_train.assert_not_called()
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(self.output.exists())
        self.assertTrue(self.output.with_suffix('.json').exists())

    def test_increasing_seed_count_only_trains_missing_seed(self):
        self.run_cached()
        self.mock_train.reset_mock()
        results = self.run_cached(seeds=range(3), output=self.root / 'more_seeds.csv')
        self.mock_train.assert_called_once()
        self.assertEqual(self.mock_train.call_args.kwargs['seeds'], [2])
        self.assertEqual(len(results), 6)

    def test_legacy_csv_is_reused_without_training(self):
        self.train(None, None, None, range(2), None, None).to_csv(
            self.output, index=False)
        self.run_cached()
        self.mock_train.assert_not_called()

    def test_recompute_replaces_requested_scores(self):
        self.run_cached()
        self.mock_train.reset_mock()
        self.run_cached(recompute=True)
        self.assertEqual(self.mock_train.call_count, 2)

    def test_changed_training_settings_do_not_reuse_scores(self):
        self.run_cached()
        self.mock_train.reset_mock()
        self.run_cached(epochs=20)
        self.assertEqual(self.mock_train.call_count, 2)
        metadata = json.loads(self.output.with_suffix('.json').read_text())
        self.assertEqual(metadata['epochs'], 20)
        self.mock_train.reset_mock()
        self.run_cached(epochs=10)
        self.mock_train.assert_not_called()

    def test_interrupted_run_checkpoints_completed_seed(self):
        def interrupted(*args, **kwargs):
            if kwargs['seeds'] == [1]:
                raise RuntimeError('interrupted')
            return self.train(*args, **kwargs)

        self.mock_train.side_effect = interrupted
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.run_cached()
        self.assertFalse(self.output.exists())
        self.mock_train.side_effect = self.train
        self.mock_train.reset_mock()
        results = self.run_cached()
        self.mock_train.assert_called_once()
        self.assertEqual(self.mock_train.call_args.kwargs['seeds'], [1])
        self.assertEqual(len(results), 4)

    def test_partial_seed_is_retrained_without_duplicating_rows(self):
        rows = self.train(None, None, None, range(2), None, None)
        rows.iloc[:-1].to_csv(self.output, index=False)
        results = self.run_cached()
        self.mock_train.assert_called_once()
        self.assertEqual(self.mock_train.call_args.kwargs['seeds'], [1])
        self.assertEqual(len(results), 4)


if __name__ == '__main__':
    unittest.main()
