import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from src import experiments


class DecodingStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.hidden_path = self.root / 'run.pt'
        self.history_path = self.root / 'history.csv'
        self.main_path = self.root / 'heldout.csv'
        self.run = {}
        for split in ('train', 'test'):
            self.run[f'hidden_{split}'] = torch.zeros(2, 3, 2)
            self.run[f'inputs_{split}'] = torch.zeros(2, 3, dtype=torch.long)
            self.run[f'truths_{split}'] = torch.tensor([[[1., 0.]] * 3] * 2)
        self.scores = [dict(k=k, accuracy=0.75, n_history_classes=2,
                            history_train_samples_per_class=2,
                            n_projection_train_samples=4, n_components=1,
                            n_state_classes=2, n_train_samples=4, n_test_samples=4,
                            status='ok', reason='') for k in (1, 2, 3)]
        for name, value in [('HISTORY_ACCURACIES_CSV', self.history_path),
                            ('ACCURACIES_CSV', self.main_path),
                            ('RESULTS_DIR', self.root),
                            ('HIDDEN_STATES_DIR', self.root)]:
            p = patch.object(experiments, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(experiments, '_hidden_path', return_value=self.hidden_path)
        p.start()
        self.addCleanup(p.stop)

    def test_reuses_aligned_cache_and_resumes_without_training(self):
        torch.save(self.run, self.hidden_path)
        with patch.object(experiments, '_train_run') as train, \
                patch.object(experiments, 'history_decoding_scores',
                             return_value=self.scores) as score:
            rows = experiments.history_run_accuracies('nback3', 1)
            self.assertEqual(len(rows), 3)
            self.assertFalse(self.hidden_path.exists())
            repeated = experiments.history_run_accuracies('nback3', 1)
            self.assertEqual(len(repeated), 3)
            self.assertEqual(score.call_count, 1)
            self.assertIn('truths_train', score.call_args.kwargs)
            self.assertIn('truths_test', score.call_args.kwargs)
            train.assert_not_called()
        self.assertEqual(len(pd.read_csv(self.history_path)), 3)

    def test_old_cache_requires_backfill_and_keeps_scatter_seed(self):
        torch.save({'hidden_test': self.run['hidden_test']}, self.hidden_path)
        with patch.object(experiments, '_train_run', return_value=self.run) as train, \
                patch.object(experiments, 'history_decoding_scores',
                             return_value=self.scores):
            experiments.history_run_accuracies('nback3', 0, keep_hidden=True)
            train.assert_called_once_with('nback3', 0, decode=False)
        self.assertIn('inputs_train', torch.load(self.hidden_path, weights_only=False))

    def test_recompute_replaces_only_requested_run_and_repairs_partial_rows(self):
        with patch.object(experiments, '_train_run', return_value=self.run) as train, \
                patch.object(experiments, 'history_decoding_scores',
                             return_value=self.scores):
            experiments.history_run_accuracies('nback3', 0)
            experiments.history_run_accuracies('nback3', 1)
            experiments.history_run_accuracies('nback3', 0, recompute=True)
            self.assertEqual(train.call_count, 3)
            pd.read_csv(self.history_path).iloc[:5].to_csv(self.history_path, index=False)
            experiments.history_run_accuracies('nback3', 0)
            self.assertEqual(train.call_count, 4)
        rows = pd.read_csv(self.history_path)
        self.assertEqual(rows.groupby('seed').size().to_dict(), {0: 3, 1: 3})

    def test_history_cannot_be_requested_as_training_accuracy(self):
        with self.assertRaisesRegex(ValueError, 'only on the test'):
            experiments.history_run_accuracies('nback3', 0, split='train')

    def test_listmax_computes_and_resumes_only_two_history_lengths(self):
        torch.save(self.run, self.hidden_path)
        with patch.object(experiments, '_train_run') as train, \
                patch.object(experiments, 'history_decoding_scores',
                             return_value=self.scores[:2]) as score:
            rows = experiments.history_run_accuracies('listmax', 0, keep_hidden=True)
            self.assertEqual(rows['k'].tolist(), [1, 2])
            self.assertEqual(score.call_args.kwargs['max_k'], 2)
            repeated = experiments.history_run_accuracies('listmax', 0, keep_hidden=True)
            self.assertEqual(repeated['k'].tolist(), [1, 2])
            self.assertEqual(score.call_count, 1)
            train.assert_not_called()
        self.assertEqual(pd.read_csv(self.history_path)['k'].tolist(), [1, 2])

    def decoded_run(self, run, seed):
        run['accs_test'] = {method: 0.6 for method in experiments.METHODS}
        run['acc_lda_shuffled_test'] = 0.5
        run['decoding_version'] = experiments.DECODING_VERSION
        return run

    def test_combined_scores_share_one_training_run_and_resume_from_both_csvs(self):
        with patch.object(experiments, '_train_run', return_value=self.run) as train, \
                patch.object(experiments, '_ensure_decoding',
                             side_effect=self.decoded_run) as decode, \
                patch.object(experiments, 'history_decoding_scores',
                             return_value=self.scores) as history:
            main, rows = experiments.comparison_run_accuracies('nback3', 1)
            self.assertEqual((len(main), len(rows)), (5, 3))
            train.assert_called_once_with('nback3', 1, decode=False)
            self.assertFalse(self.hidden_path.exists())
            cached_main, cached_history = experiments.comparison_run_accuracies('nback3', 1)
            self.assertEqual((len(cached_main), len(cached_history)), (5, 3))
            self.assertEqual(train.call_count, 1)
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(history.call_count, 1)

    def test_combined_reuses_hidden_cache_when_only_history_rows_exist(self):
        torch.save(self.run, self.hidden_path)
        with patch.object(experiments, 'history_decoding_scores',
                          return_value=self.scores):
            experiments.history_run_accuracies('nback3', 0, keep_hidden=True)
        with patch.object(experiments, '_train_run') as train, \
                patch.object(experiments, '_ensure_decoding',
                             side_effect=self.decoded_run), \
                patch.object(experiments, 'history_decoding_scores') as history:
            main, rows = experiments.comparison_run_accuracies(
                'nback3', 0, keep_hidden=True)
        train.assert_not_called()
        history.assert_not_called()
        self.assertEqual((len(main), len(rows)), (5, 3))
        self.assertTrue(self.hidden_path.exists())

    def test_combined_backfills_missing_inputs_without_changing_cached_main_scores(self):
        main = experiments._accuracy_rows('listmax', 0, self.decoded_run(self.run, 0))
        experiments._store_run_rows(main)
        torch.save({'hidden_test': self.run['hidden_test']}, self.hidden_path)
        with patch.object(experiments, '_train_run', return_value=self.run) as train, \
                patch.object(experiments, '_ensure_decoding') as decode, \
                patch.object(experiments, 'history_decoding_scores',
                             return_value=self.scores[:2]) as history:
            saved_main, rows = experiments.comparison_run_accuracies(
                'listmax', 0, keep_hidden=True)
        train.assert_called_once_with('listmax', 0, decode=False)
        decode.assert_not_called()
        self.assertEqual(history.call_args.kwargs['max_k'], 2)
        self.assertEqual(saved_main['accuracy'].tolist(), [0.6] * 4 + [0.5])
        self.assertEqual(rows['k'].tolist(), [1, 2])

    def test_combined_recompute_trains_once_and_replaces_both_score_tables(self):
        with patch.object(experiments, '_train_run', return_value=self.run) as train, \
                patch.object(experiments, '_ensure_decoding',
                             side_effect=self.decoded_run), \
                patch.object(experiments, 'history_decoding_scores',
                             return_value=self.scores):
            experiments.comparison_run_accuracies('nback3', 0, keep_hidden=True)
            experiments.comparison_run_accuracies(
                'nback3', 0, recompute=True, keep_hidden=True)
        self.assertEqual(train.call_count, 2)
        self.assertEqual(len(pd.read_csv(self.main_path)), 5)
        self.assertEqual(len(pd.read_csv(self.history_path)), 3)
        self.assertTrue(self.hidden_path.exists())

    def test_main_rescores_legacy_hidden_cache_instead_of_reusing_old_accuracy(self):
        self.run['accs_test'] = {method: 1. for method in experiments.METHODS}
        self.run['acc_lda_shuffled_test'] = 1.
        torch.save(self.run, self.hidden_path)
        corrected = {method: 0.6 for method in experiments.METHODS}
        with patch.object(experiments, '_train_run') as train, \
                patch.object(experiments, 'project_and_score',
                             side_effect=[({}, corrected, None),
                                          ({}, {'LDA': 0.5}, None)]) as score:
            rows = experiments.run_accuracies('nback3', 0, keep_hidden=True)
            self.assertEqual(score.call_count, 2)
            train.assert_not_called()
        self.assertEqual(set(rows['split']), {'test'})
        self.assertEqual(set(rows['accuracy']), {0.6, 0.5})
        cached = torch.load(self.hidden_path, weights_only=False)
        self.assertEqual(cached['decoding_version'], experiments.DECODING_VERSION)

    def test_train_run_captures_current_inputs_and_truths_in_same_order(self):
        hidden = torch.arange(24).reshape(2, 4, 3).float()
        ids = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
        inputs = torch.nn.functional.one_hot(ids, 2).float()
        truths = 1 - inputs
        with patch.object(experiments, 'get_task_diagram') as diagram, \
                patch.object(experiments, 'DataGenerator'), \
                patch.object(experiments, 'ConstrainedRNN'), \
                patch.object(experiments, 'train_model',
                             return_value=(object(), {'loss': [0.1], 'acc': [1.]})), \
                patch.object(experiments, 'get_hidden_states',
                             return_value=(hidden, truths, inputs, None)):
            diagram.return_value.state_labels = {0: 'a', 1: 'b'}
            cfg = dict(experiments.TASKS['nback3'], generate=lambda _: [])
            with patch.dict(experiments.TASKS, {'nback3': cfg}):
                run = experiments._train_run('nback3', 0, decode=False)
        for split in ('train', 'test'):
            torch.testing.assert_close(run[f'hidden_{split}'], hidden[:, :-1])
            torch.testing.assert_close(run[f'truths_{split}'], truths[:, :-1])
            torch.testing.assert_close(run[f'inputs_{split}'], ids[:, :-1])


if __name__ == '__main__':
    unittest.main()
