import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import numpy as np
import torch

from scripts import constrained_penalty_sweep as sweep


class FirstClassModel(torch.nn.Module):
    def forward(self, inputs):
        logits = torch.zeros(*inputs.shape[:2], 2)
        logits[..., 0] = 1
        return logits, None


class UnevenBatches:
    def batch_generator(self, train=True):
        targets = ([0, 0, 0, 1], [1]) if train else ([1, 0],)
        for classes in targets:
            target = torch.nn.functional.one_hot(torch.tensor([classes]), 2)
            yield torch.zeros(1, len(classes), 1), target, None


class PenaltySweepTests(unittest.TestCase):
    def test_accuracy_counts_timesteps_not_batches(self):
        model = FirstClassModel()
        self.assertEqual(sweep.evaluate_accuracy(model, UnevenBatches(), train=True), 0.6)
        self.assertEqual(sweep.evaluate_accuracy(model, UnevenBatches(), train=False), 0.5)

    def test_default_sweep_visits_all_runs_and_saves_progress(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch('sys.argv', ['sweep', '--output-dir', temp]), \
                patch.object(sweep, 'run_one', side_effect=lambda *args, **kwargs: {}) as run, \
                patch.object(sweep, 'save_summary', return_value=Path(temp)) as save:
            sweep.main()
        self.assertEqual(run.call_args_list, [
            call(Path(temp), lnorm, seed, 2000, retrain=False, readout_norm='fixed',
                 readout_scale=1.0)
            for lnorm in sweep.LNORMS for seed in sweep.SEEDS
        ])
        expected = len(sweep.LNORMS) * len(sweep.SEEDS)
        self.assertEqual(save.call_count, expected)
        self.assertEqual(len(save.call_args.args[2]), expected)

    def test_learnable_condition_is_forwarded_to_all_runs(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch('sys.argv', ['sweep', '--output-dir', temp, '--lnorms', '0.2',
                                   '--seeds', '1', '--readout-norm', 'learnable']), \
                patch.object(sweep, 'run_one', return_value={}) as run, \
                patch.object(sweep, 'save_summary', return_value=Path(temp)):
            sweep.main()
        run.assert_called_once_with(Path(temp), 0.2, 1, 2000,
                                    retrain=False, readout_norm='learnable',
                                    readout_scale=1.0)

    def test_invalid_inputs_fail_before_training(self):
        for options in (['--lnorms', '-1'], ['--lnorms', 'nan'],
                        ['--lnorms', '0.5', '0.5'], ['--seeds', '0'],
                        ['--seeds', '1', '1'], ['--epochs', '0'],
                        ['--readout-scale', '0'],
                        ['--readout-scale', '2', '--readout-norm', 'learnable']):
            with self.subTest(options=options), \
                    patch('sys.argv', ['sweep', *options]), \
                    patch.object(sweep, 'run_one') as run:
                with self.assertRaises(SystemExit):
                    sweep.main()
                run.assert_not_called()

    def test_summary_saves_unfiltered_per_run_accuracies(self):
        rows = [dict(seed=1, lnorm=0.4, train_accuracy=0.8, test_accuracy=0.7),
                dict(seed=2, lnorm=0.4, train_accuracy=0.6, test_accuracy=0.5)]
        with tempfile.TemporaryDirectory() as temp:
            directory = sweep.save_summary(Path(temp), 'test', rows, [0.4, 0.5])
            self.assertTrue((directory / 'accuracy.png').exists())
            import pandas as pd
            summary = pd.read_csv(directory / 'summary.csv')
            self.assertEqual(summary.loc[0, 'networks'], 2)
            self.assertAlmostEqual(summary.loc[0, 'test_mean'], 0.6)
            self.assertAlmostEqual(summary.loc[0, 'test_sem'], 0.1)
            self.assertEqual(len(pd.read_csv(directory / 'runs.csv')), 2)

    def test_heatmap_cache_is_invalidated_by_changed_checkpoint(self):
        class Diagram:
            X = {'x'}
            Y = {'n'}

        class Batches:
            train = [1]
            test = [2]

            def __init__(self, *args, **kwargs):
                pass

        def network(directory, seed, inputs, outputs, **kwargs):
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / 'network.pt'
            if not path.exists():
                path.write_bytes(b'checkpoint')
            return FirstClassModel(), {'seed': seed}

        with tempfile.TemporaryDirectory() as temp, \
                patch.object(sweep, 'get_task_diagram', return_value=Diagram()), \
                patch.object(sweep, 'load_or_train_network', side_effect=network), \
                patch.object(sweep, 'DataGenerator', Batches), \
                patch.object(sweep, 'evaluate_accuracy', return_value=0.75), \
                patch.object(sweep, 'readout_weight_norm', return_value=1.5), \
                patch.object(sweep, 'save_decoding_heatmap') as heatmap:
            root = Path(temp)
            row = sweep.run_one(root, 0.4, 1, 2)
            directory = root / 'runs' / row['run_id']
            metadata = heatmap.call_args.args[1]
            np.savez(directory / 'decoding_heatmap.npz',
                     metadata=json.dumps(metadata))
            sweep.run_one(root, 0.4, 1, 2)
            self.assertFalse(heatmap.call_args.kwargs['recompute'])
            (directory / 'network.pt').write_bytes(b'different checkpoint')
            sweep.run_one(root, 0.4, 1, 2)
            self.assertTrue(heatmap.call_args.kwargs['recompute'])
            metrics = json.loads((directory / 'metrics.json').read_text())
            self.assertEqual(metrics['metrics']['train_accuracy'], 0.75)
            self.assertEqual(metrics['metrics']['readout_weight_norm'], 1.5)
            self.assertEqual(metrics['configuration']['split_seed'], 0)
            self.assertTrue(metrics['configuration']['fixed_readout_norm'])
            learnable = sweep.run_one(root, 0.4, 1, 2, readout_norm='learnable')
            self.assertNotEqual(row['run_id'], learnable['run_id'])
            self.assertEqual(learnable['readout_norm'], 'learnable')
            no_penalty = sweep.run_one(root, 0.0, 1, 2)
            self.assertEqual(no_penalty['penalty'], 0)
            self.assertEqual(no_penalty['readout_norm'], 'fixed')
            self.assertNotEqual(no_penalty['run_id'], row['run_id'])


if __name__ == '__main__':
    unittest.main()
