import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from scripts import constrained_penalty_sweep as sweep
from scripts import readout_scale_experiment as experiment


class ReadoutScaleExperimentTests(unittest.TestCase):
    def test_experiment_runs_all_scales_and_seeds_and_writes_summaries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def fake_run(root_arg, lnorm, seed, epochs, *, readout_norm, readout_scale):
                self.assertEqual(root_arg, root)
                self.assertEqual(lnorm, 0)
                self.assertEqual(readout_norm, 'fixed')
                run_id = f'{readout_scale}-{seed}'
                directory = root / 'runs' / run_id
                directory.mkdir(parents=True)
                torch.save({'training_history': dict(epochs=[0, 100],
                           acc=[0.6, 0.8], loss=[1.0, 0.5])},
                           directory / 'network.pt')
                return dict(run_id=run_id, seed=seed, readout_scale=readout_scale,
                            train_accuracy=0.8, test_accuracy=0.7)
            with patch.object(sweep, 'OUTPUT_DIR', root), \
                    patch.object(sweep, 'run_one', side_effect=fake_run) as run:
                directory = experiment.run_experiment(
                    [1.0, 2.0, 4.0, 8.0], [1, 2], 200, output_dir=root)
            self.assertEqual(run.call_count, 8)
            runs = pd.read_csv(directory / 'runs.csv')
            self.assertEqual(runs.groupby('readout_scale').size().tolist(), [2] * 4)
            history = pd.read_csv(directory / 'learning_curves.csv')
            self.assertEqual(set(history['epoch']), {1, 101})
            self.assertEqual(len(history), 16)
            self.assertEqual(pd.read_csv(directory / 'summary.csv')['networks'].tolist(),
                             [2] * 4)
            self.assertTrue((directory / 'accuracy.png').exists())
            self.assertTrue((directory / 'learning_curves.png').exists())

    def test_invalid_scale_rejected_before_training(self):
        for value in ('0', '-1', 'nan'):
            with self.subTest(value=value), \
                    patch('sys.argv', ['experiment', '--scales', value]), \
                    patch.object(experiment, 'run_experiment') as run:
                with self.assertRaises(SystemExit):
                    experiment.main()
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
