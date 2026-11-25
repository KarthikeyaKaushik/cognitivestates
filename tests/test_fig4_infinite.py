import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts import fig4_infinite as figure

LEGACY_RW = Path(__file__).resolve().parents[1] / 'scripts_old/run_rescorla_wagner.py'


class InfiniteFigureTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    @unittest.skipUnless(LEGACY_RW.exists(), 'legacy scripts_old/ not available')
    def test_rw_matches_legacy_training_and_projection(self):
        spec = importlib.util.spec_from_file_location('legacy_rw', LEGACY_RW)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        config = dict(figure.CONFIGS['rescorla_wagner'], length=8, samples=20, epochs=2)
        pairs = figure.rw_sequences(8, 20, alpha=0.5, seed=0)

        torch.manual_seed(42)
        old_model = legacy.ConstrainedRNN(2, 6, 1, seed=0, lnorm=0.5)
        old_data = legacy.RWDataGenerator(pairs, {0, 1})
        old_hidden, old_states, _, _ = legacy.get_hidden_states(old_model, old_data)
        old_projection = legacy.project_low_dim(
            old_hidden.detach().numpy().reshape(-1, 6),
            old_states.numpy().reshape(-1), n_components=2)
        legacy.train_model(old_model, old_data, 0.001, 2, 10)

        torch.manual_seed(42)
        new_model = figure.ConstrainedRNN(2, 6, 1, seed=0, lnorm=0.5)
        new_data = figure.SequenceData(pairs, {0, 1})
        new_hidden, new_states = figure.collect_hidden(new_model, new_data, False)
        np.testing.assert_array_equal(new_hidden.numpy(),
                                      old_hidden.detach().numpy().reshape(-1, 6))
        pd.testing.assert_frame_equal(
            figure.pls_projection(new_hidden, new_states), old_projection)
        figure.train_network(new_model, new_data, config, False)
        for key, value in old_model.state_dict().items():
            torch.testing.assert_close(value, new_model.state_dict()[key], rtol=0, atol=0)

    def test_majority_counts_and_targets(self):
        torch.manual_seed(42)
        data = figure.SequenceData([
            ((1, 0, 0, 1), (1, 0, 0, 0)),
            ((0, 1, 1, 0), (0, 0, 1, 0)),
        ], {0, 1}, discrete=True)
        inputs, targets, states = data.samples[0]
        self.assertEqual(states.tolist(), [1, 0, -1, 0])
        self.assertEqual(targets.tolist(), [1, 0, 0, 0])
        self.assertEqual(inputs.argmax(dim=-1).tolist(), [1, 0, 0, 1])

    def test_majority_end_to_end_and_plots(self):
        config = dict(figure.CONFIGS['majority'], samples=100, length=30, epochs=1)
        result, history = figure.compute_analysis('majority', config)
        self.assertEqual(list(result.columns), ['x', 'y', 'states', 'train_states'])
        self.assertTrue((result.groupby('states').size() == 100).all())
        self.assertTrue(result['train_states'].any())
        self.assertTrue((~result['train_states']).any())
        self.assertEqual(len(history), 1)
        self.assertTrue(np.isfinite(result[['x', 'y']]).all().all())
        with patch.object(figure, 'save_figure') as save:
            figure.plot_majority(result)
        self.assertEqual(save.call_count, 2)

    def test_cached_run_never_retrains_and_config_changes_are_separate(self):
        result = pd.DataFrame(dict(z_1=[0.1], z_2=[0.2], y=[0.5], model=['trained']))
        history = pd.DataFrame(dict(epoch=[0], loss=[0.1]))
        config = figure.CONFIGS['rescorla_wagner'].copy()
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(figure, 'RESULTS_DIR', Path(temporary)), \
                patch.object(figure, 'compute_analysis', return_value=(result, history)) as run:
            figure.load_or_compute('rescorla_wagner', config)
            run.reset_mock()
            cached, _, _ = figure.load_or_compute('rescorla_wagner', config)
            run.assert_not_called()
            pd.testing.assert_frame_equal(cached, result)
            figure.load_or_compute('rescorla_wagner', config, recompute=True)
            run.assert_called_once()
            run.reset_mock()
            figure.load_or_compute('rescorla_wagner', dict(config, epochs=2))
            run.assert_called_once()

    def test_rw_end_to_end_and_plots(self):
        config = dict(figure.CONFIGS['rescorla_wagner'], samples=20, length=8, epochs=1)
        result, history = figure.compute_analysis('rescorla_wagner', config)
        self.assertEqual(set(result['model']), {'untrained', 'trained'})
        self.assertEqual(len(result), 2 * 16 * 7)
        self.assertEqual(len(history), 1)
        with patch.object(figure, 'save_figure') as save:
            figure.plot_rw(result)
        self.assertEqual(save.call_count, 2)


if __name__ == '__main__':
    unittest.main()
