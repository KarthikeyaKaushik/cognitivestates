import copy
import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts import fig7_phase_transitions as figure
from src.training import train_model


class TinyDiagram:
    X = Y = {0, 1}
    states = {0, 1, 2}
    state_labels = {0: 'a', 1: 'b', 2: 'c'}

    def trace(self, inputs):
        return [index % 3 for index in range(len(inputs))]


def tiny_lattice():
    return SimpleNamespace(nodes=[
        SimpleNamespace(id=1, pairwise_matrix=np.ones((3, 3))),
        SimpleNamespace(id=2, pairwise_matrix=np.eye(3)),
        SimpleNamespace(id=3, pairwise_matrix=np.array([[1, 1, 0],
                                                       [1, 1, 0],
                                                       [0, 0, 1]])),
    ])


def tiny_data():
    return [(inputs, inputs) for inputs in itertools.product((0, 1), repeat=3)]


class PhaseTransitionTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_loss_score_matches_legacy_and_checkpoints_are_valid(self):
        loss = np.array([3, 2.9, 2.8, 2.7, 2.6, 2.5, 1.1, 1, 0.9, 0.8, 0.7])
        delta = 2
        expected = [0, 0] + [
            max(0, abs(loss[i + delta] - loss[i]) - abs(loss[i] - loss[i - delta]))
            for i in range(delta, len(loss) - delta)
        ] + [0, 0]
        history = figure.identify_breakthroughs(
            pd.DataFrame(dict(loss=loss, epochs=np.arange(len(loss)))), delta)
        np.testing.assert_array_equal(history['values'], expected)
        before, after = figure.transition_epochs(history, delta)
        self.assertEqual(after - before, 2 * delta)
        self.assertGreaterEqual(before, 0)
        self.assertLess(after, len(history))
        with self.assertRaisesRegex(ValueError, 'No positive'):
            figure.transition_epochs(
                figure.identify_breakthroughs(pd.DataFrame(
                    dict(loss=np.ones(7), epochs=np.arange(7))), 2), 2)
        with self.assertRaises(ValueError):
            figure.identify_breakthroughs(history, 6)

    def test_proposals_exclude_undefined_auc_and_are_reproducible(self):
        first = figure.make_proposals(tiny_lattice(), 10, 0)
        second = figure.make_proposals(tiny_lattice(), 10, 0)
        self.assertEqual(len(first), 11)
        self.assertEqual(set(first) - {f'rand_{i}' for i in range(10)}, {'cc_3'})
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
            self.assertEqual(set(first[key]), {0, 1})
        matrix = np.array([[1, 0.9, 0.6], [0.9, 1, 0.5], [0.6, 0.5, 1]])
        scores = figure.score_proposals(matrix, first, 0, 2, 't+')
        self.assertEqual(scores.loc[scores['candidate'] == 'cc_3', 'auc'].iloc[0], 1)

    def test_winners_selected_after_transition_and_reused_before(self):
        scores = pd.DataFrame([
            dict(seed=0, epoch=epoch, t=t, comparison=family,
                 candidate=f'{family}_{candidate}', auc=auc)
            for family in ('cc', 'rand')
            for candidate, pair in ((1, [0.99, 0.7]), (2, [0.6, 0.9]))
            for t, epoch, auc in zip(('t-', 't+'), (1, 3), pair)
        ])
        summary = figure.select_winners(scores)
        self.assertEqual(set(summary['candidate']), {'cc_2', 'rand_2'})
        np.testing.assert_array_equal(summary.loc[summary['t'] == 't-', 'auc'], [0.6, 0.6])

    def test_training_matches_shared_trainer_and_probes_do_not_change_rng(self):
        torch.manual_seed(0)
        data = figure.DataGenerator(tiny_data(), TinyDiagram(), batch_size=3)
        model = figure.ConstrainedRNN(2, 3, 2, seed=0, lnorm=1.2)
        initial = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state()
        train_model(model, data, 0.001, 4)
        expected = copy.deepcopy(model.state_dict())
        model.load_state_dict(initial)
        torch.set_rng_state(rng)
        config = dict(figure.DEFAULTS, epochs=4)
        calls = []

        def probe(network, epoch):
            calls.append(epoch)
            figure.get_hidden_states(network, data)
            torch.rand(100)

        history = figure.train_trajectory(model, data, config, (0, 3), probe)
        self.assertEqual(calls, [0, 3])
        self.assertEqual(len(history), 4)
        for key in expected:
            torch.testing.assert_close(model.state_dict()[key], expected[key], rtol=0, atol=0)

    def test_small_analysis_replays_exactly_and_generates_all_four_figures(self):
        config = dict(figure.DEFAULTS, epochs=4, delta=1, num_random=3, hidden_dim=3)

        def mark_transition(history, delta):
            history['values'] = [0, 1, 0, 0]
            return history

        with patch.object(figure, 'get_task_diagram', return_value=TinyDiagram()), \
                patch.object(figure, 'get_task_lattice', return_value=tiny_lattice()), \
                patch.dict(figure.TASKS, limitedstack={'generate': lambda _: tiny_data()}), \
                patch.object(figure, 'identify_breakthroughs', side_effect=mark_transition):
            history, scores, summary, heatmaps, labels = figure.compute_analysis(config, 0)
        self.assertEqual(len(history), 4)
        self.assertEqual(len(scores), 8)
        self.assertEqual(len(summary), 4)
        self.assertEqual(set(summary['epoch']), {0, 2})
        self.assertEqual(set(heatmaps), {'t-', 't+'})
        for matrix in heatmaps.values():
            self.assertEqual(matrix.shape, (3, 3))
        with tempfile.TemporaryDirectory() as temporary, \
                patch('scripts.common.FIGURES_DIR', Path(temporary)):
            (Path(temporary) / 'panels').mkdir()
            figure.plot_seed(history, summary, heatmaps, labels, 0, 'panels')
            figure.plot_comparison(summary)
            outputs = list(Path(temporary).rglob('*.png'))
            self.assertEqual(len(outputs), 4)
            self.assertTrue(all(path.stat().st_size > 1000 for path in outputs))

    def test_cache_reuses_results_and_separates_seeds_and_configurations(self):
        config = figure.DEFAULTS.copy()
        history = pd.DataFrame(dict(epochs=[0], loss=[1.0], acc=[0.5], values=[0.0]))
        scores = pd.DataFrame(dict(auc=[0.75]))
        summary = pd.DataFrame(dict(seed=[0], t=['t-'], auc=[0.75]))
        result = (history, scores, summary, {'t-': np.eye(3), 't+': np.eye(3)},
                  ['a', 'b', 'c'])
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(figure, 'RESULTS_DIR', Path(temporary)), \
                patch.object(figure, 'compute_analysis', return_value=result) as compute:
            figure.load_or_compute(config, 0)
            compute.reset_mock()
            cached = figure.load_or_compute(config, 0)
            compute.assert_not_called()
            pd.testing.assert_frame_equal(cached[0], history)
            pd.testing.assert_frame_equal(cached[2], summary)
            np.testing.assert_array_equal(cached[3]['t+'], np.eye(3))
            self.assertEqual(cached[4], ['a', 'b', 'c'])
            figure.load_or_compute(config, 0, recompute=True)
            compute.assert_called_once()
            compute.reset_mock()
            figure.load_or_compute(config, 1)
            compute.assert_called_once()
            compute.reset_mock()
            changed = dict(config, epochs=1000)
            figure.load_or_compute(changed, 0)
            compute.assert_called_once()
            compute.reset_mock()
            key, _ = figure.configuration_key(changed)
            (Path(temporary) / 'phase_transitions' / key / 'seed0' / 'scores.csv').unlink()
            figure.load_or_compute(changed, 0)
            compute.assert_called_once()


if __name__ == '__main__':
    unittest.main()
