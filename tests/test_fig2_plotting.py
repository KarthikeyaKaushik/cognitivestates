import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1] / 'scripts'),
                                *sys.path]):
    import fig3_task_panels as panels

from src.experiments import SHUFFLED_METHOD


class AccuracyPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(plt.close, 'all')
        self.root = Path(self.temp.name)
        self.main = pd.DataFrame([
            dict(seed=seed, split='test', method=method, accuracy=0.6 + seed * 0.01)
            for seed in range(2)
            for method in panels.METHODS + [SHUFFLED_METHOD]])
        self.main.loc[self.main['method'] == SHUFFLED_METHOD, 'accuracy'] = 0.1
        self.history = pd.DataFrame([
            dict(seed=seed, split='test', k=k, accuracy=0.8 + seed * 0.01)
            for seed in range(2) for k in (1, 2, 3)])
        self.captured = {}

    def save(self, name):
        ax = plt.gca()
        self.captured[name] = dict(
            labels=[label.get_text() for label in ax.get_xticklabels()],
            annotations=[text.get_text() for text in ax.texts],
            patches=len(ax.patches), autoscale=ax.get_autoscaley_on(),
            limits=ax.get_ylim())
        plt.savefig(self.root / f'{name}.png', dpi=60)

    def test_main_and_supplementary_have_requested_conditions_and_seed_values(self):
        with patch.object(panels, 'save_figure', side_effect=self.save) as save, \
                patch.object(panels.sns, 'boxplot', wraps=panels.sns.boxplot) as box:
            panels.save_comparison_boxplots('nback3', self.main, self.history)
        self.assertEqual([call.args[0] for call in save.call_args_list],
                         ['nback3_simulations', 'nback3_supplementary'])
        main = self.captured['nback3_simulations']
        supplementary = self.captured['nback3_supplementary']
        self.assertEqual(main['labels'], ['LDA', 'PCA', 'ICA', 'FA'])
        self.assertEqual(supplementary['labels'],
                         ['LDA', 'k=1', 'k=2', 'k=3', 'Random'])
        self.assertEqual((main['patches'], supplementary['patches']), (4, 5))
        main_data, data = [call.kwargs['data'] for call in box.call_args_list]
        self.assertEqual(main_data.groupby('method').size().tolist(), [2] * 4)
        self.assertEqual(data.groupby('method').size().tolist(), [2] * 5)
        self.assertNotIn(SHUFFLED_METHOD, set(main_data['method']) | set(data['method']))
        self.assertEqual(
            self.main.loc[self.main['method'] == SHUFFLED_METHOD, 'accuracy'].tolist(),
            [0.1, 0.1])
        self.assertEqual(data[data['method'] == 'k=1']['accuracy'].tolist(),
                         [0.8, 0.81])
        self.assertEqual(data[data['method'] == 'LDA']['accuracy'].tolist(),
                         main_data[main_data['method'] == 'LDA']['accuracy'].tolist())
        for name, plot in self.captured.items():
            self.assertTrue(plot['autoscale'])
            # The supplementary plot includes the near-chance Random column.
            if name.endswith('_simulations'):
                self.assertGreater(plot['limits'][0], 0.5)
            self.assertLess(plot['limits'][1], 0.9)
            self.assertGreater((self.root / f'{name}.png').stat().st_size, 0)
        self.assertEqual(plt.get_fignums(), [])

    def test_listmax_omits_k_three_even_if_old_history_rows_are_supplied(self):
        with patch.object(panels, 'save_figure', side_effect=self.save):
            panels.save_comparison_boxplots('listmax', self.main, self.history)
        supplementary = self.captured['listmax_supplementary']
        self.assertEqual(supplementary['labels'],
                         ['LDA', 'k=1', 'k=2', 'Random'])
        self.assertEqual(supplementary['annotations'], [])
        self.assertEqual(self.captured['listmax_simulations']['labels'], panels.METHODS)

    def test_unavailable_history_is_annotated_without_hiding_other_conditions(self):
        self.history.loc[self.history['k'] == 3, 'accuracy'] = np.nan
        with patch.object(panels, 'save_figure', side_effect=self.save):
            panels.save_comparison_boxplots('nback3', self.main, self.history)
        supplementary = self.captured['nback3_supplementary']
        self.assertEqual(supplementary['annotations'], ['Unavailable'])
        self.assertEqual(len(supplementary['labels']), 5)
        self.assertEqual(supplementary['patches'], 4)
        self.assertEqual(self.captured['nback3_simulations']['annotations'], [])

    def test_both_cli_modes_produce_accuracy_plots_and_only_default_generates_scatters(self):
        run = dict(state_labels={0: 'a'}, hidden_train=None, truths_train=None,
                   hidden_test=None, truths_test=None)
        projections = {'LDA': np.zeros((1, 2)), 'PCA': np.zeros((1, 2))}
        for flag in ([], ['--history-decoding']):
            with self.subTest(flag=flag), \
                    patch.object(sys, 'argv', ['fig3_task_panels.py', '--tasks', 'listmax',
                                              '--seeds', '1', '--recompute', *flag]), \
                    patch.object(panels, 'comparison_run_accuracies',
                                 return_value=(self.main, self.history)) as scores, \
                    patch.object(panels, 'save_comparison_boxplots') as plot, \
                    patch.object(panels, 'scatter_run', return_value=run) as scatter, \
                    patch.object(panels, 'untrained_run', return_value=run), \
                    patch.object(panels, 'project_and_score',
                                 return_value=(projections, None, np.array([0]))), \
                    patch.object(panels, 'save_scatter') as save_scatter:
                panels.main()
            scores.assert_called_once_with(
                'listmax', 0, recompute=True, keep_hidden=True)
            plot.assert_called_once()
            self.assertEqual(scatter.call_count, 0 if flag else 1)
            self.assertEqual(save_scatter.call_count, 0 if flag else 3)


if __name__ == '__main__':
    unittest.main()
