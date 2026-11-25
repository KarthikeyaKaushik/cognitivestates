import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from scripts import fig5_constrained_stack as figure


class Coloring:
    def __init__(self, color_id, matrix, count):
        self.id = color_id
        self.pairwise_matrix = np.asarray(matrix)
        self.count = count

    def get_sets(self):
        return list(range(self.count))


class Lattice:
    nodes = [
        Coloring('two', [[0, 0, 1], [0, 0, 1], [1, 1, 0]], 2),
        Coloring('other', [[0, 1, 0], [1, 0, 1], [0, 1, 0]], 3),
        Coloring('undefined', [[0, 1, 1], [1, 0, 1], [1, 1, 0]], 3),
    ]


class FixedReadoutFigureTests(unittest.TestCase):
    def test_fixed_defaults_and_legacy_path_are_distinct(self):
        with patch('sys.argv', ['fig5', '--fixed-readout-sweep']), \
                patch.object(figure, 'run_fixed_readout_sweep') as run:
            figure.main()
        run.assert_called_once_with([float(x) for x in range(2, 16)], 50, 2000,
                                    readout_scale=1.0)

    def test_no_activity_penalty_is_forwarded_to_fixed_sweep(self):
        with patch('sys.argv', ['fig5', '--fixed-readout-sweep', '--lnorms',
                                '0', '2', '--seeds', '3']), \
                patch.object(figure, 'run_fixed_readout_sweep') as run:
            figure.main()
        run.assert_called_once_with([0.0, 2.0], 3, 2000, readout_scale=1.0)

    def test_scale_is_forwarded_and_invalid_values_rejected(self):
        with patch('sys.argv', ['fig5', '--fixed-readout-sweep', '--lnorms',
                                '0', '2', '--seeds', '3', '--readout-scale', '4']), \
                patch.object(figure, 'run_fixed_readout_sweep') as run:
            figure.main()
        run.assert_called_once_with([0.0, 2.0], 3, 2000, readout_scale=4.0)
        for options in (['--fixed-readout-sweep', '--readout-scale', '0'],
                        ['--fixed-readout-sweep', '--readout-scale', 'nan'],
                        ['--readout-scale', '4']):
            with self.subTest(options=options), \
                    patch('sys.argv', ['fig5', *options]), \
                    patch.object(figure, 'run_fixed_readout_sweep') as run:
                with self.assertRaises(SystemExit):
                    figure.main()
                run.assert_not_called()

    def test_color_proportions_include_all_runs_and_unique_penalty_labels(self):
        rows = pd.DataFrame([dict(run_key='zero', lnorm=0.0),
                             dict(run_key='one', lnorm=10.0),
                             dict(run_key='two', lnorm=10.0),
                             dict(run_key='three', lnorm=11.0)])
        matrix = np.array([[1, 0.5, 0.9], [0.5, 1, 0.9], [0.9, 0.9, 1]])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            figure.save_ncolors_stackedbar(
                rows, {key: matrix for key in rows.run_key}, Lattice(),
                lnorms=[0.0, 10.0, 11.0, 12.0], output_dir=output)
            winners = pd.read_csv(output / 'coloring_winners.csv')
            proportions = pd.read_csv(output / 'coloring_proportions.csv')
            self.assertEqual(len(winners), 4)
            self.assertEqual(set(winners['cc']), {'two'})
            self.assertEqual(proportions['count'].sum(), 4)
            self.assertTrue((output / 'ncolors_stackedbar.png').exists())

    def test_group_plots_every_completed_configuration_and_reuses_sweep(self):
        from scripts import constrained_penalty_sweep as sweep
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            matrices = root / 'runs'
            def fake_run(root_arg, lnorm, seed, epochs, *, readout_norm, readout_scale):
                self.assertEqual(readout_norm, 'fixed')
                self.assertEqual(readout_scale, 4.0)
                run_id = f'{lnorm}-{seed}'
                directory = matrices / run_id
                directory.mkdir(parents=True)
                np.savez(directory / 'decoding_heatmap.npz',
                         matrix=np.array([[1, 0.5, 0.9], [0.5, 1, 0.9],
                                          [0.9, 0.9, 1]]))
                return dict(run_id=run_id, lnorm=lnorm, seed=seed,
                            train_accuracy=0.8, test_accuracy=0.7)
            with patch.object(sweep, 'OUTPUT_DIR', root), \
                    patch.object(sweep, 'run_one', side_effect=fake_run), \
                    patch.object(figure, 'get_task_lattice', return_value=Lattice()):
                directory = figure.run_fixed_readout_sweep(
                    [2.0, 3.0], 2, 1, readout_scale=4.0)
            winners = pd.read_csv(directory / 'coloring_winners.csv')
            self.assertEqual(len(winners), 4)
            self.assertEqual(winners.groupby('lnorm').size().tolist(), [2, 2])
            expected_id = sweep.configuration_id(dict(
                lnorms=[2.0, 3.0], seeds=[1, 2],
                settings=dict(sweep.SETTINGS, epochs=1, fixed_readout_norm=True,
                              readout_scale=4.0), cache_version=1))
            self.assertEqual(directory.name, expected_id)

    def test_curated_plot_excludes_only_identified_runs_and_renormalizes(self):
        rows = [dict(run_key=f'keep-{lnorm}-{color}', lnorm=lnorm, nstates=color, auc=0.9)
                for lnorm in range(1, 8) for color in (5, 6)]
        rows += [dict(run_key=key, lnorm=lnorm, nstates=15, auc=1.0)
                 for key, lnorm in figure.CURATED_EXCLUSIONS.items()]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'coloring_winners.csv'
            pd.DataFrame(rows).to_csv(source, index=False)
            before = source.read_bytes()
            path = figure.save_curated_stackedbar(root)
            self.assertTrue(path.exists())
            self.assertEqual(source.read_bytes(), before)
            winners = pd.read_csv(root / 'coloring_winners_curated.csv')
            self.assertEqual(len(winners), 12)
            self.assertEqual(set(winners['lnorm']), set(range(1, 7)))
            self.assertFalse(winners['run_key'].isin(figure.CURATED_EXCLUSIONS).any())
            proportions = pd.read_csv(root / 'coloring_proportions_curated.csv')
            np.testing.assert_allclose(proportions['proportion'], 0.5)
            np.testing.assert_allclose(proportions.groupby('lnorm')['proportion'].sum(), 1)

    def test_curated_plot_requires_both_exact_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame([dict(run_key='other', lnorm=1, nstates=5, auc=1)]).to_csv(
                root / 'coloring_winners.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'Expected 15-color winner'):
                figure.save_curated_stackedbar(root)

    def test_curated_cli_never_runs_training(self):
        with patch('sys.argv', ['fig5', '--curated-summary', '/tmp/summary']), \
                patch.object(figure, 'save_curated_stackedbar') as plot, \
                patch.object(figure, 'run_fixed_readout_sweep') as train:
            figure.main()
        plot.assert_called_once_with(Path('/tmp/summary'))
        train.assert_not_called()

    def test_cached_accuracy_boxplots_use_all_runs_for_both_splits(self):
        rows = [dict(run_id=f'{lnorm}-{seed}', lnorm=float(lnorm), seed=seed,
                     train_accuracy=0.5 + lnorm * 0.05 + seed * 0.01,
                     test_accuracy=0.1)
                for lnorm in range(1, 8) for seed in (1, 2)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'runs.csv'
            pd.DataFrame(rows).to_csv(source, index=False)
            before = source.read_bytes()
            captured = []
            def capture(fig, *args, **kwargs):
                captured.append(fig)
            with patch.object(Figure, 'savefig', capture), \
                    patch.object(figure.sns, 'boxplot', wraps=figure.sns.boxplot) as boxplot:
                paths = figure.save_cached_accuracy_boxplot(root)
            self.assertEqual(paths, [root / 'train_accuracy_boxplot.png',
                                     root / 'test_accuracy_boxplot.png'])
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(captured), 2)
            self.assertEqual(boxplot.call_count, 2)
            for fig, call, column in zip(captured, boxplot.call_args_list,
                                         ('train_accuracy', 'test_accuracy')):
                data = call.kwargs['data']
                self.assertEqual(len(data), 12)
                self.assertEqual(set(data['lnorm']), {str(value) for value in range(1, 7)})
                self.assertEqual(call.kwargs['order'], [str(value) for value in range(1, 7)])
                self.assertEqual(call.kwargs['y'], column)
                self.assertEqual(call.kwargs['color'], 'lightgray')
                np.testing.assert_array_equal(fig.get_size_inches(), [5, 5])
                ax = fig.axes[0]
                self.assertEqual(len(ax.patches), 6)
                self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()],
                                 [str(value) for value in range(1, 7)])
                self.assertEqual(ax.get_xlabel(), 'Lnorm penalty')
                self.assertEqual(ax.get_ylabel(), 'Accuracy')

    def test_cached_accuracy_boxplot_requires_valid_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [dict(run_id=str(lnorm), lnorm=lnorm, seed=1, train_accuracy=0.8,
                         test_accuracy=0.7)
                    for lnorm in range(1, 7)]
            for modified in (rows[:-1], [*rows, rows[0]],
                             [dict(row, train_accuracy=float('nan')) for row in rows],
                             [dict(row, test_accuracy=1.1) for row in rows]):
                pd.DataFrame(modified).to_csv(root / 'runs.csv', index=False)
                with self.assertRaises(ValueError):
                    figure.save_cached_accuracy_boxplot(root)

    def test_accuracy_cli_reads_summary_without_training(self):
        with patch('sys.argv', ['fig5', '--accuracy-summary', '/tmp/summary']), \
                patch.object(figure, 'save_cached_accuracy_boxplot') as plot, \
                patch.object(figure, 'run_fixed_readout_sweep') as train:
            figure.main()
        plot.assert_called_once_with(Path('/tmp/summary'))
        train.assert_not_called()

    def test_histograms_compare_three_colorings_for_each_network(self):
        class HistogramLattice:
            nodes = [
                Coloring(1, [[0, 0, 1], [0, 0, 1], [1, 1, 0]], 4),
                Coloring(2, [[0, 0, 1], [0, 0, 1], [1, 1, 0]], 2),
                Coloring(11, [[0, 1, 0], [1, 0, 1], [0, 1, 0]], 3),
            ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = root / 'summary'
            summary.mkdir()
            rows = []
            for lnorm in range(1, 7):
                for seed in (1, 2):
                    run_id = f'run-{lnorm}-{seed}'
                    rows.append(dict(run_id=run_id, seed=seed, lnorm=lnorm))
                    directory = root / 'runs' / run_id
                    directory.mkdir(parents=True)
                    np.savez(directory / 'decoding_heatmap.npz',
                             matrix=[[1, .5, .9], [.5, 1, .9], [.9, .9, 1]],
                             metadata=json.dumps(dict(task='limitedstack', seed=seed,
                                                      lnorm=lnorm)))
            pd.DataFrame(rows).to_csv(summary / 'runs.csv', index=False)
            winners = [dict(run_key=f'keep-{lnorm}', lnorm=lnorm, nstates=size, auc=1)
                       for lnorm, size in enumerate((2, 3, 4, 2, 3, 4), start=1)]
            winners += [dict(run_key=key, lnorm=lnorm, nstates=15, auc=1)
                        for key, lnorm in figure.CURATED_EXCLUSIONS.items()]
            pd.DataFrame(winners).to_csv(summary / 'coloring_winners.csv', index=False)
            captured = []
            def capture(fig, *args, **kwargs):
                captured.append(fig)
            with patch.object(Figure, 'savefig', capture):
                figure.save_auc_histograms(summary, root / 'runs', HistogramLattice())
            fig = captured[0]
            np.testing.assert_array_equal(fig.get_size_inches(), [5, 5])
            self.assertTrue(all(not ax.get_ylabel() for ax in fig.axes))
            self.assertTrue(all(not label.get_visible() for ax in fig.axes
                                for label in ax.get_yticklabels()))
            self.assertTrue(all(ax.get_legend() is None for ax in fig.axes))
            self.assertEqual(fig.axes[-1].get_xlim(), (0.5, 1.0))
            palette = figure.coloring_colors((2, 3, 4))
            nbins = len(fig.axes[0].patches) // len(figure.HISTOGRAM_COLORINGS)
            for index, nstates in enumerate((4, 2, 3)):
                np.testing.assert_allclose(fig.axes[0].patches[index * nbins].get_facecolor()[:3],
                                           palette[nstates][:3])
                self.assertEqual(sum(patch.get_height() for patch in
                                     fig.axes[0].patches[index * nbins:(index + 1) * nbins]), 2)
            scores = pd.read_csv(summary / 'auc_histogram_scores.csv')
            self.assertEqual(len(scores), 36)
            self.assertEqual(scores.groupby(['lnorm', 'cc']).size().tolist(), [2] * 18)
            np.testing.assert_allclose(scores.loc[scores['cc'] == 1, 'auc'], 1.0)
            np.testing.assert_allclose(scores.loc[scores['cc'] == 2, 'auc'], 1.0)
            np.testing.assert_allclose(scores.loc[scores['cc'] == 11, 'auc'], 0.25)

    def test_histogram_cli_reads_caches_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'limitedstack_Ldiag6_lattice.pkl').touch()
            with patch('sys.argv', ['fig5', '--auc-histogram-summary', '/tmp/summary']), \
                    patch.object(figure, 'DIAGRAMS_DIR', root), \
                    patch.object(figure, 'get_task_lattice', return_value=Lattice()), \
                    patch.object(figure, 'save_auc_histograms') as plot, \
                    patch.object(figure, 'run_fixed_readout_sweep') as train:
                figure.main()
            from scripts import constrained_penalty_sweep as sweep
            plot.assert_called_once_with(Path('/tmp/summary'), sweep.OUTPUT_DIR / 'runs',
                                         unittest.mock.ANY)
            train.assert_not_called()

if __name__ == '__main__':
    unittest.main()
