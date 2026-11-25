import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts import constrained_stack_behavior as behavior
from src.diagrams import build_vocab
from src.limitedstack import generate_sequences
from src.training import ConstrainedRNN


class ConstantNetwork(torch.nn.Module):
    def forward(self, inputs):
        logits = torch.zeros(*inputs.shape[:2], 4)
        logits[..., 0] = 1
        return logits, torch.zeros(*inputs.shape[:2], 5)


class BehaviorTests(unittest.TestCase):
    def test_default_runs_five_seeds_with_identical_settings(self):
        with patch('sys.argv', ['constrained_stack_behavior.py']), \
                patch.object(behavior, 'run_network') as run:
            behavior.main()
        self.assertEqual(run.call_args_list, [
            call(seed, length=None, retrain=False) for seed in (1, 2, 3, 4, 5)])

    def test_explicit_seeds_share_length_and_retrain_options(self):
        with patch('sys.argv', ['constrained_stack_behavior.py', '--seeds', '2', '4', '6',
                                '--length', '9', '--retrain']), \
                patch.object(behavior, 'run_network') as run:
            behavior.main()
        self.assertEqual(run.call_args_list, [
            call(seed, length=9, retrain=True) for seed in (2, 4, 6)])

    def test_sweep_cli_routes_each_selected_run_without_training(self):
        argv = ['constrained_stack_behavior.py', '--sweep-summary', '/tmp/summary',
                '--run', '2', '18', '--run', '2', '31', '--run', '3', '48']
        with patch('sys.argv', argv), \
                patch.object(behavior, 'run_sweep_network') as run, \
                patch.object(behavior, 'run_network') as legacy:
            behavior.main()
        self.assertEqual(run.call_args_list, [
            call(Path('/tmp/summary'), lnorm, seed, length=None)
            for lnorm, seed in ((2, 18), (2, 31), (3, 48))])
        legacy.assert_not_called()

    def test_sweep_cli_rejects_conflicting_and_invalid_arguments(self):
        base = ['constrained_stack_behavior.py', '--sweep-summary', '/tmp/summary',
                '--run', '2', '18']
        for extra in (['--retrain'], ['--seeds', '1', '2', '3', '4', '5'],
                      ['--run', '2', '18'], ['--run', 'nan', '3'],
                      ['--length', '0']):
            with self.subTest(extra=extra), patch('sys.argv', base + extra), \
                    patch.object(behavior, 'run_sweep_network') as run:
                with self.assertRaises(SystemExit):
                    behavior.main()
                run.assert_not_called()

    def test_sweep_checkpoint_inference_and_cache_validation(self):
        from scripts import constrained_penalty_sweep as sweep
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = root / 'summary'
            summary.mkdir()
            run_id = 'lnorm2_seed18_example'
            run_dir = root / 'sweep' / 'runs' / run_id
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / 'network.pt'
            checkpoint.write_bytes(b'checkpoint')
            X, Y = build_vocab(generate_sequences(6))
            config = dict(task='limitedstack', seed=18,
                          **dict(behavior.SETTINGS, lnorm=2, epochs=2000,
                                 fixed_readout_norm=True, readout_scale=4.0),
                          input_tokens=sorted(X), output_tokens=sorted(Y),
                          split_seed=0, train_fraction=0.8, cache_version=1)
            metrics = dict(run_id=run_id, readout_norm='fixed', readout_scale=4.0)
            (run_dir / 'metrics.json').write_text(json.dumps(
                dict(configuration=config, metrics=metrics)))
            with (summary / 'runs.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['run_id', 'seed', 'lnorm',
                                                       'epochs', 'readout_norm',
                                                       'readout_scale'])
                writer.writeheader()
                writer.writerow(dict(run_id=run_id, seed=18, lnorm=2, epochs=2000,
                                     readout_norm='fixed', readout_scale=4))
            model = ConstantNetwork()
            with patch.object(sweep, 'OUTPUT_DIR', root / 'sweep'), \
                    patch.object(behavior, 'RESULTS_DIR', root), \
                    patch.object(behavior, 'load_checkpoint_network', return_value=model), \
                    patch.object(behavior, 'load_or_train_network') as train, \
                    patch.object(behavior, 'collect_behavior', return_value=[]) as collect, \
                    patch.object(behavior, 'save_analysis') as save, \
                    patch.object(behavior, 'save_traversal_analysis') as replay:
                directory = behavior.run_sweep_network(summary, 2, 18)
                self.assertEqual(directory, root / 'network_behavior' / f'{run_id}_Ldiag6')
                self.assertEqual(len(collect.call_args.args[1]), 729)
                self.assertEqual(save.call_args.kwargs['diagram_length'], 6)
                metadata = save.call_args.args[1]
                self.assertEqual(metadata['source_run_id'], run_id)
                directory.mkdir(parents=True)
                (directory / 'observations.json').write_text(json.dumps(
                    dict(diagram_L=6, metadata=metadata, observations=[])))
                (directory / 'diagram.json').write_text(json.dumps({'states': []}))
                behavior.run_sweep_network(summary, 2, 18)
                replay.assert_called_once_with(directory, {'states': []}, [])
                self.assertEqual(collect.call_count, 1)
                checkpoint.write_bytes(b'changed checkpoint')
                with self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
                    behavior.run_sweep_network(summary, 2, 18)
                train.assert_not_called()

    def test_five_cached_diagrams_are_replayed_without_training_or_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            machines = []
            for seed in (1, 2, 3, 4, 5):
                settings = behavior.SETTINGS
                key = behavior.constrained_run_key(
                    'limitedstack', seed, settings['lnorm'], settings['train_L'],
                    settings['hidden_dim'], settings['epochs'])
                directory = root / 'network_behavior' / key
                directory.mkdir(parents=True)
                machine = dict(start_state=0, states=[0], transitions=[])
                machines.append((machine, directory))
                (directory / 'diagram.json').write_text(json.dumps(machine))
                (directory / 'observations.json').write_text(json.dumps(dict(
                    metadata=dict(task='limitedstack', seed=seed, **settings),
                    diagram_L=behavior.TASKS['limitedstack']['diagram_L'],
                    observations=[])))
            with patch('sys.argv', ['constrained_stack_behavior.py']), \
                    patch.object(behavior, 'RESULTS_DIR', root), \
                    patch.object(behavior, 'save_decoding_heatmap'), \
                    patch.object(behavior, 'save_traversal_analysis') as replay, \
                    patch.object(behavior, 'load_or_train_network') as network, \
                    patch.object(behavior, 'infer_behavior_diagram') as infer:
                behavior.main()
            self.assertEqual(replay.call_args_list,
                             [call(directory, machine, []) for machine, directory in machines])
            network.assert_not_called()
            infer.assert_not_called()

    def test_seed_validation_precedes_any_training(self):
        for options in (['--seeds', '1', '1'], ['--seeds', '0'],
                        ['--seed', '1', '--seeds', '2', '3']):
            with self.subTest(options=options), \
                    patch('sys.argv', ['constrained_stack_behavior.py', *options]), \
                    patch.object(behavior, 'run_network') as run:
                with self.assertRaises(SystemExit):
                    behavior.main()
                run.assert_not_called()

    def test_length_override_routes_all_length_nine_sequences_to_inference(self):
        model = ConstantNetwork()
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(behavior, 'RESULTS_DIR', Path(temp)), \
                patch('sys.argv', ['constrained_stack_behavior.py', '--seed', '1',
                                   '--length', '9']), \
                patch.object(behavior, 'load_or_train_network',
                             return_value=(model, {})) as network, \
                patch.object(behavior, 'save_decoding_heatmap'), \
                patch.object(behavior, 'collect_behavior', return_value=[]) as collect, \
                patch.object(behavior, 'save_analysis') as save:
            behavior.main()
            settings = network.call_args.kwargs['settings']
            self.assertEqual(settings, dict(behavior.SETTINGS, train_L=9))
            directory = network.call_args.args[0]
            self.assertIn('_L9_', directory.name)
            self.assertTrue(directory.name.endswith('_Ldiag9'))
            pairs = collect.call_args.args[1]
            self.assertEqual(len(pairs), 19_683)
            self.assertTrue(all(len(inp) == len(out) == 9 for inp, out in pairs))
            save.assert_called_once_with(directory, {}, [], diagram_length=9)

    def test_all_729_inputs_and_all_timesteps_are_recorded(self):
        pairs = generate_sequences(6)
        X, Y = build_vocab(pairs)
        observations = behavior.collect_behavior(
            ConstantNetwork(), pairs, sorted(X), sorted(Y), batch_size=17)
        self.assertEqual(len(observations), 729)
        self.assertEqual([tuple(row['inputs']) for row in observations],
                         [inp for inp, _ in pairs])
        self.assertEqual([tuple(row['target_outputs']) for row in observations],
                         [out for _, out in pairs])
        self.assertTrue(all(row['network_outputs'] == ['e'] * 6
                            for row in observations))
        self.assertEqual(sum(len(row['network_outputs']) for row in observations),
                         4374)

    def test_diagram_uses_predictions_not_task_targets_and_is_saved(self):
        pairs = generate_sequences(2)
        X, Y = build_vocab(generate_sequences(6))
        observations = behavior.collect_behavior(
            ConstantNetwork(), pairs, sorted(X), sorted(Y))
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(behavior, 'render_diagram') as render, \
                    patch.object(behavior, 'save_figure'):
                cd = behavior.save_analysis(directory, {}, observations)
            self.assertEqual(len(cd.states), 1)
            self.assertEqual(set(cd.outputtrans.values()), {'e'})
            for inp, _ in pairs:
                self.assertEqual(cd.apply_transitions(cd.start_state, inp)[1],
                                 ('e', 'e'))
            dataset = json.loads((directory / 'observations.json').read_text())
            machine = json.loads((directory / 'diagram.json').read_text())
            self.assertEqual(render.call_args_list, [
                call(machine, directory),
                call(machine, directory, selected_states=(0,)),
            ])
            self.assertEqual(dataset['observations'], observations)
            self.assertLess(dataset['summary']['timestep_accuracy'], 1)
            self.assertEqual(machine['verified_sequences'], len(pairs))
            self.assertEqual(machine['unobserved_transitions'], [])
            self.assertTrue((directory / 'diagram.pkl').exists())
            self.assertFalse((directory / 'diagram.tex').exists())
            with (directory / 'transition_counts.csv').open() as f:
                counts = list(csv.DictReader(f))
            self.assertEqual([int(edge['count']) for edge in counts], [6, 6, 6])

    def test_cached_main_counts_parallel_arrows_without_network_or_inference(self):
        config = behavior.SETTINGS
        key = behavior.constrained_run_key('limitedstack', 1, config['lnorm'],
                                            config['train_L'], config['hidden_dim'],
                                            config['epochs'])
        machine = dict(start_state=0, states=[0, 1, 2], transitions=[
            dict(state=0, input='push(x)', output='n', next_state=1),
            dict(state=0, input='push(y)', output='n', next_state=1),
            dict(state=1, input='pop()', output='x', next_state=1),
            dict(state=1, input='push(x)', output='e', next_state=0),
        ])
        observations = [
            dict(inputs=['push(x)', 'pop()', 'pop()'], network_outputs=['n', 'x', 'x']),
            dict(inputs=['push(y)', 'pop()'], network_outputs=['n', 'x']),
        ]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'network_behavior' / key
            directory.mkdir(parents=True)
            (directory / 'diagram.json').write_text(json.dumps(machine))
            dataset = dict(metadata=dict(task='limitedstack', seed=1, **config),
                           diagram_L=behavior.TASKS['limitedstack']['diagram_L'],
                           observations=observations)
            (directory / 'observations.json').write_text(json.dumps(dataset))
            def check_plot(name):
                self.assertEqual(name, f'state_visits_cumulative_{key}')
                ax = plt.gca()
                # With <= 6 states, all labels collapse into a single ellipsis.
                self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()],
                                 ['...'])
                self.assertEqual([p.get_height() for p in ax.patches], [5 / 7, 1.0, 1.0])
                self.assertEqual([p.get_width() for p in ax.patches], [5 / 7, 2 / 7, 0.0])

            with patch.object(behavior, 'RESULTS_DIR', Path(temp)), \
                    patch.object(behavior, 'save_decoding_heatmap'), \
                    patch('sys.argv', ['constrained_stack_behavior.py', '--seed', '1']), \
                    patch.object(behavior, 'load_or_train_network') as network, \
                    patch.object(behavior, 'infer_behavior_diagram') as infer, \
                    patch.object(behavior, 'render_diagram') as render, \
                    patch.object(behavior, 'save_figure', side_effect=check_plot) as save:
                behavior.main()
            save.assert_called_once()
            network.assert_not_called()
            infer.assert_not_called()
            self.assertEqual(render.call_args_list, [
                call(machine, directory),
                call(machine, directory, selected_states=(1, 0, 2)),
            ])
            with (directory / 'transition_counts.csv').open() as f:
                self.assertEqual([int(edge['count']) for edge in csv.DictReader(f)],
                                 [1, 1, 3, 0])
            with (directory / 'state_visit_counts.csv').open() as f:
                self.assertEqual(list(csv.reader(f)),
                                 [['state', 'visits'], ['0', '2'], ['1', '5'], ['2', '0']])
            self.assertEqual(json.loads((directory / 'diagram.json').read_text()), machine)
        observations[1]['network_outputs'][0] = 'e'
        with self.assertRaisesRegex(ValueError, 'Sequence 2, step 1'):
            behavior.count_transitions(machine, observations)

    def test_subset_plot_selects_top_five_for_each_network(self):
        machine = dict(states=list(range(21)))
        directory = Path('unused')
        visits = {state: 1 for state in machine['states']}
        visits.update({3: 10, 8: 9, 15: 8, 20: 7, 4: 6, 5: 6})
        with patch.object(behavior, 'render_diagram') as render:
            behavior.plot_subset_diagram(machine, directory, visits)
        render.assert_called_once_with(machine, directory, selected_states=(3, 8, 15, 20, 4))

    def test_cached_heatmap_replots_without_loading_or_training_network(self):
        metadata = dict(task='limitedstack', seed=1, **behavior.SETTINGS)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            np.savez_compressed(directory / 'decoding_heatmap.npz',
                                matrix=np.array([[1.0, 0.75], [0.75, 1.0]]),
                                labels=['e', 'x'], metadata=json.dumps(metadata))
            with patch.object(behavior, 'load_or_train_network') as network, \
                    patch.object(behavior, 'get_pairwise_decodings') as decode:
                behavior.save_decoding_heatmap(directory, metadata)
            network.assert_not_called()
            decode.assert_not_called()
            self.assertGreater((directory / 'decoding_heatmap.png').stat().st_size, 1000)

    def test_missing_heatmap_without_weights_does_not_train(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(behavior, 'load_or_train_network') as network:
            with self.assertRaises(FileNotFoundError):
                behavior.save_decoding_heatmap(Path(temp), {})
            network.assert_not_called()

    def test_nonfinite_predictions_fail_explicitly(self):
        model = ConstantNetwork()
        with patch.object(model, 'forward',
                          return_value=(torch.full((1, 1, 4), float('nan')), None)):
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                behavior.collect_behavior(model, [(('a',), ('e',))],
                                          ['a'], ['e', 'n', 'x', 'y'])

    def test_unverified_diagram_is_not_saved_but_observations_are(self):
        observations = [dict(inputs=['a'], target_outputs=['e'],
                             network_outputs=['n'])]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(behavior, 'veelenturf') as infer:
                infer.return_value.apply_transitions.return_value = (0, ('e',))
                with self.assertRaisesRegex(RuntimeError, 'disagrees'):
                    behavior.save_analysis(directory, {}, observations)
            self.assertTrue((directory / 'observations.json').exists())
            self.assertFalse((directory / 'diagram.pkl').exists())

    def test_checkpoint_roundtrip_reuses_weights_and_rejects_mismatches(self):
        inputs, outputs = ['pop()', 'push(x)', 'push(y)'], ['e', 'n', 'x', 'y']
        model = ConstrainedRNN(3, 5, 4, seed=1, lnorm=0.6)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(behavior, 'get_task_diagram') as diagram, \
                    patch.object(behavior, 'DataGenerator'), \
                    patch.object(behavior, 'train_model',
                                 return_value=(model, {'loss': [1.0]})) as train:
                trained, _ = behavior.load_or_train_network(
                    directory, 1, inputs, outputs)
                loaded, _ = behavior.load_or_train_network(
                    directory, 1, inputs, outputs)
                self.assertEqual(train.call_count, 1)
                self.assertEqual(diagram.call_count, 1)
                for key, value in trained.state_dict().items():
                    torch.testing.assert_close(value, loaded.state_dict()[key])
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    behavior.load_or_train_network(directory, 2, inputs, outputs)
                behavior.load_or_train_network(directory, 1, inputs, outputs,
                                               retrain=True)
                self.assertEqual(train.call_count, 2)


if __name__ == '__main__':
    unittest.main()
