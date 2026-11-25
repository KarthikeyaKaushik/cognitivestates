import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts import constrained_stack_behavior as behavior
from src.training import ConstrainedRNN, readout_weight_norm, train_model


class SingleBatch:
    def batch_generator(self, train=True):
        inputs = torch.tensor([[[1., 0.], [0., 1.]]])
        targets = torch.tensor([[[1., 0.], [0., 1.]]])
        yield inputs, targets, torch.empty(0)


class ReadoutNormTests(unittest.TestCase):
    def test_log_tracks_current_effective_weight_after_optimizer_update(self):
        model = ConstrainedRNN(2, 3, 2, seed=1, lnorm=0.2)
        initial_norm = readout_weight_norm(model)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _, history = train_model(model, SingleBatch(), 0.1, 3,
                                     loss_tracking_epochs=2)
        self.assertEqual(history['epochs'], [0, 2])
        self.assertEqual(len(history['readout_weight_norm']), 2)
        self.assertNotEqual(initial_norm, history['readout_weight_norm'][-1])
        with torch.no_grad():
            model(torch.tensor([[[1., 0.]]]))
            actual = torch.linalg.matrix_norm(model.fc.weight).item()
        self.assertAlmostEqual(history['readout_weight_norm'][-1], actual, places=6)
        self.assertIn(f'Readout weight norm : {actual:.4f}', output.getvalue())

    def test_fixed_readout_preserves_each_scale_but_learns_directions(self):
        seed = 7
        initial = ConstrainedRNN(2, 3, 2, seed=seed, lnorm=0.2)
        scales = initial.fc.weight_g.detach().clone()
        directions = initial.fc.weight_v.detach().clone()
        initial_bias = initial.fc.bias.detach().clone()
        settings = dict(behavior.SETTINGS, hidden_dim=3, epochs=3,
                        learning_rate=0.1, fixed_readout_norm=True)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(behavior, 'get_task_diagram'), \
                patch.object(behavior, 'DataGenerator', return_value=SingleBatch()), \
                patch.dict(behavior.TASKS, {behavior.TASK: {'generate': lambda _: [1]}}):
            directory = Path(temp)
            model, metadata = behavior.load_or_train_network(
                directory, seed, ['a', 'b'], ['n', 'x'], settings=settings)
            self.assertTrue(torch.equal(scales, model.fc.weight_g))
            self.assertFalse(torch.equal(directions, model.fc.weight_v))
            self.assertFalse(torch.equal(initial_bias, model.fc.bias))
            self.assertFalse(model.fc.weight_g.requires_grad)
            checkpoint = torch.load(directory / 'network.pt', weights_only=True)
            self.assertAlmostEqual(checkpoint['initial_readout_weight_norm'],
                                   readout_weight_norm(model), places=6)
            self.assertEqual(len(checkpoint['training_history']['readout_weight_norm']), 1)
            reloaded, restored_metadata = behavior.load_or_train_network(
                directory, seed, ['a', 'b'], ['n', 'x'], settings=settings)
            self.assertEqual(metadata, restored_metadata)
            self.assertFalse(reloaded.fc.weight_g.requires_grad)
            self.assertTrue(torch.equal(scales, reloaded.fc.weight_g))
            with self.assertRaises(ValueError):
                behavior.load_or_train_network(
                    directory, seed, ['a', 'b'], ['n', 'x'],
                    settings={key: value for key, value in settings.items()
                              if key != 'fixed_readout_norm'})

    def test_no_activity_penalty_keeps_fixed_readout_trainable(self):
        seed = 7
        initial = ConstrainedRNN(2, 3, 2, seed=seed, lnorm=0)
        scales = initial.fc.weight_g.detach().clone()
        directions = initial.fc.weight_v.detach().clone()
        settings = dict(behavior.SETTINGS, hidden_dim=3, epochs=3, lnorm=0.0,
                        learning_rate=0.1, fixed_readout_norm=True)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(behavior, 'get_task_diagram'), \
                patch.object(behavior, 'DataGenerator', return_value=SingleBatch()), \
                patch.dict(behavior.TASKS, {behavior.TASK: {'generate': lambda _: [1]}}):
            model, _ = behavior.load_or_train_network(
                Path(temp), seed, ['a', 'b'], ['n', 'x'], settings=settings)
        self.assertEqual(model.multiplier, 0)
        self.assertTrue(torch.equal(scales, model.fc.weight_g))
        self.assertFalse(torch.equal(directions, model.fc.weight_v))

    def test_scaled_fixed_readout_matches_seeded_initialization_and_reloads(self):
        seed = 7
        initial = ConstrainedRNN(2, 3, 2, seed=seed, lnorm=0)
        settings = dict(behavior.SETTINGS, hidden_dim=3, epochs=3, lnorm=0.0,
                        learning_rate=0.1, fixed_readout_norm=True,
                        readout_scale=4.0)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(behavior, 'get_task_diagram'), \
                patch.object(behavior, 'DataGenerator', return_value=SingleBatch()), \
                patch.dict(behavior.TASKS, {behavior.TASK: {'generate': lambda _: [1]}}):
            directory = Path(temp)
            model, _ = behavior.load_or_train_network(
                directory, seed, ['a', 'b'], ['n', 'x'], settings=settings)
            torch.testing.assert_close(model.fc.weight_g, initial.fc.weight_g * 4)
            self.assertFalse(model.fc.weight_g.requires_grad)
            self.assertFalse(torch.equal(initial.fc.weight_v, model.fc.weight_v))
            loaded, _ = behavior.load_or_train_network(
                directory, seed, ['a', 'b'], ['n', 'x'], settings=settings)
            torch.testing.assert_close(loaded.fc.weight_g, initial.fc.weight_g * 4)
            with self.assertRaises(ValueError):
                behavior.load_or_train_network(
                    directory, seed, ['a', 'b'], ['n', 'x'],
                    settings=dict(settings, readout_scale=2.0))


if __name__ == '__main__':
    unittest.main()
