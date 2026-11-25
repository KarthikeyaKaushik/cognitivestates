"""Reproduce the legacy infinite discrete/continuous state visualizations.

Usage:
  python scripts/fig4_infinite.py
  python scripts/fig4_infinite.py --tasks majority
  python scripts/fig4_infinite.py --tasks rescorla_wagner --recompute

Defaults follow the commands in scripts_old/run_majority_cfg.py and
scripts_old/run_rescorla_wagner.py. Majority trains on bounded count walks,
then visualizes a three-times-wider count range with 1D LDA (100 samples per
state, requiring strictly more than 100 available). RW visualizes separately
fitted 2D PLS projections before/after MSE training.

These are legacy in-sample visualizations, NOT held-out decoding evaluations:
both use the generators' 80% training split for projection fitting/plotting,
including the new wider-range majority dataset. RW drops the last timestep;
majority retains all timesteps. Legacy lnorm values are recorded but neither
loss applies an activity penalty. Seed 0 retains the legacy ambient torch
seed of 42; majority sampling uses Python seed 42.

Each configuration caches projection/training CSVs under results/infinite/.
Re-running only replots; --recompute explicitly regenerates the analysis.
Outputs: figures/fig4_infinite_{majority_scatter,majority_ridges,rw_strip,
rw_scatter}.png. Nondefault configurations have a distinguishing suffix.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from scripts.common import save_figure
from src.majority_cfg import generate_sequences as majority_sequences
from src.paths import RESULTS_DIR
from src.projections import get_lda_projection
from src.rescorlawagner import generate_sequences as rw_sequences
from src.training import ConstrainedRNN


CONFIGS = {
    'majority': dict(length=300, samples=500, max_states=4, hidden_dim=30,
                     lnorm=1.0, epochs=2000, seed=0, learning_rate=0.0001),
    'rescorla_wagner': dict(length=100, samples=1000, alpha=0.5, hidden_dim=6,
                           lnorm=0.5, epochs=100, seed=0, learning_rate=0.001),
}
CACHE_VERSION = 1


class SequenceData:
    """Legacy one-hot inputs and 80/20 split for fixed-length task sequences."""

    def __init__(self, data, vocabulary, discrete=False):
        input_map = {token: index for index, token in enumerate(sorted(vocabulary))}
        self.samples = []
        for inputs, outputs in data:
            encoded = F.one_hot(
                torch.tensor([input_map[token] for token in inputs]),
                num_classes=len(input_map)).float()
            if discrete:
                states = torch.tensor(np.cumsum(
                    [1 if token == 1 else -1 for token in inputs]), dtype=torch.long)
                targets = (states > 0).long()
            else:
                states = torch.tensor(outputs).float()
                targets = torch.tensor(outputs).float()
            self.samples.append((encoded, targets, states))
        indices = torch.randperm(len(self.samples)).tolist()
        split = int(len(indices) * 0.8)
        self.train = [self.samples[index] for index in indices[:split]]
        self.test = [self.samples[index] for index in indices[split:]]

    def batches(self):
        indices = torch.randperm(len(self.train))
        for start in range(0, len(indices), 32):
            batch = [self.train[index] for index in indices[start:start + 32]]
            yield tuple(torch.stack(items) for items in zip(*batch))


def train_network(model, data, config, discrete):
    """Legacy Adam/CE or Adam/MSE training, without hidden-state regularization."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    history = []
    for epoch in range(config['epochs']):
        losses = []
        for inputs, targets, _ in data.batches():
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            predictions = outputs.reshape(-1, outputs.shape[-1])
            loss = (F.cross_entropy(predictions, targets.reshape(-1))
                    if discrete else F.mse_loss(predictions, targets.reshape(-1, 1)))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        average = float(np.mean(losses))
        history.append(dict(epoch=epoch, loss=average))
        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}/{config["epochs"]} - Loss: {average:.4f}')
        if discrete and average < 0.001:
            break
    return pd.DataFrame(history)


def collect_hidden(model, data, discrete):
    """Collect training-split samples in the legacy shuffled batch order."""
    model.eval()
    hidden_batches, state_batches = [], []
    with torch.no_grad():
        for inputs, _, states in data.batches():
            _, hidden = model(inputs)
            if not discrete:
                hidden, states = hidden[:, :-1], states[:, :-1]
            hidden_batches.append(hidden.reshape(-1, hidden.shape[-1]))
            state_batches.append(states.reshape(-1))
    return torch.cat(hidden_batches), torch.cat(state_batches)


def pls_projection(hidden, states):
    """Legacy standardized X/y followed by PLSRegression's own default scaling."""
    predictors = StandardScaler().fit_transform(hidden.numpy())
    targets = StandardScaler().fit_transform(states.numpy().reshape(-1, 1))
    projection = PLSRegression(n_components=2).fit_transform(predictors, targets)[0]
    return pd.DataFrame(dict(z_1=projection[:, 0], z_2=projection[:, 1],
                             y=states.numpy()))


def compute_analysis(task, config):
    """Execute the original task-specific training and projection procedure."""
    random.seed(42)
    torch.manual_seed(42)
    discrete = task == 'majority'
    if discrete:
        pairs = majority_sequences(config['samples'], config['length'],
                                   config['max_states'])
    else:
        pairs = rw_sequences(config['length'], config['samples'],
                             alpha=config['alpha'], seed=config['seed'])
    vocabulary = {token for inputs, _ in pairs for token in inputs}
    model = ConstrainedRNN(
        input_dim=len(vocabulary), hidden_dim=config['hidden_dim'],
        output_dim=2 if discrete else 1, seed=config['seed'], lnorm=config['lnorm'])
    data = SequenceData(pairs, vocabulary, discrete=discrete)
    if discrete:
        train_states = {int(state) for _, _, states in data.samples for state in states}
    else:
        hidden, states = collect_hidden(model, data, discrete=False)
        untrained = pls_projection(hidden, states)
        untrained['model'] = 'untrained'

    history = train_network(model, data, config, discrete)
    if not discrete:
        hidden, states = collect_hidden(model, data, discrete=False)
        trained = pls_projection(hidden, states)
        trained['model'] = 'trained'
        return pd.concat([untrained, trained], ignore_index=True), history

    wider_pairs = majority_sequences(config['samples'] * 2, config['length'],
                                     config['max_states'] * 3)
    wider_data = SequenceData(wider_pairs, vocabulary, discrete=True)
    hidden, states = collect_hidden(model, wider_data, discrete=True)
    indices = []
    for state in states.unique():
        candidates = (states == state).nonzero(as_tuple=True)[0].tolist()
        if len(candidates) > 100:
            indices.extend(random.sample(candidates, 100))
    if not indices:
        raise ValueError('Majority LDA needs states with more than 100 samples.')
    hidden, states = hidden[indices], states[indices]
    if len(states.unique()) < 2:
        raise ValueError('Majority LDA needs at least two sampled states.')
    projection = get_lda_projection(hidden, states, d=1)
    result = pd.DataFrame(dict(
        x=projection[:, 0],
        y=[random.uniform(-1, 1) for _ in range(len(states))],
        states=states.numpy()))
    result['train_states'] = result['states'].isin(train_states)
    return result, history


def load_or_compute(task, config, recompute=False):
    """Cache only the compact plot data and training history, not hidden tensors."""
    metadata = dict(version=CACHE_VERSION, task=task, **config)
    signature = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    directory = RESULTS_DIR / 'infinite' / f'{task}_{signature}'
    projections = directory / 'projections.csv'
    history_path = directory / 'training.csv'
    metadata_path = directory / 'config.json'
    if not recompute and all(path.exists() for path in
                             (projections, history_path, metadata_path)):
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError(f'Cache metadata mismatch at {directory}.')
        print(f'Using cached {task} projections: {projections}')
        return pd.read_csv(projections), pd.read_csv(history_path), signature
    print(f'Computing {task}: {config}')
    result, history = compute_analysis(task, config)
    directory.mkdir(parents=True, exist_ok=True)
    # Metadata is the completion marker, so an interrupted write is not reused.
    if metadata_path.exists():
        metadata_path.unlink()
    result.to_csv(projections, index=False)
    history.to_csv(history_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    return result, history, signature


def plot_majority(result, suffix=''):
    """Reproduce the legacy 1D-LDA jitter scatter and stacked state densities."""
    fig, ax = plt.subplots()
    for seen, marker, legend in ((False, 'o', True), (True, 'x', False)):
        subset = result[result['train_states'] == seen]
        sns.scatterplot(data=subset, x='x', y='y', hue='states', palette='magma',
                        marker=marker, alpha=0.4, s=10, ax=ax, legend=legend)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='States')
    save_figure(f'fig4_infinite_majority_scatter{suffix}')
    plt.close(fig)

    result = result.copy()
    result['x'] = -result['x']
    states = sorted(result['states'].unique())
    colors = dict(zip(states, sns.color_palette('magma', n_colors=len(states))))
    seen_states = set(result.loc[result['train_states'], 'states'])
    minimum, maximum = result['x'].min(), result['x'].max()
    bins = np.linspace(minimum, maximum, 61)
    centers = 0.5 * (bins[:-1] + bins[1:])
    overlap, height = 0.2, 0.4
    fig, ax = plt.subplots(figsize=(
        8, max(4, len(states) * height * (1 - overlap) + height * 2)))
    for index, state in enumerate(reversed(states)):
        offset = index * height * (1 - overlap)
        density, _ = np.histogram(
            result.loc[result['states'] == state, 'x'], bins=bins, density=True)
        values = density * height * 0.9 / (density.max() + 1e-9) + offset
        alpha = 0.1 if state in seen_states else 0.75
        color = colors[state]
        ax.fill_between(centers, offset, values, color=color, alpha=alpha, linewidth=0)
        ax.plot(centers, values, color=color, linewidth=0.8, alpha=alpha)
        mean = result.loc[result['states'] == state, 'x'].mean()
        ax.scatter(mean, np.interp(mean, centers, values), s=26, color='white',
                   edgecolors=color, linewidths=0.9, zorder=5)
        ax.text(minimum - (maximum - minimum) * 0.02, offset + height * 0.15,
                str(state), ha='right', va='bottom', fontsize=14, color=color)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('1D LDA projection (a.u)', fontsize=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_figure(f'fig4_infinite_majority_ridges{suffix}')
    plt.close(fig)


def plot_rw(result, suffix=''):
    """Reproduce both legacy RW views without overwriting one with the other."""
    fig, ax = plt.subplots(figsize=(8, 4))
    # Strip jitter is cosmetic; seed it so cached replots are reproducible.
    np.random.seed(42)
    sns.stripplot(data=result, x='z_1', y='model', hue='y', palette='magma',
                  legend=True, jitter=0.2, alpha=0.5, ax=ax)
    ax.set_xlabel('PLS projection', fontsize=14)
    ax.set_ylabel(None)
    ax.set_yticks([0, 1], ['Untrained', 'Trained'], rotation=90,
                  va='center', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 0.75), title='Association', loc='upper left',
              fontsize=12, title_fontsize=12)
    save_figure(f'fig4_infinite_rw_strip{suffix}')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=result[result['model'] == 'trained'],
                    x='z_1', y='z_2', hue='y', palette='magma',
                    legend=True, alpha=0.5, s=10, ax=ax)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.legend(bbox_to_anchor=(1.05, 0.75), title='Association', loc='upper left',
              fontsize=12, title_fontsize=12)
    save_figure(f'fig4_infinite_rw_scatter{suffix}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tasks', nargs='+', choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--epochs', type=int, help='Override each selected task default.')
    parser.add_argument('--samples', type=int, help='Override each selected task default.')
    parser.add_argument('--length', type=int, help='Override each selected task default.')
    parser.add_argument('--seed', type=int, help='Override each selected task default.')
    args = parser.parse_args()
    for key in ('epochs', 'samples', 'length'):
        value = getattr(args, key)
        if value is not None and value < (2 if key != 'epochs' else 1):
            parser.error(f'--{key} must be at least {2 if key != "epochs" else 1}.')
    for task in args.tasks:
        config = CONFIGS[task].copy()
        for key in ('epochs', 'samples', 'length', 'seed'):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        result, history, signature = load_or_compute(task, config, args.recompute)
        suffix = '' if config == CONFIGS[task] else f'_{signature}'
        (plot_majority if task == 'majority' else plot_rw)(result, suffix)
        print(f'{task}: {len(result)} projected samples; '
              f'{len(history)} training epochs; final loss={history.iloc[-1]["loss"]:.6f}')


if __name__ == '__main__':
    main()
