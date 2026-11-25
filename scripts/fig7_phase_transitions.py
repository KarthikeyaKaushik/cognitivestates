"""Figure 7: consistent-coloring alignment around learning phase transitions.

Usage:
  python scripts/fig7_phase_transitions.py --seeds 5
  python scripts/fig7_phase_transitions.py
  python scripts/fig7_phase_transitions.py --recompute

Defaults reproduce scripts_old/simulations_probe.sh: limitedstack, length 7,
hidden dimension 5, lnorm 1.2, 5000 epochs, seeds 0..100, Adam lr=.001, batch
size 128, and an 80/20 split. The legacy loss-change score uses delta=100.
At t-=peak-delta and t+=peak+delta (zero-based epochs, AFTER the update),
full-hidden-vector pairwise LDA balanced TRAINING accuracies are compared with
the upper triangles of consistent-coloring matrices and 100 Bernoulli random
pair masks. This is the legacy in-sample analysis, NOT the held-out 2D SVM
analysis used for the other decoding figures; unlike the legacy code, the
pairwise accuracies are balanced, so unequal state visitation cannot inflate
them. Each family's winner is selected at t+ and that SAME candidate is
plotted at both checkpoints.

Two legacy bugs are corrected: one-class proposals are excluded because
ROC-AUC is undefined, and the checkpoint replay restores the original weights
and batching RNG. Probing cannot change subsequent training batches. Thus
both passes follow the same trajectory, including seed 0. Random masks are
seeded and have at least one pair of each class; they are not partitions.

Each seed caches training.csv, scores.csv, summary.csv, heatmaps.npz, and
config.json under results/phase_transitions/<configuration>/seed<N>/.
The first run uses two training passes to locate and probe the transition;
reruns only plot, and incomplete sweeps resume at the next missing seed.
No weights or raw hidden tensors are retained. The unversioned legacy CSV
is not reused because it lacks configuration and trajectory provenance.

Outputs in figures/: fig7_phase_transition_lineplot.png,
fig7_phase_transitions/seed<N>_{loss,tminus_heatmap,tplus_heatmap}.png.
Nondefault settings use configuration-specific figure names/directories.
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import colorcet as cc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import roc_auc_score

from scripts.common import save_figure
from src.experiments import get_task_diagram, get_task_lattice
from src.paths import FIGURES_DIR, RESULTS_DIR
from src.projections import get_pairwise_decodings
from src.tasks import TASKS
from src.training import (ConstrainedRNN, DataGenerator, flatten_states,
                          get_hidden_states)


DEFAULTS = dict(task='limitedstack', train_L=7, hidden_dim=5, lnorm=1.2,
                epochs=5000, learning_rate=0.001, batch_size=128,
                delta=100, num_random=100)
N_SEEDS = 101
CACHE_VERSION = 1
COLORS = {'cc': cc.glasbey[3], 'rand': '#9A9498'}


def identify_breakthroughs(loss_df, delta=100):
    """Legacy forward-minus-backward absolute loss change, clipped at zero."""
    if delta < 1 or len(loss_df) <= 2 * delta:
        raise ValueError('Require delta >= 1 and epochs > 2 * delta.')
    result = loss_df.copy()
    loss = result['loss'].to_numpy()
    if not np.isfinite(loss).all():
        raise ValueError('Cannot identify a transition from nonfinite losses.')
    values = np.zeros(len(loss))
    values[delta:-delta] = np.maximum(
        0, np.abs(loss[2 * delta:] - loss[delta:-delta])
        - np.abs(loss[delta:-delta] - loss[:-2 * delta]))
    result['values'] = values
    return result


def transition_epochs(history, delta):
    """Return two valid checkpoint epochs around a positive interior maximum."""
    interior = history.iloc[delta:-delta]
    peak = interior['values'].idxmax()
    if history.loc[peak, 'values'] <= 0:
        raise ValueError('No positive phase-transition score; no transition selected.')
    peak_epoch = int(history.loc[peak, 'epochs'])
    return peak_epoch - delta, peak_epoch + delta


def make_proposals(lattice, num_random, seed):
    """Nontrivial consistent matrices and legacy Bernoulli upper-triangle masks."""
    if num_random < 1:
        raise ValueError('num_random must be positive.')
    n_states = lattice.nodes[0].pairwise_matrix.shape[0]
    triangle = np.triu_indices(n_states, k=1)
    proposals = {}
    skipped = 0
    for node in lattice.nodes:
        proposal = np.asarray(node.pairwise_matrix[triangle], dtype=int)
        if np.unique(proposal).size != 2:
            skipped += 1
            continue
        proposals[f'cc_{node.id}'] = proposal
    if not proposals:
        raise ValueError('No consistent coloring has a defined pairwise ROC-AUC.')
    print(f'Excluded {skipped} one-class consistent coloring(s) from ROC-AUC.')
    probabilities = [proposal.mean() for proposal in proposals.values()]
    rng = np.random.RandomState(seed)
    for index in range(num_random):
        for _ in range(10000):
            probability = rng.uniform(min(probabilities), max(probabilities))
            proposal = rng.binomial(1, probability, size=len(triangle[0]))
            if np.unique(proposal).size == 2:
                proposals[f'rand_{index}'] = proposal
                break
        else:
            raise RuntimeError('Could not generate a two-class random pair mask.')
    return proposals


def score_proposals(matrix, proposals, seed, epoch, timepoint):
    """Treat coloring pair identities as truth and decoding accuracies as scores."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Expected a square pairwise decoding matrix.')
    scores = matrix[np.triu_indices(matrix.shape[0], k=1)]
    if not np.isfinite(scores).all():
        raise ValueError('Pairwise decoding contains nonfinite accuracies.')
    return pd.DataFrame([
        dict(seed=seed, epoch=epoch, t=timepoint, candidate=name,
             comparison='rand' if name.startswith('rand_') else 'cc',
             auc=roc_auc_score(proposal, scores))
        for name, proposal in proposals.items()
    ])


def select_winners(scores):
    """Choose each family's t+ winner, keeping its paired t- measurement."""
    rows = []
    for comparison in ('cc', 'rand'):
        family = scores[scores['comparison'] == comparison]
        after = family[family['t'] == 't+']
        winner = after.loc[after['auc'].idxmax(), 'candidate']
        selected = family[family['candidate'] == winner].set_index('t')
        if set(selected.index) != {'t-', 't+'} or len(selected) != 2:
            raise ValueError('Every winning candidate must have exactly two checkpoints.')
        for timepoint in ('t-', 't+'):
            row = selected.loc[timepoint]
            rows.append(dict(seed=int(row['seed']), t=timepoint,
                             comparison=comparison, candidate=winner,
                             epoch=int(row['epoch']), auc=row['auc']))
    return pd.DataFrame(rows)


def train_trajectory(model, data, config, probe_epochs=(), on_probe=None):
    """Legacy penalized CE loop with RNG-neutral, post-update checkpoint probes."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    history = []
    stop = max(probe_epochs) + 1 if probe_epochs else config['epochs']
    for epoch in range(stop):
        model.train()
        losses, accuracies = [], []
        for inputs, targets, _ in data.batch_generator(train=True):
            optimizer.zero_grad()
            outputs, hidden = model(inputs)
            targets = targets.argmax(dim=-1).reshape(-1)
            predictions = outputs.reshape(-1, outputs.shape[-1])
            loss = criterion(predictions, targets).mean()
            loss += hidden.norm(p=2, dim=-1).mean() * model.multiplier
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss at epoch {epoch}.')
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            accuracies.append((predictions.argmax(-1) == targets).float().mean().item())
        history.append(dict(epochs=epoch, loss=np.mean(losses), acc=np.mean(accuracies)))
        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}/{stop} - Loss: {history[-1]['loss']:.4f}"
                  f" - Acc: {history[-1]['acc']:.4f}")
        if epoch in probe_epochs:
            if on_probe is None:
                raise ValueError('A checkpoint callback is required for probe epochs.')
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                model.eval()
                on_probe(model, epoch)
    return pd.DataFrame(history)


def compute_analysis(config, seed):
    cd = get_task_diagram(config['task'])
    lattice = get_task_lattice(config['task'])
    proposals = make_proposals(lattice, config['num_random'], seed)
    torch.manual_seed(0)
    data = DataGenerator(TASKS[config['task']]['generate'](config['train_L']),
                         cd, batch_size=config['batch_size'])
    model = ConstrainedRNN(len(cd.X), config['hidden_dim'], len(cd.Y),
                           seed=seed, lnorm=config['lnorm'])
    initial_weights = copy.deepcopy(model.state_dict())
    initial_rng = torch.get_rng_state()
    print(f'[seed {seed}] locating transition')
    history = identify_breakthroughs(train_trajectory(model, data, config),
                                    config['delta'])
    checkpoints = transition_epochs(history, config['delta'])
    heatmaps, scores = {}, []

    def probe(network, epoch):
        hidden, truths, _, _ = get_hidden_states(network, data, train=True)
        predictor, target = flatten_states(hidden, truths)
        expected = torch.arange(len(cd.states), device=target.device)
        if not torch.equal(target.unique(), expected):
            raise ValueError('Training observations do not cover all diagram states; '
                             'cannot align decoding and coloring matrices.')
        matrix = get_pairwise_decodings(predictor, target)
        timepoint = 't-' if epoch == checkpoints[0] else 't+'
        heatmaps[timepoint] = matrix
        scores.append(score_proposals(matrix, proposals, seed, epoch, timepoint))

    model.load_state_dict(initial_weights)
    torch.set_rng_state(initial_rng)
    print(f'[seed {seed}] probing epochs {checkpoints}')
    replay = train_trajectory(model, data, config, checkpoints, probe)
    if not np.array_equal(replay[['loss', 'acc']].to_numpy(),
                          history[['loss', 'acc']].iloc[:len(replay)].to_numpy()):
        raise RuntimeError('Checkpoint replay diverged from the detected trajectory.')
    scores = pd.concat(scores, ignore_index=True)
    labels = [cd.state_labels[index] for index in range(len(cd.states))]
    return history, scores, select_winners(scores), heatmaps, labels


def configuration_key(config):
    metadata = dict(version=CACHE_VERSION, config=config)
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    return digest, metadata


def load_or_compute(config, seed, recompute=False):
    key, metadata = configuration_key(config)
    directory = RESULTS_DIR / 'phase_transitions' / key / f'seed{seed}'
    marker = directory / 'config.json'
    files = ['training.csv', 'scores.csv', 'summary.csv', 'heatmaps.npz']
    if not recompute and marker.exists() and all((directory / f).exists() for f in files):
        saved = json.loads(marker.read_text())
        if saved['metadata'] == metadata and saved['seed'] == seed:
            with np.load(directory / 'heatmaps.npz') as arrays:
                heatmaps = {name: arrays[name].copy() for name in ('t-', 't+')}
            print(f'[seed {seed}] using cached phase-transition results')
            return (pd.read_csv(directory / 'training.csv'),
                    pd.read_csv(directory / 'scores.csv'),
                    pd.read_csv(directory / 'summary.csv'), heatmaps, saved['labels'])
    result = compute_analysis(config, seed)
    history, scores, summary, heatmaps, labels = result
    directory.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    for name, frame in zip(files[:3], (history, scores, summary)):
        frame.to_csv(directory / name, index=False)
    np.savez_compressed(directory / 'heatmaps.npz', **heatmaps)
    marker.write_text(json.dumps(dict(metadata=metadata, seed=seed, labels=labels),
                                 indent=2))
    return result


def plot_seed(history, summary, heatmaps, labels, seed, prefix):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'phase_viridis', plt.get_cmap('viridis')(np.linspace(0.2, 0.8, 16)))
    norm = mcolors.Normalize(history['values'].min(), history['values'].max())
    fig, ax = plt.subplots(figsize=(6, 4))
    for epoch, value in zip(history['epochs'], history['values']):
        ax.axvspan(epoch, epoch + 1, color=cmap(norm(value)), alpha=0.4)
    ax.plot(history['epochs'], history['loss'], color='black')
    for timepoint in ('t-', 't+'):
        epoch = summary.loc[summary['t'] == timepoint, 'epoch'].iloc[0]
        ax.axvline(epoch, color='black', linestyle='--', linewidth=0.8)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    bar.set_label('Phase transition score', rotation=270, labelpad=15)
    ax.set(xlabel='Epochs', ylabel='Loss')
    save_figure(f'{prefix}/seed{seed}_loss')
    plt.close(fig)
    for timepoint, suffix in (('t-', 'tminus'), ('t+', 'tplus')):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(heatmaps[timepoint], vmin=0.5, vmax=1, ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.tick_params(axis='x', labelrotation=90)
        ax.tick_params(axis='y', labelrotation=0)
        save_figure(f'{prefix}/seed{seed}_{suffix}_heatmap')
        plt.close(fig)


def plot_comparison(summary, suffix=''):
    wide = summary.pivot(index='seed', columns=['t', 'comparison'], values='auc')
    order = [('t-', 'cc'), ('t-', 'rand'), ('t+', 'cc'), ('t+', 'rand')]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(order))
    if wide.isna().any().any():
        raise ValueError('The comparison plot requires both families at both checkpoints.')
    positions = np.array([1.1, 1.4, 2.3, 2.6])
    jitter = np.random.default_rng(0).normal(0, 0.05, wide.shape) + positions
    fig, ax = plt.subplots(figsize=(6, 5))
    for index, (_, family) in enumerate(order):
        ax.plot(jitter[:, index], wide.iloc[:, index], 'o', alpha=0.7,
                zorder=1, ms=8, mew=1, color=COLORS[family])
    for index in range(len(wide)):
        for family, columns in (('cc', [0, 2]), ('rand', [1, 3])):
            ax.plot(jitter[index, columns], wide.iloc[index, columns],
                    color=COLORS[family], linewidth=0.5, zorder=-1, alpha=0.3)
    ax.set_xticks([1.25, 2.45], labels=['t-', 't+'], fontsize=16)
    ax.set_xlabel('Phase Transition', fontsize=16)
    ax.set_ylabel('AUC', fontsize=16)
    handles = [Line2D([0], [0], marker='o', color='w', label=label,
                      markerfacecolor=COLORS[family], markersize=8)
               for family, label in (('cc', 'Consistent coloring'), ('rand', 'Best Random'))]
    ax.legend(handles=handles, loc='lower right', frameon=False)
    save_figure(f'fig7_phase_transition_lineplot{suffix}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seeds', type=int, default=N_SEEDS, help='Seeds 0..N-1.')
    for option in ('epochs', 'delta', 'hidden_dim', 'num_random'):
        parser.add_argument('--' + option.replace('_', '-'), type=int,
                            default=DEFAULTS[option])
    parser.add_argument('--lnorm', type=float, default=DEFAULTS['lnorm'])
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    if min(args.seeds, args.delta, args.hidden_dim, args.num_random) < 1:
        parser.error('Seeds, delta, hidden dimension, and random-mask count must be positive.')
    if not np.isfinite(args.lnorm) or args.lnorm <= 0:
        parser.error('--lnorm must be finite and positive.')
    if args.epochs <= 2 * args.delta:
        parser.error('--epochs must be greater than twice --delta.')
    config = dict(DEFAULTS, **{key: getattr(args, key)
                              for key in ('epochs', 'delta', 'hidden_dim',
                                          'num_random', 'lnorm')})
    key, _ = configuration_key(config)
    suffix = '' if config == DEFAULTS else f'_{key}'
    prefix = f'fig7_phase_transitions{suffix}'
    (FIGURES_DIR / prefix).mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in range(args.seeds):
        history, _, summary, heatmaps, labels = load_or_compute(
            config, seed, recompute=args.recompute)
        plot_seed(history, summary, heatmaps, labels, seed, prefix)
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)
    plot_comparison(summary, suffix)
    print(summary.groupby(['t', 'comparison'])['auc'].agg(['count', 'mean', 'sem']))


if __name__ == '__main__':
    main()
