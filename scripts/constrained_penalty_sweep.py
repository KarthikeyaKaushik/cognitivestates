"""Sweep stack-RNN activity penalties and save final accuracies and heatmaps.

Defaults to lnorm 2.5, 3, 4, 5, seeds 1 through 5, and 2,000 epochs.
The activity penalty coefficient is 1/lnorm for positive values. lnorm=0
is a special no-activity-penalty condition, not an inverse penalty. Each run saves its checkpoint,
final-checkpoint train/test timestep accuracy, and the same in-sample,
reference-state pairwise LDA heatmap as constrained_stack_behavior.py.
The Frobenius norm of the effective readout weights is logged every 100
epochs and stored in newly trained checkpoints; final norms are also reported
for cached checkpoints.
By default, each readout row's weight norm is frozen at initialization;
directions and biases remain trainable. Use --readout-norm learnable for the
original baseline. The two conditions use distinct caches and matching seeds.

Usage:
  python scripts/constrained_penalty_sweep.py
  python scripts/constrained_penalty_sweep.py --lnorms 0.6 --seeds 1 --epochs 10
  python scripts/constrained_penalty_sweep.py --readout-norm learnable
  python scripts/constrained_penalty_sweep.py --lnorms 0
  python scripts/constrained_penalty_sweep.py --readout-scale 4 --lnorms 2 3 4
  python scripts/constrained_penalty_sweep.py --retrain

Completed networks and matrices are cached per configuration and seed; summary
CSVs and plots are updated after every run so an interrupted sweep is resumable.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts.constrained_stack_behavior import (
    SETTINGS, TASK, load_or_train_network, save_decoding_heatmap,
)
from src.experiments import get_task_diagram
from src.paths import RESULTS_DIR, ensure_dir
from src.tasks import TASKS
from src.training import DataGenerator, readout_weight_norm


LNORMS = (2.5, 3.0, 4.0, 5.0)
SEEDS = (1, 2, 3, 4, 5)
EPOCHS = 2000
OUTPUT_DIR = RESULTS_DIR / 'constrained_penalty_sweep'


def configuration_id(configuration):
    encoded = json.dumps(configuration, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]


def evaluate_accuracy(model, batches, *, train):
    """Count correct outputs across every timestep of one split."""
    correct = total = 0
    model.eval()
    with torch.inference_mode():
        for inputs, targets, _ in batches.batch_generator(train=train):
            logits, _ = model(inputs)
            if not torch.isfinite(logits).all():
                raise ValueError('Network produced non-finite output logits')
            expected = targets.argmax(dim=-1)
            correct += (logits.argmax(dim=-1) == expected).sum().item()
            total += expected.numel()
    if not total:
        raise ValueError('Cannot evaluate accuracy on an empty split')
    return correct / total


def run_one(root, lnorm, seed, epochs, *, retrain=False, readout_norm='fixed',
            readout_scale=1.0):
    if readout_norm not in ('fixed', 'learnable'):
        raise ValueError(f'Unsupported readout norm condition: {readout_norm}')
    if not np.isfinite(lnorm) or lnorm < 0:
        raise ValueError('lnorm must be finite and nonnegative')
    if not np.isfinite(readout_scale) or readout_scale <= 0:
        raise ValueError('readout scale must be finite and positive')
    if readout_norm != 'fixed' and readout_scale != 1.0:
        raise ValueError('readout scaling requires fixed readout norms')
    settings = dict(SETTINGS, lnorm=lnorm, epochs=epochs)
    if readout_norm == 'fixed':
        settings['fixed_readout_norm'] = True
        if readout_scale != 1.0:
            settings['readout_scale'] = float(readout_scale)
    cd = get_task_diagram(TASK)
    input_tokens, output_tokens = sorted(cd.X), sorted(cd.Y)
    configuration = dict(task=TASK, seed=seed, **settings,
                         input_tokens=input_tokens, output_tokens=output_tokens,
                         split_seed=0, train_fraction=0.8, cache_version=1)
    run_id = (f'lnorm{lnorm:g}_seed{seed}_ep{epochs}_'
              f'{configuration_id(configuration)}')
    directory = root / 'runs' / run_id
    model, metadata = load_or_train_network(
        directory, seed, input_tokens, output_tokens, retrain=retrain,
        settings=settings)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        batches = DataGenerator(TASKS[TASK]['generate'](settings['train_L']), cd,
                                batch_size=settings['batch_size'])
        train_accuracy = evaluate_accuracy(model, batches, train=True)
        test_accuracy = evaluate_accuracy(model, batches, train=False)

    checkpoint_hash = hashlib.sha256((directory / 'network.pt').read_bytes()).hexdigest()
    heatmap_metadata = dict(metadata, checkpoint_sha256=checkpoint_hash)
    heatmap_path = directory / 'decoding_heatmap.npz'
    recompute_heatmap = retrain
    if heatmap_path.exists() and not recompute_heatmap:
        with np.load(heatmap_path) as saved:
            previous = json.loads(str(saved['metadata'].item()))
        if previous != heatmap_metadata:
            recompute_heatmap = True
    save_decoding_heatmap(directory, heatmap_metadata, model,
                          recompute=recompute_heatmap)
    row = dict(run_id=run_id, seed=seed, lnorm=lnorm,
               penalty=1 / lnorm if lnorm else 0.0,
               epochs=epochs, readout_norm=readout_norm, readout_scale=readout_scale,
               train_accuracy=train_accuracy,
               test_accuracy=test_accuracy,
               readout_weight_norm=readout_weight_norm(model),
               train_sequences=len(batches.train),
               test_sequences=len(batches.test))
    (directory / 'metrics.json').write_text(json.dumps(
        dict(configuration=configuration, metrics=row), indent=2) + '\n')
    print(f'Final checkpoint: lnorm={lnorm:g}, seed={seed}, readout={readout_norm}, '
          f'train={train_accuracy:.4f}, test={test_accuracy:.4f}, '
          f'readout weight norm={row["readout_weight_norm"]:.4f}', flush=True)
    return row


def save_summary(root, sweep_id, rows, lnorms):
    directory = ensure_dir(root / 'summaries' / sweep_id)
    df = pd.DataFrame(rows)
    df.to_csv(directory / 'runs.csv', index=False)
    summary = df.groupby('lnorm', sort=True).agg(
        networks=('seed', 'size'),
        train_mean=('train_accuracy', 'mean'),
        train_sem=('train_accuracy', 'sem'),
        test_mean=('test_accuracy', 'mean'),
        test_sem=('test_accuracy', 'sem'),
    ).reset_index()
    summary.to_csv(directory / 'summary.csv', index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        for offset, column, label, color in (
                (-0.12, 'train_accuracy', 'Train', 'tab:blue'),
                (0.12, 'test_accuracy', 'Held-out test', 'tab:orange')):
            for index, lnorm in enumerate(lnorms):
                group = df[df['lnorm'] == lnorm][column]
                if group.empty:
                    continue
                x = index + offset
                ax.scatter([x] * len(group), group, color=color, alpha=0.55,
                           s=25, zorder=2)
                sem = group.sem() if len(group) > 1 else 0
                ax.errorbar(x, group.mean(), yerr=sem, fmt='_', color=color,
                            capsize=4, markersize=13, linewidth=1.5, zorder=3,
                            label=label if index == 0 else None)
        ax.set_xticks(range(len(lnorms)), [
            f'{value:g}' if value else 'No penalty' for value in lnorms])
        ax.set_xlabel('lnorm (activity penalty = 1 / lnorm, except no penalty)')
        ax.set_ylabel('Final checkpoint timestep accuracy')
        ax.set_ylim(0, 1.02)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / 'accuracy.png', dpi=300)
    finally:
        plt.close(fig)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--lnorms', type=float, nargs='+', default=LNORMS,
                        help='Positive lnorm values, or 0 for no activity penalty.')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--readout-norm', choices=('fixed', 'learnable'),
                        default='fixed',
                        help='Freeze readout row norms at initialization (default: fixed).')
    parser.add_argument('--readout-scale', type=float, default=1.0,
                        help='Multiply each initial readout row norm by this factor.')
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()
    if any(not (0 <= value < float('inf')) for value in args.lnorms):
        parser.error('lnorm values must be finite and nonnegative')
    if len(set(args.lnorms)) != len(args.lnorms):
        parser.error('lnorm values must be distinct')
    if any(seed < 1 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be distinct positive integers')
    if args.epochs < 1:
        parser.error('epochs must be positive')
    if not np.isfinite(args.readout_scale) or args.readout_scale <= 0:
        parser.error('readout scale must be finite and positive')
    if args.readout_norm != 'fixed' and args.readout_scale != 1.0:
        parser.error('--readout-scale requires --readout-norm fixed')

    sweep = dict(lnorms=list(args.lnorms), seeds=list(args.seeds),
                 settings=dict(SETTINGS, epochs=args.epochs), cache_version=1)
    if args.readout_norm == 'fixed':
        sweep['settings']['fixed_readout_norm'] = True
        if args.readout_scale != 1.0:
            sweep['settings']['readout_scale'] = args.readout_scale
    sweep_id = configuration_id(sweep)
    rows = []
    for lnorm in args.lnorms:
        for seed in args.seeds:
            rows.append(run_one(args.output_dir, lnorm, seed, args.epochs,
                                retrain=args.retrain, readout_norm=args.readout_norm,
                                readout_scale=args.readout_scale))
            summary_dir = save_summary(args.output_dir, sweep_id, rows, args.lnorms)
    print(f'Sweep results saved to {summary_dir}', flush=True)


if __name__ == '__main__':
    main()
