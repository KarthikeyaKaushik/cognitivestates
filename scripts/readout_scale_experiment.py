"""Compare fixed readout scales with no hidden activity penalty on the stack task.

Each seed shares an initialization and training split across scales; each
readout row's initial norm is multiplied by the scale and frozen. Directions,
biases and recurrent weights remain trainable. Defaults: scales 1, 2, 4, 8;
seeds 1-5; 2,000 epochs. Runs reuse constrained_penalty_sweep caches.

Usage:
  python scripts/readout_scale_experiment.py
  python scripts/readout_scale_experiment.py --seeds 1 2 --epochs 100

Per-run checkpoints and reference-state in-sample LDA heatmaps are saved under
results/constrained_penalty_sweep/runs/. Experiment-specific summary CSVs and
plots are saved under results/readout_scale_experiment/<configuration-id>/.
Learning curves are periodically logged training-epoch minibatch averages,
not evaluation of fixed checkpoints or held-out accuracy.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts import constrained_penalty_sweep as sweep
from src.paths import RESULTS_DIR, ensure_dir


SCALES = (2, 4, 8, 1)
SEEDS = (1, 2, 3, 4, 5)
OUTPUT_DIR = RESULTS_DIR / 'readout_scale_experiment'


def save_results(directory, rows, history_rows, scales):
    ensure_dir(directory)
    df = pd.DataFrame(rows)
    df.to_csv(directory / 'runs.csv', index=False)
    summary = df.groupby('readout_scale').agg(
        networks=('seed', 'size'),
        train_mean=('train_accuracy', 'mean'),
        train_sem=('train_accuracy', 'sem'),
        test_mean=('test_accuracy', 'mean'),
        test_sem=('test_accuracy', 'sem'),
    ).reset_index()
    summary.to_csv(directory / 'summary.csv', index=False)
    history = pd.DataFrame(history_rows)
    history.to_csv(directory / 'learning_curves.csv', index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        for offset, column, label, color in (
                (-0.12, 'train_accuracy', 'Train', 'tab:blue'),
                (0.12, 'test_accuracy', 'Held-out test', 'tab:orange')):
            first = True
            for index, scale in enumerate(scales):
                group = df.loc[df['readout_scale'] == scale, column]
                if group.empty:
                    continue
                x = index + offset
                ax.scatter([x] * len(group), group, color=color, alpha=0.55)
                sem = group.sem() if len(group) > 1 else 0
                ax.errorbar(x, group.mean(), yerr=sem, fmt='_', color=color,
                            capsize=4, markersize=13, label=label if first else None)
                first = False
        ax.set_xticks(range(len(scales)), [f'{scale:g}x' for scale in scales])
        ax.set_xlabel('Fixed readout norm relative to initialization')
        ax.set_ylabel('Final checkpoint timestep accuracy')
        ax.set_ylim(0, 1.02)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / 'accuracy.png', dpi=300)
    finally:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        for scale in scales:
            subset = history[history['readout_scale'] == scale]
            if subset.empty:
                continue
            mean = subset.groupby('epoch')['training_accuracy'].mean()
            ax.plot(mean.index, mean.values, label=f'{scale:g}x (n={subset.seed.nunique()})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training-epoch minibatch accuracy')
        ax.set_ylim(0, 1.02)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / 'learning_curves.png', dpi=300)
    finally:
        plt.close(fig)


def run_experiment(scales, seeds, epochs, *, output_dir=OUTPUT_DIR):
    configuration = dict(scales=list(scales), seeds=list(seeds), epochs=epochs,
                         settings=dict(sweep.SETTINGS, lnorm=0.0, epochs=epochs,
                                       fixed_readout_norm=True), version=1)
    directory = output_dir / sweep.configuration_id(configuration)
    rows, history_rows = [], []
    for scale in scales:
        for seed in seeds:
            row = sweep.run_one(sweep.OUTPUT_DIR, 0.0, seed, epochs,
                                readout_norm='fixed', readout_scale=scale)
            checkpoint_path = sweep.OUTPUT_DIR / 'runs' / row['run_id'] / 'network.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            history = checkpoint['training_history']
            if len(history['epochs']) != len(history['acc']):
                raise ValueError(f'Inconsistent training history: {checkpoint_path}')
            if not history['epochs']:
                raise ValueError(f'Empty training history: {checkpoint_path}')
            history_rows.extend(
                dict(run_id=row['run_id'], readout_scale=scale, seed=seed,
                     epoch=epoch + 1, training_accuracy=accuracy, training_loss=loss)
                for epoch, accuracy, loss in zip(history['epochs'], history['acc'],
                                                 history['loss'], strict=True))
            rows.append(row)
            save_results(directory, rows, history_rows, scales)
    print(f'Readout scaling results saved to {directory}', flush=True)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scales', type=float, nargs='+', default=SCALES)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--epochs', type=int, default=sweep.EPOCHS)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if any(not np.isfinite(x) or x <= 0 for x in args.scales) \
            or len(set(args.scales)) != len(args.scales):
        parser.error('scales must be distinct, finite, and positive')
    if any(seed < 1 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be distinct positive integers')
    if args.epochs < 1:
        parser.error('epochs must be positive')
    run_experiment(args.scales, args.seeds, args.epochs, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
