"""Figure 2: hidden-state geometry across the five paper tasks.

For each task (3-back, listmax, limitedset, limitedlist, limitedstack), saves
LDA/PCA scatter panels, a main held-out state-decoding boxplot (LDA/PCA/ICA/FA),
a supplementary boxplot (LDA, history-LDA, shuffled-label control), and an
untrained LDA panel.

Legacy pipeline: scripts_old/run_<task>.py -> run_comparison.py (projections,
dimensionality_test, results/simulations/<task>.csv) -> visualization.py viz_1
(boxplots). Hyperparameters per task are in src/tasks.py (from simulations.sh).

Usage:
  python scripts/fig3_task_panels.py                 # cached runs if available
  python scripts/fig3_task_panels.py --seeds 10      # boxplot seeds 0..9
  python scripts/fig3_task_panels.py --tasks nback3 listmax
  python scripts/fig3_task_panels.py --recompute     # retrain everything
  python scripts/fig3_task_panels.py --history-decoding --seeds 5
      # main and supplementary accuracy plots, without regenerating scatter panels

Output: figures/{lda,pca}_<task>.png, figures/<task>_simulations.png,
        figures/<task>_supplementary.png,
        figures/lda_untrained_<task>.png (scatter panels unless --history-decoding)
Both modes save main and supplementary accuracy plots with automatic y limits.
Storage: results/decoding_accuracies_heldout.csv (all seeds, tiny, permanent);
         shuffled controls remain CSV-only as method=LDA_shuffled.
         results/hidden_states/<run>.pt only for the scatter seed per task --
         all other seeds are trained, scored into the CSV, and discarded.
History scores are stored in results/history_state_decoding_heldout.csv.
They use complete windows ending at x_t, paired with h_t. LDA uses history
labels, but classifiers predict original diagram-state labels. All projections,
scalers, and classifiers are fitted on training sequences only, then evaluated
on the existing 20% test sequences. Binary k=1 histories have one LDA axis.
Listmax stops at k=2 because its k=3 classes have zero within-class variation.
Unavailable state pairs are excluded with a console message. No scatter-only
filters are applied to accuracies. Old caches without aligned inputs require
retraining for the supplement. Legacy in-sample CSVs are left untouched.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import save_figure
from src.experiments import (METHODS, SHUFFLED_METHOD, comparison_run_accuracies,
                             project_and_score, scatter_run, untrained_run)
from src.tasks import TASKS

# ---------------- parameters ----------------
TASK_NAMES = ['nback3', 'listmax', 'limitedset', 'limitedlist', 'limitedstack']
SCATTER_SEED = 0      # the run shown in the LDA/PCA plots
N_SEEDS = 5           # seeds 0..N-1 feed the accuracy boxplots (legacy: 101)
JITTER_SD = 0.1       # visual jitter for near-point state clusters (nback only)


def save_scatter(name, projection, target, state_labels, jitter, seed):
    """One scatter plot colored by diagram state, legend outside right
    (single column, as in the legacy code)."""
    pos_2d = pd.DataFrame(projection, columns=['x', 'y'])
    pos_2d['states'] = target
    pos_2d['states'] = pos_2d['states'].astype('category').map(state_labels)
    order = [state_labels[k] for k in sorted(state_labels)
             if state_labels[k] in set(pos_2d['states'])]
    pos_2d['states'] = pd.Categorical(pos_2d['states'], categories=order, ordered=True)
    if jitter:
        rng = np.random.default_rng(seed)
        pos_2d[['x', 'y']] += rng.normal(0, JITTER_SD, pos_2d[['x', 'y']].shape)

    plt.figure()
    sns.scatterplot(data=pos_2d, x='x', y='y', hue='states',
                    palette=cc.glasbey_bw, alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)
    save_figure(name)
    plt.close()


def save_accuracy_boxplot(name, df, order):
    """Plot seed distributions without imposing y-axis limits."""
    plt.figure(figsize=(4, 4))
    sns.boxplot(data=df, x='method', y='accuracy', order=order, color='lightgray')
    for position, method in enumerate(order):
        if df.loc[df['method'] == method, 'accuracy'].notna().sum() == 0:
            plt.gca().text(position, 0.05, 'Unavailable', ha='center',
                           transform=plt.gca().get_xaxis_transform())
    plt.xlabel('')
    plt.ylabel('')
    save_figure(name)
    plt.close()


def save_comparison_boxplots(task_name, accuracy_df, history_df):
    """Save main methods and supplementary history/shuffled-label comparisons."""
    main = accuracy_df[(accuracy_df['split'] == 'test')
                       & accuracy_df['method'].isin(METHODS)]
    save_accuracy_boxplot(f'{task_name}_simulations', main, METHODS)

    history_lengths = list(range(1, TASKS[task_name].get('history_max_k', 3) + 1))
    history_labels = {k: f'k={k}' for k in history_lengths}
    lda = accuracy_df[(accuracy_df['split'] == 'test')
                     & (accuracy_df['method'] == 'LDA')]
    shuffled = accuracy_df[(accuracy_df['split'] == 'test')
                          & (accuracy_df['method'] == SHUFFLED_METHOD)].copy()
    shuffled['method'] = 'Random'
    history = history_df[(history_df['split'] == 'test')
                         & history_df['k'].isin(history_lengths)].copy()
    history['method'] = history['k'].map(history_labels)
    supplementary = pd.concat(
        [lda[['method', 'accuracy']],
         history[['method', 'accuracy']],
         shuffled[['method', 'accuracy']]],
        ignore_index=True)
    order = ['LDA'] + list(history_labels.values()) + ['Random']
    save_accuracy_boxplot(f'{task_name}_supplementary', supplementary, order)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tasks', nargs='+', default=TASK_NAMES, choices=TASK_NAMES)
    parser.add_argument('--seeds', type=int, default=N_SEEDS,
                        help='Number of seeds (0..N-1) for the boxplots.')
    parser.add_argument('--recompute', action='store_true',
                        help='Retrain even if cached runs exist.')
    parser.add_argument('--history-decoding', action='store_true',
                        help='Generate main and supplementary accuracy plots; skip scatter panels.')
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('--seeds must be at least 1.')

    for task_name in args.tasks:
        runs = [comparison_run_accuracies(
            task_name, seed, recompute=args.recompute,
            keep_hidden=(seed == SCATTER_SEED)) for seed in range(args.seeds)]
        acc_df = pd.concat([main for main, _ in runs], ignore_index=True)
        history_df = pd.concat([history for _, history in runs], ignore_index=True)
        save_comparison_boxplots(task_name, acc_df, history_df)
        if args.history_decoding:
            continue

        # scatter plots from the persisted scatter-seed run (optionally
        # restricted to a subset of states, e.g. nback3 full-history states
        # only, limitedstack w/o 'E')
        run = scatter_run(task_name, SCATTER_SEED)
        state_filter = TASKS[task_name].get('plot_states')
        keep_states = None
        if state_filter is not None:
            keep_states = {s for s, lab in run['state_labels'].items()
                           if state_filter(lab)}
        projections, _, target = project_and_score(
            run['hidden_train'], run['truths_train'],
            run['hidden_test'], run['truths_test'],
            keep_states=keep_states, score=False, seed=SCATTER_SEED)
        jitter = TASKS[task_name]['jitter']
        save_scatter(f'lda_{task_name}', projections['LDA'], target,
                     run['state_labels'], jitter, SCATTER_SEED)
        save_scatter(f'pca_{task_name}', projections['PCA'], target,
                     run['state_labels'], jitter, SCATTER_SEED)

        # reviewer baseline: LDA projection of the UNTRAINED network (same
        # data, same initial weights, no training)
        base = untrained_run(task_name, SCATTER_SEED)
        base_proj, _, base_target = project_and_score(
            base['hidden_train'], base['truths_train'],
            base['hidden_test'], base['truths_test'],
            keep_states=keep_states, score=False, seed=SCATTER_SEED)
        save_scatter(f'lda_untrained_{task_name}', base_proj['LDA'], base_target,
                     base['state_labels'], jitter, SCATTER_SEED)

if __name__ == '__main__':
    main()
