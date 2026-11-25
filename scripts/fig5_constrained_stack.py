"""Figure 3: limitedstack trained under strong activity constraints.

Sweeps lnorm in {0.6, 1.4, 2.2} (penalty weights 1/lnorm ~ 1.7 / 0.7 / 0.45)
over seeds 1..N, as in the legacy simulations_constrained.sh. Produces,
individually stored:

  figures/fig3_heatmaps/<run_key>.png
      end-of-training pairwise LDA decodability heatmap for every run
      (to pick a representative one for the paper)
  figures/limitedstack_lnorm_vs_acc.png
      boxplot: final training accuracy vs penalty weight (2-sigma outliers
      removed per lnorm, as in legacy visualization.py)
  figures/limitedstack_lnorm_vs_ncolors_stackedbar.png
      for each run, the consistent coloring of the diagram lattice with the
      highest ROC-AUC against the measured decodability matrix; stacked bars
      show the proportion of winning coloring sizes ("N colors") per penalty

Legacy pipeline: run.py limitedstack --mode constrained (run_constrained.py)
-> visualization.py viz_2 / viz_3.

Usage:
  python scripts/fig5_constrained_stack.py --seeds 5
  python scripts/fig5_constrained_stack.py --fixed-readout-sweep
  python scripts/fig5_constrained_stack.py --fixed-readout-sweep --readout-scale 4 --lnorms 0 2 3 4
  python scripts/fig5_constrained_stack.py --curated-summary results/constrained_penalty_sweep/summaries/31ef8554536726b5
  python scripts/fig5_constrained_stack.py --auc-histogram-summary results/constrained_penalty_sweep/summaries/31ef8554536726b5
  python scripts/fig5_constrained_stack.py --accuracy-summary results/constrained_penalty_sweep/summaries/31ef8554536726b5

The fixed-readout sweep uses lnorm 2..15, seeds 1..50, 2000 epochs and no
outlier filtering. It reuses the per-run checkpoints and heatmaps from
constrained_penalty_sweep.py and writes separate summary and coloring outputs.
--lnorms, --seeds, --epochs and --readout-scale can override the fixed-sweep
defaults. Readout scale defaults to 1 (the seeded initial row norms).
Use --lnorms 0 for the no-activity-penalty condition, or include 0 alongside
positive values to compare them. Zero is a special case, not 1/0.
--curated-summary replots saved ROC-AUC winners for lnorm 1..6, excluding
two manually identified runs. It writes separate outputs and never retrains.
--auc-histogram-summary plots count histograms for colorings 1, 2 and 11 across
all runs at lnorm 1..6, including the two excluded from the curated barplot.
It recomputes AUC from cached heatmaps, never trains or regenerates heatmaps.
The visible AUC range is 0.5 to 1; scores outside that range remain in the CSV.
--accuracy-summary plots final-checkpoint training and held-out accuracies in
separate figures for all saved runs at lnorm 1..6, without filtering or retraining.

Runs are resumable: each finished run stores a CSV row + a small .npy matrix
(results/constrained_runs.csv, results/constrained_heatmaps/); rerunning skips
completed runs and only regenerates missing heatmap PNGs.
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
import seaborn as sns
from sklearn.metrics import roc_auc_score

from scripts.common import save_figure
from src.experiments import constrained_run, get_task_diagram, get_task_lattice
from src.paths import DIAGRAMS_DIR, FIGURES_DIR, ensure_dir

# ---------------- parameters (legacy simulations_constrained.sh) ----------------
TASK = 'limitedstack'
LNORMS = [0.6, 1.4, 2.2]   # penalty weights 1/lnorm ~ 2.0, 0.71, 0.45
N_SEEDS = 100              # seeds 1..N (legacy: 1..100)
TRAIN_L = 7
HIDDEN_DIM = 5
EPOCHS = 1000
HEATMAP_DIR = FIGURES_DIR / 'fig3_heatmaps'
# Dropping because of misidentification
CURATED_EXCLUSIONS = {
    'lnorm1_seed74_ep2000_2d08607dd7cfa48c': 1.0,
    'lnorm3_seed10_ep2000_9bf612a28eecbae6': 3.0,
}
CURATED_LNORMS = (1, 2, 3, 4, 5, 6)
HISTOGRAM_COLORINGS = (1, 2, 11)


def remove_outliers(df):
    """Drop runs whose final_acc is >2 sigma from their lnorm group mean
    (legacy get_outliers in visualization.py)."""
    stats = df.groupby('lnorm')['final_acc'].agg(['mean', 'std'])
    df = df.merge(stats, on='lnorm')
    keep = ((df['final_acc'] <= df['mean'] + 2 * df['std'])
            & (df['final_acc'] >= df['mean'] - 2 * df['std']))
    return df[keep].drop(columns=['mean', 'std'])


def save_heatmap_png(run_key, matrix, state_labels):
    """One decodability heatmap per run (skipped if the PNG already exists)."""
    ensure_dir(HEATMAP_DIR)
    out_path = HEATMAP_DIR / f'{run_key}.png'
    if out_path.exists():
        return
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.heatmap(matrix, vmin=0.5, vmax=1.0, ax=ax)
    tick_labels = [state_labels[i] for i in range(matrix.shape[0])]
    ax.set_xticks(np.arange(len(tick_labels)) + 0.5)
    ax.set_yticks(np.arange(len(tick_labels)) + 0.5)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels, rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def save_acc_boxplot(df):
    plt.figure(figsize=(5, 5))
    ax = sns.boxplot(data=df, x='lnorm', y='final_acc', order=LNORMS,
                     color='lightgray')
    ax.set_xticklabels([f'{1 / l:.1f}' for l in LNORMS])
    ax.set_xlabel('Lnorm penalty', fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=16)
    save_figure(f'{TASK}_lnorm_vs_acc')
    plt.close()


def save_cached_accuracy_boxplot(summary_dir):
    """Plot final-checkpoint training and held-out accuracy from the saved sweep."""
    df = pd.read_csv(summary_dir / 'runs.csv')
    required = {'run_id', 'seed', 'lnorm', 'train_accuracy', 'test_accuracy'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing run columns: {sorted(missing)}')
    if df['run_id'].duplicated().any() or df.duplicated(['lnorm', 'seed']).any():
        raise ValueError('Duplicate run IDs or seed/lnorm pairs in runs.csv')
    selected = df[df['lnorm'].isin(CURATED_LNORMS)]
    if set(selected['lnorm']) != set(CURATED_LNORMS):
        raise ValueError('Saved runs do not cover every requested lnorm 1 through 6')
    for column in ('epochs', 'readout_norm', 'readout_scale'):
        if column in selected and selected[column].nunique(dropna=False) != 1:
            raise ValueError(f'Sweep mixes {column} settings')
    selected = selected.copy()
    for column in ('train_accuracy', 'test_accuracy'):
        accuracy = pd.to_numeric(selected[column], errors='raise')
        if not np.isfinite(accuracy).all() or not accuracy.between(0, 1).all():
            raise ValueError(f'{column} must be finite and between 0 and 1')
        selected[column] = accuracy
    labels = [str(value) for value in CURATED_LNORMS]
    selected['lnorm'] = selected['lnorm'].map(lambda value: f'{value:g}')
    paths = []
    for column, filename in (('train_accuracy', 'train_accuracy_boxplot.png'),
                             ('test_accuracy', 'test_accuracy_boxplot.png')):
        fig, ax = plt.subplots(figsize=(5, 5))
        try:
            sns.boxplot(data=selected, x='lnorm', y=column,
                        order=labels, color='lightgray', ax=ax)
            ax.set_xlabel('Lnorm penalty', fontsize=16)
            ax.set_ylabel('Accuracy', fontsize=16)
            fig.tight_layout()
            path = summary_dir / filename
            fig.savefig(path, dpi=300)
        finally:
            plt.close(fig)
        print(f'Saved {column} boxplot to {path}', flush=True)
        paths.append(path)
    return paths


def save_ncolors_stackedbar(df, matrices, lattice, *, lnorms=None, output_dir=None,
                            readout_scale=None):
    """Winner-take-all consistent coloring per run -> stacked proportions.

    AUC compares each coloring's pairwise-identity matrix against the measured
    decodability matrix, on the upper triangle excluding the diagonal (the
    diagonal and symmetric duplicates always agree and only pad the score).

    The full-state (root) coloring is excluded: it merges nothing, so its
    prediction is a single class and the AUC is undefined (the legacy code
    silently produced NaN here, which made the root 'win' every run).
    """
    triu = None
    winners = []
    for _, row in df.iterrows():
        matrix = matrices[row['run_key']]
        if triu is None:
            triu = np.triu_indices(matrix.shape[0], k=1)
        aucs = {}
        for cc in lattice.nodes:
            proposal = cc.pairwise_matrix[triu]
            if proposal.min() == proposal.max():
                continue  # nothing merged (root): AUC undefined
            aucs[cc.id] = roc_auc_score(proposal, matrix[triu])
        best = max(aucs, key=aucs.get)
        nstates = next(len(cc.get_sets()) for cc in lattice.nodes if cc.id == best)
        winners.append(dict(run_key=row['run_key'], lnorm=row['lnorm'],
                            cc=best, nstates=nstates, auc=aucs[best]))
    winners = pd.DataFrame(winners)

    rel = winners.groupby(['lnorm', 'nstates']).size().reset_index(name='count')
    rel['proportion'] = rel.groupby('lnorm')['count'].transform(lambda x: x / x.sum())
    lnorms = LNORMS if lnorms is None else lnorms
    pivot = rel.pivot_table(index='lnorm', columns='nstates', values='proportion',
                            fill_value=0).reindex(lnorms, fill_value=0)
    if output_dir is None:
        tick_labels = [str(np.round(1 / l, 1)) for l in pivot.index]
    else:
        tick_labels = [f'{l:g} ({1 / l:.3g})' if l else 'No activity penalty (0)'
                       for l in pivot.index]

    colors = plt.cm.viridis(np.linspace(0, 1, len(pivot.columns)))
    fig, ax = plt.subplots(figsize=(6, max(4, len(pivot) * 0.35)))
    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))
    for color, nstates in zip(colors, pivot.columns):
        ax.barh(y, pivot[nstates], left=left, label=int(nstates),
                color=color, alpha=0.8, height=1.0)
        left += pivot[nstates].values
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel('Proportion', fontsize=16)
    ax.set_ylabel('Lnorm penalty' if output_dir is None else 'lnorm (penalty)', fontsize=16)
    if readout_scale is not None:
        ax.set_title(f'Fixed readout norm: {readout_scale:g}x initialization')
    ax.legend(title='N colors', loc='upper left', bbox_to_anchor=(1.02, 0.65),
              frameon=False)
    ax.margins(y=0)
    if output_dir is None:
        save_figure(f'{TASK}_lnorm_vs_ncolors_stackedbar')
    else:
        ensure_dir(output_dir)
        winners.to_csv(output_dir / 'coloring_winners.csv', index=False)
        rel.to_csv(output_dir / 'coloring_proportions.csv', index=False)
        fig.tight_layout()
        fig.savefig(output_dir / 'ncolors_stackedbar.png', dpi=300)
    plt.close()


def curated_winners(summary_dir):
    """Select the specified saved winners, preserving the original results."""
    winners = pd.read_csv(summary_dir / 'coloring_winners.csv')
    required = {'run_key', 'lnorm', 'nstates', 'auc'}
    missing = required.difference(winners.columns)
    if missing:
        raise ValueError(f'Missing winner columns: {sorted(missing)}')
    if winners['run_key'].duplicated().any():
        raise ValueError('Duplicate run IDs in coloring winners')
    for run_key, lnorm in CURATED_EXCLUSIONS.items():
        excluded = winners[winners['run_key'] == run_key]
        if len(excluded) != 1 or excluded['lnorm'].iloc[0] != lnorm \
                or excluded['nstates'].iloc[0] != 15:
            raise ValueError(f'Expected 15-color winner not found: {run_key}')
    selected = winners[winners['lnorm'].isin(CURATED_LNORMS)
                       & ~winners['run_key'].isin(CURATED_EXCLUSIONS)].copy()
    if set(selected['lnorm']) != set(CURATED_LNORMS):
        raise ValueError('Saved winners do not cover every requested lnorm 1 through 6')
    return selected


def coloring_colors(nstates):
    """Assign the same viridis colors to the same sorted coloring sizes."""
    sizes = sorted(set(nstates))
    return dict(zip(sizes, plt.cm.viridis(np.linspace(0, 1, len(sizes)))))


def save_curated_stackedbar(summary_dir):
    """Replot the specified ROC-AUC winners without rescoring or retraining."""
    selected = curated_winners(summary_dir)
    proportions = selected.groupby(['lnorm', 'nstates']).size().reset_index(name='count')
    proportions['proportion'] = (proportions['count']
                                 / proportions.groupby('lnorm')['count'].transform('sum'))
    pivot = proportions.pivot_table(index='lnorm', columns='nstates',
                                    values='proportion', fill_value=0).reindex(CURATED_LNORMS)
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        left = np.zeros(len(pivot))
        y = np.arange(len(pivot))
        palette = coloring_colors(pivot.columns)
        for nstates in pivot.columns:
            ax.barh(y, pivot[nstates], left=left, label=int(nstates),
                    color=palette[nstates], alpha=0.8, height=1.0)
            left += pivot[nstates].values
        ax.set_yticks(y)
        ax.set_yticklabels([f'{value:g} ({1 / value:.3g})' for value in pivot.index])
        ax.set_xlabel('Proportion', fontsize=16)
        ax.set_ylabel('lnorm penalty', fontsize=16)
        ax.legend(title='N colors', loc='upper left', bbox_to_anchor=(1.02, 0.65),
                  frameon=False)
        ax.margins(y=0)
        fig.tight_layout()
        selected.to_csv(summary_dir / 'coloring_winners_curated.csv', index=False)
        proportions.to_csv(summary_dir / 'coloring_proportions_curated.csv', index=False)
        path = summary_dir / 'ncolors_stackedbar_curated.png'
        fig.savefig(path, dpi=300)
    finally:
        plt.close(fig)
    print(f'Saved curated ROC-AUC plot to {path}', flush=True)
    return path


def save_auc_histograms(summary_dir, runs_dir, lattice):
    """Compare three coloring AUCs for every saved run, using fixed count bins."""
    df = pd.read_csv(summary_dir / 'runs.csv')
    required = {'run_id', 'seed', 'lnorm'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing run columns: {sorted(missing)}')
    if df['run_id'].duplicated().any() or df.duplicated(['lnorm', 'seed']).any():
        raise ValueError('Duplicate run IDs or seed/lnorm pairs in runs.csv')
    for column in ('epochs', 'readout_norm', 'readout_scale'):
        if column in df and df[column].nunique(dropna=False) != 1:
            raise ValueError(f'Sweep mixes {column} settings')
    selected = df[df['lnorm'].isin(CURATED_LNORMS)]
    if set(selected['lnorm']) != set(CURATED_LNORMS):
        raise ValueError('Saved runs do not cover every requested lnorm 1 through 6')
    nodes = {}
    for color_id in HISTOGRAM_COLORINGS:
        matches = [node for node in lattice.nodes if node.id == color_id]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one coloring with ID {color_id}')
        nodes[color_id] = matches[0]
    palette = coloring_colors(curated_winners(summary_dir)['nstates'])
    absent = {len(node.get_sets()) for node in nodes.values()}.difference(palette)
    if absent:
        raise ValueError(f'Histogram coloring sizes absent from curated barplot: {sorted(absent)}')
    rows = []
    for row in selected.itertuples(index=False):
        run_id = str(row.run_id)
        if Path(run_id).name != run_id or run_id in ('.', '..'):
            raise ValueError(f'Invalid run ID: {run_id}')
        path = runs_dir / run_id / 'decoding_heatmap.npz'
        with np.load(path, allow_pickle=False) as data:
            matrix = data['matrix'].copy()
            metadata = json.loads(str(data['metadata'].item()))
        if (metadata.get('task') != TASK or metadata.get('seed') != row.seed
                or metadata.get('lnorm') != row.lnorm):
            raise ValueError(f'Heatmap metadata mismatch: {path}')
        if hasattr(row, 'epochs') and metadata.get('epochs') != row.epochs:
            raise ValueError(f'Heatmap epoch mismatch: {path}')
        if hasattr(row, 'readout_scale') and metadata.get('readout_scale', 1.0) != row.readout_scale:
            raise ValueError(f'Heatmap readout scale mismatch: {path}')
        if hasattr(row, 'readout_norm') and (
                bool(metadata.get('fixed_readout_norm', False)) != (row.readout_norm == 'fixed')):
            raise ValueError(f'Heatmap readout condition mismatch: {path}')
        if 'checkpoint_sha256' in metadata:
            checkpoint = runs_dir / run_id / 'network.pt'
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest != metadata['checkpoint_sha256']:
                raise ValueError(f'Heatmap checkpoint hash mismatch: {path}')
        if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
                or matrix.shape[0] < 2 or not np.isfinite(matrix).all()):
            raise ValueError(f'Invalid decoding matrix: {path}')
        triu = np.triu_indices(matrix.shape[0], k=1)
        for color_id, node in nodes.items():
            proposal = np.asarray(node.pairwise_matrix)
            if proposal.shape != matrix.shape:
                raise ValueError(f'Coloring {color_id} does not match heatmap {path}')
            labels = proposal[triu]
            if not np.isin(labels, (0, 1)).all() or labels.min() == labels.max():
                raise ValueError(f'Invalid ROC-AUC labels for coloring {color_id}')
            rows.append(dict(run_id=run_id, seed=row.seed, lnorm=row.lnorm,
                             cc=color_id, nstates=len(node.get_sets()),
                             auc=roc_auc_score(labels, matrix[triu])))
    scores = pd.DataFrame(rows)
    bins = np.linspace(0, 1, 100)
    fig, axes = plt.subplots(len(CURATED_LNORMS), 1, figsize=(5, 5),
                             sharex=True, sharey=True)
    try:
        for ax, lnorm in zip(axes, reversed(CURATED_LNORMS)):
            for color_id in HISTOGRAM_COLORINGS:
                group = scores[(scores['lnorm'] == lnorm) & (scores['cc'] == color_id)]
                ax.hist(group['auc'], bins=bins,
                        color=palette[int(group['nstates'].iloc[0])], alpha=0.75)
            ax.tick_params(axis='y', left=False, labelleft=False)
            ax.grid(axis='y', linestyle='--', alpha=0.4)
        axes[-1].set_xlabel('ROC-AUC')
        axes[-1].set_xlim(0.5, 1.0)
        fig.tight_layout()
        scores.to_csv(summary_dir / 'auc_histogram_scores.csv', index=False)
        path = summary_dir / 'auc_histograms.png'
        fig.savefig(path, dpi=300)
    finally:
        plt.close(fig)
    print(f'Saved ROC-AUC histograms to {path}', flush=True)
    return path


def run_fixed_readout_sweep(lnorms, n_seeds, epochs, *, readout_scale=1.0):
    """Train/resume the fixed sweep and plot all runs without outlier filtering."""
    from scripts import constrained_penalty_sweep as sweep

    seeds = list(range(1, n_seeds + 1))
    settings = dict(sweep.SETTINGS, epochs=epochs, fixed_readout_norm=True)
    if readout_scale != 1.0:
        settings['readout_scale'] = float(readout_scale)
    sweep_id = sweep.configuration_id(dict(
        lnorms=list(lnorms), seeds=seeds, settings=settings, cache_version=1))
    rows = []
    matrices = {}
    lattice = get_task_lattice(TASK)
    for lnorm in lnorms:
        for seed in seeds:
            row = sweep.run_one(sweep.OUTPUT_DIR, lnorm, seed, epochs,
                                readout_norm='fixed', readout_scale=readout_scale)
            rows.append(row)
            path = sweep.OUTPUT_DIR / 'runs' / row['run_id'] / 'decoding_heatmap.npz'
            with np.load(path) as saved:
                matrices[row['run_id']] = saved['matrix'].copy()
            summary_dir = sweep.save_summary(sweep.OUTPUT_DIR, sweep_id, rows, lnorms)
        df = pd.DataFrame(rows).rename(columns={'run_id': 'run_key'})
        save_ncolors_stackedbar(df, matrices, lattice, lnorms=lnorms,
                                output_dir=summary_dir, readout_scale=readout_scale)
        print(f'Coloring summary updated after lnorm={lnorm:g}: {summary_dir}',
              flush=True)
    return summary_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seeds', type=int,
                        help='Seeds 1..N per lnorm (legacy: 100; fixed: 50).')
    parser.add_argument('--recompute', action='store_true',
                        help='Retrain even if cached runs exist.')
    parser.add_argument('--fixed-readout-sweep', action='store_true',
                        help='Run the fixed-readout sweep without outlier filtering.')
    parser.add_argument('--lnorms', type=float, nargs='+',
                        help='Fixed-readout lnorm values (default: 2 through 15); '
                             '0 means no activity penalty.')
    parser.add_argument('--epochs', type=int,
                        help='Fixed-readout training epochs (default: 2000).')
    parser.add_argument('--readout-scale', type=float,
                        help='Fixed readout norm relative to initialization (default: 1).')
    parser.add_argument('--curated-summary', type=Path,
                        help='Replot saved ROC-AUC winners for lnorm 1..6 without training.')
    parser.add_argument('--auc-histogram-summary', type=Path,
                        help='Plot coloring AUC histograms from cached heatmaps.')
    parser.add_argument('--accuracy-summary', type=Path,
                        help='Plot saved train and held-out accuracies for lnorm 1..6.')
    args = parser.parse_args()

    if args.accuracy_summary is not None:
        if (args.curated_summary is not None or args.auc_histogram_summary is not None
                or args.fixed_readout_sweep or args.recompute or args.seeds is not None
                or args.lnorms is not None or args.epochs is not None
                or args.readout_scale is not None):
            parser.error('--accuracy-summary cannot be combined with other options')
        save_cached_accuracy_boxplot(args.accuracy_summary)
        return
    if args.auc_histogram_summary is not None:
        if (args.curated_summary is not None or args.fixed_readout_sweep
                or args.recompute or args.seeds is not None or args.lnorms is not None
                or args.epochs is not None or args.readout_scale is not None):
            parser.error('--auc-histogram-summary cannot be combined with other options')
        lattice_path = DIAGRAMS_DIR / 'limitedstack_Ldiag6_lattice.pkl'
        if not lattice_path.is_file():
            raise FileNotFoundError(f'Cached coloring lattice required: {lattice_path}')
        from scripts.constrained_penalty_sweep import OUTPUT_DIR
        save_auc_histograms(args.auc_histogram_summary, OUTPUT_DIR / 'runs',
                            get_task_lattice(TASK))
        return
    if args.curated_summary is not None:
        if (args.fixed_readout_sweep or args.recompute or args.seeds is not None
                or args.lnorms is not None or args.epochs is not None
                or args.readout_scale is not None):
            parser.error('--curated-summary cannot be combined with training options')
        save_curated_stackedbar(args.curated_summary)
        return
    if args.fixed_readout_sweep:
        lnorms = args.lnorms if args.lnorms is not None else [
            float(value) for value in range(2, 16)]
        n_seeds = args.seeds if args.seeds is not None else 50
        epochs = args.epochs if args.epochs is not None else 2000
        readout_scale = args.readout_scale if args.readout_scale is not None else 1.0
        if (any(not np.isfinite(x) or x < 0 for x in lnorms)
                or len(set(lnorms)) != len(lnorms)):
            parser.error('lnorm values must be distinct, finite, and nonnegative')
        if n_seeds < 1 or epochs < 1:
            parser.error('seeds and epochs must be positive')
        if not np.isfinite(readout_scale) or readout_scale <= 0:
            parser.error('readout scale must be finite and positive')
        if args.recompute:
            parser.error('--recompute is not supported for the fixed-readout sweep; '
                         'existing runs are resumed')
        run_fixed_readout_sweep(lnorms, n_seeds, epochs, readout_scale=readout_scale)
        return
    if args.lnorms is not None or args.epochs is not None or args.readout_scale is not None:
        parser.error('--lnorms, --epochs and --readout-scale require --fixed-readout-sweep')
    n_seeds = args.seeds if args.seeds is not None else N_SEEDS
    if n_seeds < 1:
        parser.error('seeds must be positive')
    cd = get_task_diagram(TASK)

    # run (or load) the sweep; resumable at (lnorm, seed) granularity
    rows, matrices = [], {}
    for lnorm in LNORMS:
        for seed in range(1, n_seeds + 1):
            row, matrix = constrained_run(TASK, seed, lnorm, train_L=TRAIN_L,
                                          hidden_dim=HIDDEN_DIM, epochs=EPOCHS,
                                          recompute=args.recompute)
            rows.append(row)
            matrices[row['run_key']] = matrix
            save_heatmap_png(row['run_key'], matrix, cd.state_labels)
    df = pd.DataFrame(rows)

    df_filtered = remove_outliers(df)
    print(f'Outlier filter: kept {len(df_filtered)}/{len(df)} runs')
    save_acc_boxplot(df_filtered)

    # lattice of consistent colorings for the winner-take-all analysis
    # (cached in results/diagrams/ -- built once per task)
    lattice = get_task_lattice(TASK)
    save_ncolors_stackedbar(df_filtered, matrices, lattice)


if __name__ == '__main__':
    main()
