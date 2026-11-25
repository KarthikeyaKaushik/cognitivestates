"""Figure 4: cross-entropy of consistent vs. random colorings, with stats.

Two plots per task (limitedstack and nback3):

  figures/coloring_size_vs_entropy_<task>.png
      jittered scatter of per-coloring cross-entropy vs. partition size
      (consistent colorings in color, random partitions in gray)
  figures/length_vs_entropy_<task>.png
      cross-entropy vs. sequence length at a fixed coloring size: consistent
      per-length mean vs. random mean with +-std and min/max bands

Reported statistics (results/stats_fig4_<task>.csv):
  * entropy_vs_random -- per coloring size, each consistent coloring compared
    against the random-partition null: empirical (permutation) p-value with
    its resolution floor, z-score effect size, percentile; Holm-corrected
    across coloring sizes. The null is enumerated exactly when the partition
    space is small, otherwise sampled (--max-samples, default 999).
  * slope_vs_random -- per-coloring regression slopes of entropy vs. length;
    each consistent coloring's slope compared against the distribution of
    random-coloring slopes (same effect-size/p machinery).

NOTE: two flagged corrections vs. the legacy figures -- the belief-start-state
fix and the per-length (not running-mean) consistent values; see
src/coloring_analysis.py. The nback analysis uses the hand-coded simplified
8-state 3-back machine, as in the legacy code.

Usage:
  python scripts/fig4_coloring_entropy.py --task limitedstack
  python scripts/fig4_coloring_entropy.py --task nback3
  python scripts/fig4_coloring_entropy.py              # both tasks

Results are cached in results/entropies_*.csv; the coloring-size sweep is
resumable per size. --recompute discards the CSVs and recomputes.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import colorcet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import save_figure
from src.coloring_analysis import (coloring_size_entropies,
                                   get_simplified_nback3_lattice, length_sweep)
from src.experiments import get_task_lattice
from src.limitedstack import generate_sequences as limitedstack_sequences
from src.limitedstack import random_sequence as limitedstack_random_sequence
from src.nback import generate_nback_sequences, nback_output
from src.paths import RESULTS_DIR
from src.stats import empirical_comparison, holm_correction

# ---------------- parameters (legacy SIMULATE blocks) ----------------
SIZE_SWEEP_IO_LENGTH = 20   # sequence length for the coloring-size sweep
SIZE_SWEEP_NUM_PAIRS = 2000
MAX_RANDOM_SAMPLES = 999    # sampled null size (legacy: 100; exact when small)
LENGTH_SWEEP_MAX = 10       # sweep lengths 2..10


def sample_limitedstack_pairs(io_length, num_pairs):
    """Legacy sampler: full enumeration for short lengths, otherwise random
    'contentful' sequences (at most 25% error outputs)."""
    if io_length <= 10:
        return random.sample(limitedstack_sequences(io_length), k=num_pairs)
    pairs = []
    while len(pairs) < num_pairs:
        seq = limitedstack_random_sequence(io_length)
        if sum(x == 'e' for x in seq[1]) > len(seq[1]) // 4:
            continue
        pairs.append(seq)
    return pairs


def sample_nback3_pairs(io_length, num_pairs):
    """Sample random 3-back sequences directly (no 2^L enumeration)."""
    pairs = []
    for _ in range(num_pairs):
        seq = tuple(random.choice('ab') for _ in range(io_length))
        pairs.append((seq, nback_output(seq, 3)))
    return pairs


def limitedstack_setup():
    lattice = get_task_lattice('limitedstack')
    return lattice.cd, lattice


TASK_CONFIGS = {
    'limitedstack': dict(
        setup=limitedstack_setup,
        coloring_sizes=range(5, 16),
        sample_pairs=sample_limitedstack_pairs,
        length_sweep_size=5,          # legacy run_limitedstack call
        enumerate_pairs=limitedstack_sequences,
        # 3^L sequences explode; the legacy stack length sweep never actually
        # ran (commented out), so we subsample (flagged deviation)
        length_sweep_num_pairs=1000,
    ),
    'nback3': dict(
        setup=get_simplified_nback3_lattice,
        coloring_sizes=range(2, 9),
        sample_pairs=sample_nback3_pairs,
        length_sweep_size=2,          # legacy run_nback call
        enumerate_pairs=lambda L: generate_nback_sequences(L=L, n=3, k=2),
        length_sweep_num_pairs=None,  # full enumeration (2^L), as legacy
    ),
}


# ---------------- plotting (legacy viz_7 / viz_6 styling) ----------------

def plot_coloring_size(df, task):
    if task == 'nback3':
        df = df[df['coloring_size'] <= 7]
    fig, ax = plt.subplots(figsize=(5, 5))
    palette = colorcet.glasbey[:16]
    color_map = {'consistent': palette[3], 'random': '#9A9498'}
    rng = np.random.default_rng(0)
    jitter, dodge = 0.06, 0.10

    for kind, zorder in [('random', 1), ('consistent', 2)]:
        sub = df[df['coloring'] == kind]
        x = (sub['coloring_size'].astype(float).values + dodge
             + rng.uniform(-jitter, jitter, size=len(sub)))
        ax.scatter(x, sub['entropy'].values, color=color_map[kind], s=40,
                   alpha=0.7, edgecolor='none', linewidth=0.25, zorder=zorder)

    handles = [Line2D([0], [0], marker='o', color='w', label='Consistent coloring',
                      markerfacecolor=color_map['consistent'], markersize=8),
               Line2D([0], [0], marker='o', color='w', label='Random coloring',
                      markerfacecolor=color_map['random'], markersize=8)]
    ax.legend(handles=handles, loc='upper right', frameon=False)

    sizes = np.sort(df['coloring_size'].unique().astype(float))
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(int(s)) for s in sizes], fontsize=12)
    ax.set_xlabel('Coloring Size', fontsize=16)
    ax.set_ylabel('Cross entropy', fontsize=16)
    save_figure(f'coloring_size_vs_entropy_{task}')
    plt.close(fig)


def plot_length(df, task):
    """Summary plot from the long-format length sweep: per-length consistent
    mean vs. random mean with +-std and min/max bands."""
    cons = df[df['kind'] == 'consistent'].groupby('length')['entropy'].mean()
    rand = df[df['kind'] == 'random'].groupby('length')['entropy']
    summary = pd.DataFrame({'cc': cons, 'rc_mean': rand.mean(),
                            'rc_std': rand.std(), 'rc_min': rand.min(),
                            'rc_max': rand.max()}).reset_index()

    fig, ax = plt.subplots(figsize=(5, 5))
    palette = colorcet.glasbey[:16]
    color_map = {'cc': palette[3], 'rand': '#9A9498'}

    ax.plot(summary['length'], summary['cc'], color=color_map['cc'],
            label='Consistent Coloring', linewidth=2)
    ax.fill_between(summary['length'], summary['rc_min'], summary['rc_max'],
                    color=color_map['rand'], alpha=0.2, label='Random Min/Max')
    ax.fill_between(summary['length'], summary['rc_mean'] - summary['rc_std'],
                    summary['rc_mean'] + summary['rc_std'],
                    color=color_map['rand'], alpha=0.5, label='Random Mean ± Std')
    ax.plot(summary['length'], summary['rc_mean'], color=color_map['rand'],
            label='Random Mean', linewidth=2)
    ax.set_xlabel('Input sequence length', fontsize=16)
    ax.set_ylabel('Cross entropy', fontsize=16)
    ax.legend()
    save_figure(f'length_vs_entropy_{task}')
    plt.close(fig)


# ---------------- statistics ----------------

def compute_stats(task, size_df, length_df):
    """Permutation p-values + effect sizes for both analyses, Holm-corrected
    within each analysis. Written to results/stats_fig4_<task>.csv."""
    rows = []

    # (1) per-coloring-size: consistent entropy vs random-partition null
    for size, grp in size_df.groupby('coloring_size'):
        null = grp.loc[grp['coloring'] == 'random', 'entropy'].values
        cons = grp.loc[grp['coloring'] == 'consistent', 'entropy'].values
        if len(null) < 2 or len(cons) == 0:
            continue
        for i, val in enumerate(cons):
            r = empirical_comparison(val, null, alternative='less')
            rows.append(dict(task=task, analysis='entropy_vs_random',
                             coloring_size=size, unit=f'cc{i}',
                             value=float(val), **r))

    # (2) per-coloring slopes (entropy vs length), consistent vs random null
    slopes = (length_df.groupby(['kind', 'coloring_id'])
              .apply(lambda g: np.polyfit(g['length'], g['entropy'], 1)[0])
              .rename('slope').reset_index())
    null_slopes = slopes.loc[slopes['kind'] == 'random', 'slope'].values
    cons_slopes = slopes[slopes['kind'] == 'consistent']
    if len(null_slopes) >= 2:
        for _, srow in cons_slopes.iterrows():
            r = empirical_comparison(srow['slope'], null_slopes, alternative='less')
            rows.append(dict(task=task, analysis='slope_vs_random',
                             coloring_size=np.nan, unit=srow['coloring_id'],
                             value=float(srow['slope']), **r))

    stats = pd.DataFrame(rows)
    for analysis, grp in stats.groupby('analysis'):
        stats.loc[grp.index, 'holm_p'] = holm_correction(grp['empirical_p'].values)
    out_path = RESULTS_DIR / f'stats_fig4_{task}.csv'
    stats.to_csv(out_path, index=False)
    print(f'\n=== stats [{task}] (saved to {out_path}) ===')
    with pd.option_context('display.width', 150, 'display.max_columns', None):
        print(stats.round(4).to_string(index=False))
    return stats


# ---------------- sweeps with CSV caching ----------------

def run_size_sweep(task, cfg, max_samples, recompute=False):
    csv_path = RESULTS_DIR / f'entropies_coloring_size_{task}.csv'
    df = pd.DataFrame()
    if csv_path.exists() and not recompute:
        df = pd.read_csv(csv_path)
    cd, lattice = cfg['setup']()
    for size in cfg['coloring_sizes']:
        if len(df) > 0 and size in df['coloring_size'].values:
            continue  # resumable: size already computed
        print(f'[{task}] coloring-size sweep: size {size}')
        io_pairs = cfg['sample_pairs'](SIZE_SWEEP_IO_LENGTH, SIZE_SWEEP_NUM_PAIRS)
        rows = coloring_size_entropies(cd, lattice, size, io_pairs, max_samples)
        df = pd.concat([df, rows], ignore_index=True)
        df.to_csv(csv_path, index=False)  # checkpoint after every size
    return df


def run_length_sweep(task, cfg, max_samples, recompute=False):
    csv_path = RESULTS_DIR / f'entropies_length_{task}.csv'
    if csv_path.exists() and not recompute:
        df = pd.read_csv(csv_path)
        if 'coloring_id' in df.columns:  # guard against the old schema
            return df
    cd, lattice = cfg['setup']()
    print(f'[{task}] length sweep: coloring size {cfg["length_sweep_size"]}')
    df = length_sweep(cd, lattice, cfg['length_sweep_size'], LENGTH_SWEEP_MAX,
                      cfg['enumerate_pairs'], max_samples,
                      num_io_pairs=cfg['length_sweep_num_pairs'])
    df.to_csv(csv_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--task', choices=sorted(TASK_CONFIGS),
                        help='Run one task (default: both).')
    parser.add_argument('--max-samples', type=int, default=MAX_RANDOM_SAMPLES,
                        help='Random colorings in the sampled null.')
    parser.add_argument('--recompute', action='store_true',
                        help='Discard cached CSVs and recompute.')
    args = parser.parse_args()

    random.seed(42)  # legacy module-level seeding
    tasks = [args.task] if args.task else sorted(TASK_CONFIGS)
    for task in tasks:
        cfg = TASK_CONFIGS[task]
        size_df = run_size_sweep(task, cfg, args.max_samples,
                                 recompute=args.recompute)
        plot_coloring_size(size_df, task)
        length_df = run_length_sweep(task, cfg, args.max_samples,
                                     recompute=args.recompute)
        plot_length(length_df, task)
        compute_stats(task, size_df, length_df)


if __name__ == '__main__':
    main()
