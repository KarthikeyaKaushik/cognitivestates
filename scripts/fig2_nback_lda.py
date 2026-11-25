"""Figure 1: hidden-state geometry of a constrained RNN trained on 2-back.

  1. Infer the minimal cognitive diagram for 2-back (k=2) from short sequences.
  2. Train a ConstrainedRNN (CE loss + (1/lnorm)*||h|| penalty) on all 2-back
     sequences of length L_TRAIN.
  3. Collect hidden states, balance diagram-state classes, LDA-project to 2D,
     and scatter-plot colored by the diagram state.

Usage:
  python scripts/fig1_nback_lda.py              # load cached run if available
  python scripts/fig1_nback_lda.py --recompute  # retrain and overwrite cache

Output: figures/fig1_nback_lda.png
Cache:  results/hidden_states/<run_key>.pt (training is the expensive step)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import colorcet as cc
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from common import save_figure
from src.diagrams import NBACK_STATE_LABELS, build_vocab, label_order, load_or_infer_diagram
from src.nback import generate_nback_sequences
from src.paths import HIDDEN_STATES_DIR, ensure_dir
from src.projections import get_lda_projection
from src.training import (ConstrainedRNN, DataGenerator, balance_classes,
                          flatten_states, get_hidden_states, train_model)

# ---------------- parameters ----------------
N = 2                 # n-back distance
K = 2                 # number of stimuli
L_DIAGRAM = 5         # sequence length for diagram inference (veelenturf)
L_TRAIN = 10          # sequence length for RNN training
HIDDEN_DIM = 8
LNORM = 100      # activity penalty weight is 1/LNORM
EPOCHS = 1000
SEED = 0
LEARNING_RATE = 0.001
BATCH_SIZE = 128
JITTER_SD = 0.5       # visual jitter; states collapse to near-points otherwise
PLOT_SPLIT = 'test'   # 'test' = held-out data; 'train' = legacy paper behavior

RUN_KEY = f'nback_n{N}_k{K}_L{L_TRAIN}_h{HIDDEN_DIM}_lnorm{LNORM:g}_ep{EPOCHS}_seed{SEED}'
CACHE_PATH = HIDDEN_STATES_DIR / f'{RUN_KEY}.pt'
FIGURE_NAME = 'fig1_nback_lda'


def get_diagram():
    """Minimal 2-back cognitive diagram (cached per task parameters)."""
    diag_pairs = generate_nback_sequences(L=L_DIAGRAM, n=N, k=K)
    X, Y = build_vocab(diag_pairs)
    return load_or_infer_diagram(f'nback_n{N}_k{K}_Ldiag{L_DIAGRAM}',
                                 diag_pairs, X, Y, state_labels=NBACK_STATE_LABELS)


def compute():
    """Train the RNN and cache hidden states + diagram-state truths."""
    cd = get_diagram()

    # Same RNG discipline as the legacy trainer: ambient torch seed 0 set
    # before the DataGenerator split; model built afterwards (seed=0 means the
    # model keeps consuming the ambient stream).
    torch.manual_seed(0)
    io_pairs = generate_nback_sequences(L=L_TRAIN, n=N, k=K)
    print(f'Training dataset: {len(io_pairs)} sequences of length {L_TRAIN}')
    datagenerator = DataGenerator(data=io_pairs, cog_diagram=cd, batch_size=BATCH_SIZE)
    model = ConstrainedRNN(input_dim=len(cd.X), hidden_dim=HIDDEN_DIM,
                           output_dim=len(cd.Y), seed=SEED, lnorm=LNORM)
    model, track_loss = train_model(model, datagenerator, LEARNING_RATE, EPOCHS,
                                    loss_tracking_epochs=100)

    cache = {'state_labels': cd.state_labels,
             'final_loss': track_loss['loss'][-1],
             'final_acc': track_loss['acc'][-1]}
    for split, is_train in [('train', True), ('test', False)]:
        hidden, truths, _, _ = get_hidden_states(model, datagenerator, train=is_train)
        # Drop the last timestep (legacy alignment with next-step inputs/outputs).
        cache[f'hidden_{split}'] = hidden[:, :-1, :].detach()
        cache[f'truths_{split}'] = truths[:, :-1, :].detach()

    ensure_dir(HIDDEN_STATES_DIR)
    torch.save(cache, CACHE_PATH)
    print(f'Cached hidden states to {CACHE_PATH}')


def plot():
    cached = torch.load(CACHE_PATH, weights_only=False)
    # Project held-out test data (the legacy paper figure used the train split;
    # set PLOT_SPLIT='train' to reproduce that exactly).
    hidden = cached[f'hidden_{PLOT_SPLIT}']
    truths = cached[f'truths_{PLOT_SPLIT}']
    state_labels = cached['state_labels']
    print(f"Run {RUN_KEY} [{PLOT_SPLIT} split]: final train loss "
          f"{cached['final_loss']:.4f}, final train acc {cached['final_acc']:.4f}")

    predictor, target = flatten_states(hidden, truths)
    predictor, target, _ = balance_classes(predictor, target)
    pos_lda = get_lda_projection(predictor, target)

    pos_2d = pd.DataFrame(pos_lda, columns=['x', 'y'])
    pos_2d['states'] = target.numpy()
    pos_2d['states'] = pos_2d['states'].astype('category').map(state_labels)
    desired_order = [state_labels[k] for k in sorted(state_labels)]
    pos_2d['states'] = pd.Categorical(pos_2d['states'], categories=desired_order, ordered=True)

    # Jitter for visibility (legacy behavior for nback; seeded here so the
    # figure is reproducible -- the legacy jitter was unseeded).
    rng = np.random.default_rng(SEED)
    pos_2d[['x', 'y']] += rng.normal(0, JITTER_SD, pos_2d[['x', 'y']].shape)

    sns.scatterplot(data=pos_2d, x='x', y='y', hue='states',
                    palette=cc.glasbey_bw, alpha=0.5)
    import matplotlib.pyplot as plt
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    save_figure(FIGURE_NAME)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--recompute', action='store_true',
                        help='Retrain the RNN even if a cached run exists.')
    args = parser.parse_args()

    if args.recompute or not CACHE_PATH.exists():
        compute()
    else:
        print(f'Using cached run {CACHE_PATH} (pass --recompute to retrain).')
    plot()


if __name__ == '__main__':
    main()
