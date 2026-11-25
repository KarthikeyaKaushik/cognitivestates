# Neural state structure from behavior

This repository contains code to train small recurrent neural networks on commonly studied tasks in cognitive psychology. The goal here is to compare the hidden state activations of RNNs with the theoretically derived state structures using ideas from early computer science (Moore, 1956). The upshot of analyzing internal representations this way is that instead of guessing the relevant representational/encoding variables, we can use the task specification to derive the minimally necessary state structure, and then use it as a supervisory signal to interpret neural activity. This approach leads to a set of striking results that (1) have more explanatory power than traditional neuroscience techniques such as PCA, ICA, etc. (2) demonstrates effectiveness at interpreting infinite state structures, (3) show how physical constraints can induce homomorphisms of the state diagram, and (4) show how homomorphisms appear over the course of learning. 


## Layout

```
src/       task generators, diagram inference, colorings lattice, models, training, decoding
scripts/   one entry point per figure / analysis (run from the repo root)
tests/     unittest suite
docs/      extra notes on the held-out decoding protocol (docs/README.md)
```

Scripts write cached runs to `results/` and figures to `figures/`. Both
directories are created on demand and are not tracked. Rerunning a script
reuses matching caches; most scripts accept `--recompute` (or `--retrain`)
to force retraining.

## Installation

```bash
conda create -n cognitivestates python=3.12
conda activate cognitivestates
pip install -r requirements.txt
```

`scripts/plot_behavior_diagram.py` (and the behavioral-diagram outputs of
`constrained_stack_behavior.py`) also need [Graphviz](https://graphviz.org/)
(`brew install graphviz` or `apt install graphviz`).

## Tasks

`nback`, `listmax`, `limitedset`, `limitedlist`, `limitedstack` (main tasks),
plus `addk`, `anbn`, `dyck`, `gridworld`, `majority_cfg`, `rescorlawagner`
(see `src/tasks.py` for the registry and per-task settings).

## Reproducing the analyses

Run every command from the repository root.

| Script | What it produces |
| --- | --- |
| `scripts/fig2_nback_lda.py` | 2-back diagram and LDA hidden-state geometry |
| `scripts/fig3_task_panels.py` | LDA/PCA scatter panels and held-out state-decoding boxplots for the five main tasks (see `docs/README.md`) |
| `scripts/fig4_coloring_entropy.py` | Cross-entropy of consistent vs. random colorings |
| `scripts/fig4_infinite.py` | Infinite-state examples (majority CFG, Rescorla–Wagner) |
| `scripts/fig5_constrained_stack.py` | Stack task under activity penalties: coloring-size stacked bar, ROC-AUC histograms, accuracy boxplots |
| `scripts/fig7_phase_transitions.py` | Consistent-coloring alignment across learning phase transitions |
| `scripts/fig_supp_linearitybaseline.py` | Pairwise separability of immediate vs. eventual state pairs, untrained vs. trained |
| `scripts/constrained_penalty_sweep.py` | Penalty sweep runner used by Fig. 5 |
| `scripts/readout_scale_experiment.py` | Fixed readout-norm scales without an activity penalty |
| `scripts/constrained_stack_behavior.py` | Behavioral-trace state inference for trained stack RNNs |
| `scripts/plot_behavior_diagram.py` | Renders a saved behavioral diagram (optionally a state subset) |

### Examples

```bash
python scripts/fig2_nback_lda.py
python scripts/fig3_task_panels.py --seeds 5
python scripts/fig3_task_panels.py --history-decoding --seeds 5
python scripts/fig4_coloring_entropy.py --task limitedstack
python scripts/fig4_infinite.py
python scripts/fig7_phase_transitions.py --seeds 5
python scripts/fig_supp_linearitybaseline.py
```

### Activity-penalty sweep (Fig. 5)

Train the sweep: lnorm 1–6, 100 seeds, readout norm fixed at 4× its
initialization, and 2,000 epochs. This is the expensive step:

```bash
python scripts/fig5_constrained_stack.py --fixed-readout-sweep \
    --readout-scale 4 --lnorms 1 2 3 4 5 6 --seeds 100 --epochs 2000
```

This prints a summary directory, `results/constrained_penalty_sweep/summaries/<id>`.
Pairwise state decodability uses balanced LDA accuracy, so unequal state
visitation cannot inflate it (chance is 0.5 for every pair). The sweep also
draws the curated stacked bar automatically, excluding (and printing) every
run whose winning coloring uses 15 colors. The following commands replot from
the cache without retraining:

```bash
SUMMARY=results/constrained_penalty_sweep/summaries/<id>
python scripts/fig5_constrained_stack.py --curated-summary $SUMMARY        # curated stacked bar
python scripts/fig5_constrained_stack.py --auc-histogram-summary $SUMMARY  # ROC-AUC histograms
python scripts/fig5_constrained_stack.py --accuracy-summary $SUMMARY       # train/held-out boxplots
```

Run behavioral state inference on selected networks from the sweep:

```bash
python scripts/constrained_stack_behavior.py --sweep-summary $SUMMARY \
    --run 2 18 --run 2 31 --run 3 48
```

## Tests

```bash
MPLBACKEND=Agg python -m unittest discover -s tests
```

## License

MIT (see `LICENSE`).
