# Figure 2: held-out diagram-state decoding

From the project root, using the `cognitivestates` environment:

```bash
python scripts/fig3_task_panels.py --seeds 5
python scripts/fig3_task_panels.py --history-decoding --seeds 5
python scripts/fig3_task_panels.py --history-decoding --tasks listmax --seeds 1
```

Both modes save two accuracy plots per task:

- Main: `figures/<task>_simulations.png`, ordered **LDA, PCA, ICA, FA**.
- Supplementary: `figures/<task>_supplementary.png`, ordered **LDA,
  history-LDA k=1, k=2, k=3** (listmax omits k=3).

Both use automatically determined y-axis limits.
Boxplots show the distribution across seeds; dots show individual seed scores,
including when only one seed is requested. Unavailable conditions are labeled.

The default command also generates the existing scatter panels.
`--history-decoding` generates **only the two accuracy plots**, not a
history-only plot. It reuses both scalar tables where possible and computes
any missing original/control or history results. If training is needed, both
analyses share one run per seed before non-scatter tensors are discarded;
`--recompute` likewise retrains only once per seed.

## Shared train/test protocol

The existing RNN split is retained: 80% of whole sequences for training,
20% for testing. Timesteps of a sequence never occur in both splits.
Network hyperparameters and the existing final-timestep trim are unchanged.

All projections, preprocessing scalers, and pairwise linear SVMs are fitted
using **training sequences only**. They are then applied without refitting
to test sequences. Class balancing downsamples each represented diagram state
to the rarest state independently within each split. The reported accuracy
is the mean test accuracy over all unordered pairs of diagram-state labels.
Even perfectly separable training pairs are evaluated on test examples.
There is no training-accuracy shortcut.

States absent from either split are omitted from the pair comparisons, with
their IDs printed to the console. A training state missing from test data is
still retained when fitting the projection and scaler: test-class availability
does not determine the fitted training representation.
Scatter-only filters (full-history n-back
states and non-error stack states) do not restrict quantitative scores.
Scatter panels also fit projections on training data and show transformed
test data. Jitter remains display-only.

## Original analysis

LDA uses cognitive-diagram state labels for its training projection.
PCA, ICA, and FA are unsupervised, fitted to the same balanced training
activations. PCA uses the leading covariance directions; ICA and FA retain
their training-only standardization. All four methods decode original
diagram-state labels on held-out data.

The shuffled-LDA control independently permutes the state labels within each
split; LDA and SVMs fit the shuffled training labels, and scoring uses the
shuffled test labels. It is not a random-projection baseline.
These controls are still computed and saved, but are not plotted. Their
per-task, per-seed accuracies are rows with `method=LDA_shuffled` in
`results/decoding_accuracies_heldout.csv`. Use those measured values when
reporting the controls; 50% is the balanced binary chance level, not a
hard-coded result.

Outputs retain the existing names: `figures/{lda,pca}_<task>.png`,
`figures/lda_untrained_<task>.png`, and `figures/<task>_simulations.png`.
The supplementary accuracy plot is `figures/<task>_supplementary.png`.
Regeneration replaces these images. Separate `<task>_lda_shuffle.png` and
`<task>_lda_history_accuracy.png` accuracy plots are no longer generated;
any existing copies are left untouched.

## Supplement: history-supervised LDA, state-label classification

For `k=1,2,3` (**listmax: `k=1,2` only**), pair each post-input activation `h_t` with the ordered input
window `(x_{t-k+1}, ..., x_t)`. Only complete windows are included; windows
never cross sequence boundaries.

1. Downsample **training history classes** to equal counts and fit LDA with
   the history labels. Use up to two axes; binary `k=1` permits only one.
2. Transform all complete-window training and test activations with that
   fitted LDA. Test history labels are not used in fitting or scoring.
3. Balance these projected samples by their **original diagram-state labels**,
   independently in each split. Fit a linear SVM for every state pair on
   training samples, and evaluate it on test samples.

This asks how well a projection optimized for recent input histories preserves
task-state information. It does **not** decode histories or predict task outputs.
Each history condition appears after LDA in
`figures/<task>_supplementary.png`. Splitting the plots does not
change the scores, labels, or train/test protocol.

Complete windows remove early transient states at larger `k`; pairs involving
those states cannot be measured and are excluded with a console message.
Consequently, the evaluated state set can differ across `k`.

### Why listmax stops at k=2

After the final-timestep trim, `k=3` uses only the activation after the first
three inputs. Repeated occurrences of the same three-input history produce
identical activations in the deterministic RNN, giving zero within-class
variation. SVD-based LDA cannot estimate this projection, even with multiple
training samples per history. Listmax therefore computes and plots only
`k=1,2`, as configured by `history_max_k` in the task registry. Other tasks
retain `k=1,2,3`. Cache completion checks use the same task-specific limit.

If training-history balancing still leaves fewer than two observations per
class, the code warns and records `status=insufficient_samples`, a missing
accuracy, and a reason. It does not silently drop rare history classes.
The same applies when fewer than two diagram states are shared by both splits.

## Storage and migration

Corrected results use **new CSVs**, so old in-sample results are not silently
reused or overwritten:

- `results/decoding_accuracies_heldout.csv`
- `results/history_state_decoding_heldout.csv`

Only the scatter seed keeps hidden tensors plus aligned input IDs. Existing
hidden train/test tensors can be rescored for the original analysis without
retraining. The supplement needs aligned inputs in **both** splits; old caches
without them, and seeds whose tensors were discarded, require one retraining
pass. The script prints when backfill is needed. It does not infer the old
randomized extraction order. `--recompute` forces retraining.

The new history CSV records history counts used to fit LDA, actual projection
dimension, state-class counts, classifier sample counts, status, and reason.
Both analyses resume only when all expected rows for a run are present.
The previous `decoding_accuracies.csv` and `history_decoding_accuracies.csv`
are left untouched. Figure 1 and constrained Figure 3 analyses are unchanged.

## Focused tests (no task training)

```bash
python -m unittest discover -s tests
```

Numerical projection/history tests require only the existing NumPy/scikit-learn
dependencies; experiment-storage and tensor-alignment tests also need PyTorch
and the project's remaining imports.
Plotting tests use a noninteractive backend and temporary output files;
no task training is performed.

## Standalone: constrained stack network's behavioral state diagram

```bash
python scripts/constrained_stack_behavior.py
python scripts/constrained_stack_behavior.py --seed 2
python scripts/constrained_stack_behavior.py --retrain
```

Trains one network using Figure 3's maximum activity penalty: `lnorm=0.6`
(penalty weight `1/0.6`), hidden size 5, training length 7, 1000 epochs,
learning rate 0.001, and batch size 128. The default seed is 1. The original
80/20 sequence split and training procedure are unchanged. Existing Figure 3
results contain no weights, so this analysis trains its own network once and
reuses its saved checkpoint on subsequent runs. `--retrain` replaces it.

Evaluates **all 729 length-6 inputs** used for the full stack diagram, starting
each sequence from the RNN's usual initial state. Every timestep is retained;
there is no train/test filtering or final-timestep trimming. Each output is
the argmax of the network's readout logits, decoded with the training output
vocabulary. Veelenturf receives these **network predictions**, not task targets.
The resulting diagram is checked against every recorded network output.

Outputs are kept together under
`results/network_behavior/limitedstack_constrained_L7_h5_lnorm0.6_ep1000_seed1/`
(with the requested seed):

- `network.pt`: trained state dictionary, settings, vocabulary order, completed
  epoch count, and training history.
- `observations.json`: ordered inputs, network outputs, task targets for
  comparison only, metadata, and timestep/whole-sequence task accuracy.
- `diagram.json`: states, initial state, input/output-labeled transitions,
  unobserved transitions, and verification count.
- `diagram.pkl`: inferred `CognitiveDiagram`, without stack-content labels.
- `diagram.png`, `diagram.svg`, `diagram.pdf`: force-directed state diagram,
  with input-colored arrows, output labels, unlabeled circular states, and a
  diamond for the initial state. Parallel arrows and self-loops are retained.
- `diagram.dot`: editable Graphviz source. Existing TikZ files are left untouched;
  new analyses no longer generate them.

Rendering requires Graphviz's `neato` executable (`brew install graphviz` on
macOS). Redraw an existing result **without training or diagram inference**:

```bash
python scripts/plot_behavior_diagram.py \
  results/network_behavior/limitedstack_constrained_L7_h5_lnorm0.6_ep1000_seed1/diagram.json
```

This plot-only command uses only the Python standard library and Graphviz.
The force-directed layout has a fixed random seed for reproducibility.

Observations are saved before inference, which can be expensive for complex
network behavior. Numeric state IDs do not inherit the full stack's semantic
labels. This is a finite-data behavioral diagram: exact agreement on these
inputs does **not** establish equivalence to the RNN on longer sequences, and
transitions unsupported by the observations may remain undefined.

Focused tests (no real training):

```bash
python -m unittest discover -s tests -p 'test_constrained_stack_behavior.py'
python -m unittest discover -s tests -p 'test_plot_behavior_diagram.py'
```
