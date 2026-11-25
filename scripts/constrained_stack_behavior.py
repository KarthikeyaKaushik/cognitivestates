"""Infer behavioral state diagrams for five independently initialized stack RNNs.

Usage:
  python scripts/constrained_stack_behavior.py
  python scripts/constrained_stack_behavior.py --seed 2
  python scripts/constrained_stack_behavior.py --seeds 1 2 3 4 5
  python scripts/constrained_stack_behavior.py --length 9
  python scripts/constrained_stack_behavior.py --retrain
  python scripts/constrained_stack_behavior.py --sweep-summary results/constrained_penalty_sweep/summaries/31ef8554536726b5 --run 2 18 --run 2 31 --run 3 48

Uses Figure 3's strongest activity penalty (lnorm=0.6), training length 7,
hidden size 5, and 1200 epochs. Saves and reuses the fully trained network.
Defaults to seeds 1 through 5 with identical settings and the same data split.
Each seed has its own checkpoint, inferred diagram, count CSVs, and plots.
--seed retains the single-network workflow. State IDs are local to each
inferred diagram; each subset uses that network's five most visited states.
Inference uses all 729 length-6 inputs used for the full stack diagram, but
replaces target outputs with the network's per-timestep argmax predictions.
--length sets both lengths for a separate experiment without changing the
reference diagram used by the training data generator.
If diagram.json and observations.json already exist for the run, replay those
instead of running the RNN or inferring the diagram again. Outputs include
transition_counts.csv, state_visit_counts.csv,
diagram.{dot,svg,png,pdf} with state IDs, a separate diagram for the five most
visited states (or all states if fewer than five), and a cumulative state-visit
plot in figures/. Visits include sequence starts and every transition arrival.
Each run also stores decoding_heatmap.npz and decoding_heatmap.png: full-vector
pairwise LDA training accuracies for reference task states, using the original
80% training split and all timesteps (as in the earlier constrained heatmaps).
These labels are reference stack states, not inferred behavioral state IDs.
--sweep-summary with one or more --run LNORM SEED pairs analyzes existing
fully trained sweep checkpoints without training or changing the sweep files.
It uses the usual length-6 inference traces; --length changes only the inference
length in this mode. Cached observations are tied to the checkpoint hash.
"""
import argparse
from collections import Counter
import csv
import hashlib
from itertools import accumulate
import json
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

from src.cognitive_diagram import veelenturf
from src.diagrams import build_vocab
from src.experiments import constrained_run_key, get_task_diagram
from src.paths import RESULTS_DIR, ensure_dir
from src.projections import get_pairwise_decodings
from src.tasks import TASKS
from src.training import (ConstrainedRNN, DataGenerator, flatten_states,
                          get_hidden_states, readout_weight_norm, train_model)
from scripts.plot_behavior_diagram import render_diagram
from scripts.common import save_figure

TASK = 'limitedstack'
SETTINGS = dict(lnorm=0.6, train_L=7, hidden_dim=5, epochs=1200,
                learning_rate=0.001, batch_size=128)


def load_checkpoint_network(checkpoint_path, metadata):
    """Load a completed network only if its checkpoint matches the metadata."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['metadata'] != metadata:
        raise ValueError(f'Checkpoint settings/vocabulary mismatch: {checkpoint_path}')
    if checkpoint['completed_epochs'] != metadata['epochs']:
        raise ValueError(f'Checkpoint is not fully trained: {checkpoint_path}')
    model = ConstrainedRNN(len(metadata['input_tokens']), metadata['hidden_dim'],
                           len(metadata['output_tokens']), seed=metadata['seed'],
                           lnorm=metadata['lnorm'])
    model.load_state_dict(checkpoint['state_dict'])
    if metadata.get('fixed_readout_norm', False):
        model.fc.weight_g.requires_grad_(False)
    model.eval()
    print(f'Loaded fully trained network from {checkpoint_path}')
    return model


def load_or_train_network(directory, seed, input_tokens, output_tokens,
                          retrain=False, *, settings=None):
    """Keep weights and vocabulary ordering together; never reuse mismatches."""
    settings = SETTINGS if settings is None else settings
    scale = settings.get('readout_scale', 1.0)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('readout scale must be finite and positive')
    if scale != 1.0 and not settings.get('fixed_readout_norm', False):
        raise ValueError('readout scaling requires fixed readout norms')
    checkpoint_path = directory / 'network.pt'
    metadata = dict(task=TASK, seed=seed, **settings,
                    input_tokens=input_tokens, output_tokens=output_tokens)
    if checkpoint_path.exists() and not retrain:
        model = load_checkpoint_network(checkpoint_path, metadata)
    else:
        cd = get_task_diagram(TASK)
        # Match Figure 3: split RNG is seeded before model initialization.
        torch.manual_seed(0)
        data = TASKS[TASK]['generate'](settings['train_L'])
        batches = DataGenerator(data, cd, batch_size=settings['batch_size'])
        model = ConstrainedRNN(len(input_tokens), settings['hidden_dim'],
                               len(output_tokens), seed=seed,
                               lnorm=settings['lnorm'])
        if settings.get('fixed_readout_norm', False):
            with torch.no_grad():
                model.fc.weight_g.mul_(scale)
            model.fc.weight_g.requires_grad_(False)
        initial_readout_norm = readout_weight_norm(model)
        print(f'Training seed={seed}, lnorm={settings["lnorm"]}, '
              f'penalty={model.multiplier:g} on {len(data)} sequences; '
              f'readout norm={initial_readout_norm:.4f} '
              f'({"fixed" if settings.get("fixed_readout_norm", False) else "learnable"})')
        model, history = train_model(
            model, batches, settings['learning_rate'], settings['epochs'],
            loss_tracking_epochs=100)
        ensure_dir(directory)
        torch.save(dict(metadata=metadata, state_dict=model.state_dict(),
                        training_history=history,
                        initial_readout_weight_norm=initial_readout_norm,
                        completed_epochs=settings['epochs']), checkpoint_path)
        print(f'Saved fully trained network to {checkpoint_path}')
    model.eval()
    return model, metadata


def collect_behavior(model, task_pairs, input_tokens, output_tokens,
                     batch_size=128):
    """Record every timestep, retaining generator order and all input sequences."""
    input_map = {token: i for i, token in enumerate(input_tokens)}
    observations = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(task_pairs), batch_size):
            batch = task_pairs[start:start + batch_size]
            indices = torch.tensor([[input_map[token] for token in inp]
                                    for inp, _ in batch])
            inputs = F.one_hot(indices, num_classes=len(input_tokens)).float()
            logits, _ = model(inputs)
            if not torch.isfinite(logits).all():
                raise ValueError('Network produced non-finite output logits')
            predictions = logits.argmax(dim=-1).tolist()
            for (inp, target), prediction in zip(batch, predictions):
                observations.append(dict(
                    inputs=list(inp), target_outputs=list(target),
                    network_outputs=[output_tokens[i] for i in prediction]))
    return observations


def save_decoding_heatmap(directory, metadata, model=None, *, recompute=False):
    """Cache and plot the earlier reference-state decoding analysis, without training."""
    cache = directory / 'decoding_heatmap.npz'
    if cache.exists() and not recompute:
        with np.load(cache) as saved:
            if json.loads(str(saved['metadata'].item())) != metadata:
                raise ValueError(f'Heatmap settings mismatch: {cache}')
            matrix, labels = saved['matrix'].copy(), saved['labels'].tolist()
    else:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            if model is None:
                if not (directory / 'network.pt').exists():
                    raise FileNotFoundError(
                        f'Cannot generate a missing heatmap without saved weights: {directory}')
                settings = {key: metadata[key] for key in SETTINGS}
                for key in ('fixed_readout_norm', 'readout_scale'):
                    if key in metadata:
                        settings[key] = metadata[key]
                model, _ = load_or_train_network(
                    directory, metadata['seed'], metadata['input_tokens'],
                    metadata['output_tokens'], settings=settings)
            cd = get_task_diagram(TASK)
            torch.manual_seed(0)
            batches = DataGenerator(
                TASKS[TASK]['generate'](metadata['train_L']), cd,
                batch_size=metadata['batch_size'])
            hidden, truths, _, _ = get_hidden_states(model, batches, train=True)
            predictor, target = flatten_states(hidden, truths)
            if not torch.equal(target.unique(), torch.arange(len(cd.states))):
                raise ValueError('Training samples do not cover every reference state.')
            matrix = get_pairwise_decodings(predictor, target)
            labels = [cd.state_labels[state] for state in range(len(cd.states))]
        ensure_dir(directory)
        np.savez_compressed(cache, matrix=matrix, labels=labels,
                            metadata=json.dumps(metadata, sort_keys=True))
    if not np.isfinite(matrix).all() or matrix.shape != (len(labels), len(labels)):
        raise ValueError(f'Invalid decoding matrix in {cache}')
    fig, ax = plt.subplots(figsize=(5, 5))
    try:
        sns.heatmap(matrix, vmin=0.5, vmax=1.0, ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.tick_params(axis='x', labelrotation=90)
        ax.tick_params(axis='y', labelrotation=0)
        fig.tight_layout()
        path = directory / 'decoding_heatmap.png'
        fig.savefig(path, dpi=300)
        print(f'Saved decoding heatmap to {path}')
    finally:
        plt.close(fig)


def infer_behavior_diagram(observations):
    """Use only network predictions as Veelenturf's output supervision."""
    pairs = [(tuple(row['inputs']), tuple(row['network_outputs']))
             for row in observations]
    X, Y = build_vocab(pairs)
    print(f'Inferring behavioral diagram from {len(pairs):,} traces', flush=True)
    cd = veelenturf(pairs, X, Y)
    for inp, expected in pairs:
        _, predicted = cd.apply_transitions(cd.start_state, inp)
        if predicted != expected:
            raise RuntimeError(
                f'Inferred diagram disagrees with network: {inp} -> '
                f'{predicted}, expected {expected}')
    print(f'Verified {len(cd.states)}-state diagram on all {len(pairs)} inputs')
    return cd


def count_transitions(machine, observations):
    """Replay each trace from the start state, checking each emitted output."""
    transitions = {}
    for edge in machine['transitions']:
        key = edge['state'], edge['input']
        if key in transitions:
            raise ValueError(f'Duplicate transition for state/input {key}')
        transitions[key] = edge
    counts = Counter()
    for sequence, row in enumerate(observations, start=1):
        state = machine['start_state']
        for timestep, (token, expected) in enumerate(
                zip(row['inputs'], row['network_outputs'], strict=True), start=1):
            key = state, token
            edge = transitions.get(key)
            if edge is None:
                raise ValueError(f'Sequence {sequence}, step {timestep}: no transition for {key}')
            if edge['output'] != expected:
                raise ValueError(f'Sequence {sequence}, step {timestep}: '
                                 f"diagram output {edge['output']!r} != saved output {expected!r}")
            counts[key] += 1
            state = edge['next_state']
    return [dict(edge, count=counts[key]) for key, edge in transitions.items()]


def plot_cumulative_visits(visits, run_name):
    """Plot cumulative visitation, with bar widths proportional to state frequency."""
    total = sum(visits.values())
    if total <= 0 or any(count < 0 for count in visits.values()):
        raise ValueError('Expected nonnegative state counts with a positive total.')

    states, counts = zip(*sorted(visits.items(), key=lambda row: (-row[1], row[0])))
    widths = [count / total for count in counts]
    shares = list(accumulate(widths))
    left = [0] + shares[:-1]
    centers = [x + w / 2 for x, w in zip(left, widths)]

    fig, ax = plt.subplots(figsize=(5, 5))
    try:
        bars = ax.bar(left, shares, width=widths, align='edge',
                      edgecolor='white', linewidth=0.5)

        # Fade bars after the first five
        for i, bar in enumerate(bars):
            if i >= 5:
                bar.set_alpha(0.25)

        # Replace the last six state labels with a single ellipsis
        n = max(0, len(states) - 6)
        ticks = list(centers[:n])
        labels = list(states[:n])

        if n < len(states):
            ticks.append((left[n] + shares[-1]) / 2)
            labels.append('...')

        ax.set_xticks(ticks, labels=labels, rotation=90, fontsize=8)
        ax.set_xlabel('State ID (most to least visited)')
        ax.set_ylabel('Cumulative share of visits')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(1))

        if len(states) >= 5:
            ax.axvline(shares[4], ls='--', color='gray', alpha=0.7)
            ax.axhline(shares[4], ls='--', color='gray', alpha=0.7)

        save_figure(f'state_visits_cumulative_{run_name}')
    finally:
        plt.close(fig)


def plot_subset_diagram(machine, directory, visits):
    """Draw this network's top five states; break equal counts by state ID."""
    selected = tuple(sorted(machine['states'], key=lambda state: (-visits[state], state))[:5])
    total = sum(visits.values())
    if total <= 0:
        raise ValueError('Cannot select frequent states without positive visitation counts.')
    share = sum(visits[state] for state in selected) / total
    print(f'Top states {selected} account for {share:.2%} of visits.')
    render_diagram(machine, directory, selected_states=selected)


def save_traversal_analysis(directory, machine, observations):
    edges = count_transitions(machine, observations)
    visits = Counter({state: 0 for state in machine['states']})
    visits[machine['start_state']] += len(observations)
    for edge in edges:
        visits[edge['next_state']] += edge['count']
    with (directory / 'transition_counts.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['state', 'input', 'output', 'next_state', 'count'])
        writer.writeheader()
        writer.writerows(edges)
    with (directory / 'state_visit_counts.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['state', 'visits'])
        writer.writerows(sorted(visits.items()))
    plot_cumulative_visits(visits, directory.name)
    render_diagram(machine, directory)
    plot_subset_diagram(machine, directory, visits)
    print(f"Counted {sum(edge['count'] for edge in edges):,} traversals of "
          f'{len(edges)} defined arrows across {len(observations):,} sequences.')


def save_analysis(directory, metadata, observations, *, diagram_length=None):
    """Persist observations before inference, then the verified finite-data machine."""
    ensure_dir(directory)
    correct = sum(pred == target for row in observations
                  for pred, target in zip(row['network_outputs'], row['target_outputs']))
    total = sum(len(row['inputs']) for row in observations)
    summary = dict(
        n_sequences=len(observations), n_timesteps=total,
        timestep_accuracy=correct / total,
        sequence_accuracy=sum(row['network_outputs'] == row['target_outputs']
                              for row in observations) / len(observations))
    if diagram_length is None:
        diagram_length = TASKS[TASK]['diagram_L']
    dataset = dict(metadata=metadata, diagram_L=diagram_length,
                   summary=summary, observations=observations)
    with open(directory / 'observations.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f'Saved network behavior to {directory / "observations.json"}')
    print(f'Network vs task: timestep accuracy={summary["timestep_accuracy"]:.4f}, '
          f'sequence accuracy={summary["sequence_accuracy"]:.4f}')
    cd = infer_behavior_diagram(observations)
    transitions = [
        dict(state=state, input=token, output=cd.outputtrans[(state, token)],
             next_state=next_state)
        for (state, token), next_state in sorted(cd.statetrans.items())]
    machine = dict(
        start_state=cd.start_state, states=sorted(cd.states),
        input_tokens=sorted(cd.X), output_tokens=sorted(cd.Y),
        transitions=transitions,
        unobserved_transitions=[
            dict(state=state, input=token)
            for state, token in sorted(cd.get_undefined_state_input())],
        verified_sequences=len(observations))
    with open(directory / 'diagram.json', 'w') as f:
        json.dump(machine, f, indent=2)
    with open(directory / 'diagram.pkl', 'wb') as f:
        pickle.dump(cd, f)
    save_traversal_analysis(directory, machine, observations)
    print(f'Saved verified diagram (JSON, pickle, force-directed graph) to {directory}')
    return cd


def run_network(seed, *, length=None, retrain=False):
    """Train or reuse one seed without mixing its cache with other networks."""
    cfg = TASKS[TASK]
    settings = SETTINGS.copy()
    diagram_length = cfg['diagram_L']
    if length is not None:
        settings['train_L'] = diagram_length = length
    rk = constrained_run_key(
        TASK, seed, settings['lnorm'], settings['train_L'],
        settings['hidden_dim'], settings['epochs'])
    if diagram_length != cfg['diagram_L']:
        rk += f'_Ldiag{diagram_length}'
    directory = RESULTS_DIR / 'network_behavior' / rk
    diagram_path = directory / 'diagram.json'
    observations_path = directory / 'observations.json'
    if not retrain and diagram_path.exists() and observations_path.exists():
        dataset = json.loads(observations_path.read_text())
        expected = dict(task=TASK, seed=seed, **settings)
        if (dataset['diagram_L'] != diagram_length
                or any(dataset['metadata'].get(key) != value
                       for key, value in expected.items())):
            raise ValueError(f'Cached observations settings mismatch: {observations_path}. '
                             'Use --retrain to replace this run.')
        print(f'Replaying saved observations with cached diagram: {diagram_path}')
        save_decoding_heatmap(directory, dataset['metadata'])
        save_traversal_analysis(directory, json.loads(diagram_path.read_text()),
                                dataset['observations'])
        return
    task_pairs = cfg['generate'](diagram_length)
    X, Y = build_vocab(cfg['generate'](cfg['diagram_L']))
    input_tokens, output_tokens = sorted(X), sorted(Y)
    model, metadata = load_or_train_network(
        directory, seed, input_tokens, output_tokens, retrain=retrain,
        settings=settings)
    save_decoding_heatmap(directory, metadata, model, recompute=retrain)
    observations = collect_behavior(
        model, task_pairs, input_tokens, output_tokens, settings['batch_size'])
    save_analysis(directory, metadata, observations, diagram_length=diagram_length)


def run_sweep_network(summary_dir, lnorm, seed, *, length=None):
    """Infer or replay behavior for a single explicitly selected sweep checkpoint."""
    from scripts.constrained_penalty_sweep import OUTPUT_DIR

    with (summary_dir / 'runs.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    matches = [row for row in rows
               if float(row['lnorm']) == lnorm and int(row['seed']) == seed]
    if len(matches) != 1:
        raise ValueError(f'Expected one sweep run for lnorm={lnorm:g}, seed={seed}; '
                         f'found {len(matches)}')
    row = matches[0]
    run_id = row['run_id']
    if Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError(f'Invalid run ID: {run_id}')
    run_dir = OUTPUT_DIR / 'runs' / run_id
    checkpoint_path = run_dir / 'network.pt'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Fully trained checkpoint required: {checkpoint_path}')
    saved = json.loads((run_dir / 'metrics.json').read_text())
    config = saved['configuration']
    metrics = saved['metrics']
    if (config['task'] != TASK or config['seed'] != seed or config['lnorm'] != lnorm
            or metrics['run_id'] != run_id or int(row['epochs']) != config['epochs']
            or row['readout_norm'] != metrics['readout_norm']
            or float(row['readout_scale']) != metrics['readout_scale']
            or bool(config.get('fixed_readout_norm', False)) !=
            (row['readout_norm'] == 'fixed')
            or config.get('readout_scale', 1.0) != float(row['readout_scale'])):
        raise ValueError(f'Sweep summary and run configuration mismatch: {run_id}')
    cfg = TASKS[TASK]
    X, Y = build_vocab(cfg['generate'](cfg['diagram_L']))
    input_tokens, output_tokens = sorted(X), sorted(Y)
    if config['input_tokens'] != input_tokens or config['output_tokens'] != output_tokens:
        raise ValueError(f'Checkpoint vocabulary does not match task: {run_id}')
    metadata = {key: value for key, value in config.items()
                if key not in ('split_seed', 'train_fraction', 'cache_version')}
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    model = load_checkpoint_network(checkpoint_path, metadata)
    metadata = dict(metadata, source_run_id=run_id, checkpoint_sha256=digest)
    diagram_length = cfg['diagram_L'] if length is None else length
    directory = RESULTS_DIR / 'network_behavior' / f'{run_id}_Ldiag{diagram_length}'
    observations_path = directory / 'observations.json'
    diagram_path = directory / 'diagram.json'
    if observations_path.exists() and diagram_path.exists():
        dataset = json.loads(observations_path.read_text())
        if dataset['diagram_L'] != diagram_length or dataset['metadata'] != metadata:
            raise ValueError(f'Cached observations or checkpoint mismatch: {observations_path}')
        print(f'Replaying saved observations with cached diagram: {diagram_path}')
        save_traversal_analysis(directory, json.loads(diagram_path.read_text()),
                                dataset['observations'])
    else:
        observations = collect_behavior(model, cfg['generate'](diagram_length),
                                        input_tokens, output_tokens,
                                        metadata['batch_size'])
        save_analysis(directory, metadata, observations, diagram_length=diagram_length)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument('--seed', type=int, help='Run just this seed.')
    seeds.add_argument('--seeds', type=int, nargs='+',
                       help='Independent initialization seeds (default: 1 2 3 4 5).')
    parser.add_argument('--length', type=int,
                        help='Use this sequence length for both training and inference.')
    parser.add_argument('--retrain', action='store_true',
                        help='Replace the saved networks with fresh training runs.')
    parser.add_argument('--sweep-summary', type=Path,
                        help='Saved penalty sweep summary directory; no training.')
    parser.add_argument('--run', nargs=2, action='append', metavar=('LNORM', 'SEED'),
                        help='Select a sweep run by lnorm and seed; repeat for multiple runs.')
    args = parser.parse_args()
    if args.sweep_summary is not None or args.run is not None:
        if args.sweep_summary is None or not args.run:
            parser.error('--sweep-summary and at least one --run are required together')
        if args.retrain or args.seed is not None or args.seeds is not None:
            parser.error('Sweep runs cannot be combined with --seed, --seeds, or --retrain')
        if args.length is not None and args.length < 1:
            parser.error('--length must be positive')
        try:
            selected_runs = [(float(lnorm), int(seed)) for lnorm, seed in args.run]
        except ValueError:
            parser.error('--run requires a numeric lnorm and integer seed')
        if (any(not math.isfinite(lnorm) or lnorm < 0 or seed < 1
                for lnorm, seed in selected_runs)
                or len(set(selected_runs)) != len(selected_runs)):
            parser.error('--run pairs must be distinct, with nonnegative lnorm and positive seed')
        for lnorm, seed in selected_runs:
            print(f'Analyzing sweep network lnorm={lnorm:g}, seed={seed}')
            run_sweep_network(args.sweep_summary, lnorm, seed, length=args.length)
        return
    selected = [args.seed] if args.seed is not None else (args.seeds or [1, 2, 3, 4, 5])
    if any(seed < 1 for seed in selected):
        parser.error('Seeds must be positive (as in Figure 3)')
    if len(set(selected)) != len(selected):
        parser.error('--seeds must contain distinct initialization seeds')
    if args.length is not None and args.length < 1:
        parser.error('--length must be positive')
    for seed in selected:
        print(f'Analyzing network seed={seed}')
        run_network(seed, length=args.length, retrain=args.retrain)


if __name__ == '__main__':
    main()
