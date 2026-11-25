"""Compare 2D-LDA pairwise separability in untrained and trained task RNNs.

Run from the repository root with:
  python scripts/fig_supp_linearitybaseline.py

Completed seeds are cached by task, model/training settings, diagram, and
decoding version. Re-running regenerates plots without training cached seeds.
Use --recompute to replace scores; each completed seed is saved immediately.
Existing output CSVs without metadata are reused with an explicit warning.
"""
import argparse
from collections import Counter, deque
import hashlib
from itertools import combinations
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import colorcet as cc
import seaborn as sns
import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decoding import balanced_indices, fit_projection
from scripts.common import save_figure
from src.paths import DIAGRAMS_DIR
from src.tasks import TASKS
from src.training import ConstrainedRNN, DataGenerator, train_model


TASK_NAMES = ('limitedstack', 'nback3')
N_SEEDS = 10
N_FOLDS = 5
RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
COLUMNS = ['seed', 'training', 'state_1', 'state_2', 'type', 'accuracy']
CACHE_VERSION = 'grouped_cv_lda2_v1'


def load_task_diagram(task_name):
    """Load and validate an existing task diagram cache."""
    config = TASKS[task_name]
    cache_path = DIAGRAMS_DIR / f'{task_name}_Ldiag{config["diagram_L"]}.pkl'
    if not cache_path.exists():
        raise FileNotFoundError(
            f'Expected cached {task_name} diagram at {cache_path}; '
            'this analysis does not infer a replacement diagram.')

    with cache_path.open('rb') as cache_file:
        diagram = pickle.load(cache_file)

    states = set(diagram.states)
    if len(states) != config['expected_states']:
        raise ValueError(
            f'{task_name}: expected {config["expected_states"]} states, '
            f'found {len(states)}.')
    if set(diagram.state_labels) != states:
        raise ValueError(f'Cached {task_name} diagram does not label every state.')
    for state in states:
        for symbol in diagram.X:
            if ((state, symbol) not in diagram.statetrans
                    or (state, symbol) not in diagram.outputtrans):
                raise ValueError(
                    f'Cached {task_name} diagram has no complete transition for '
                    f'state {state}, input {symbol!r}.')
    return diagram, cache_path


def shortest_distinguishing_sequence(diagram, first, second):
    """Return a shortest input sequence whose emitted outputs differ."""
    queue = deque([(first, second, ())])
    visited = {(first, second)}
    for_input = sorted(diagram.X)

    while queue:
        left, right, prefix = queue.popleft()
        for symbol in for_input:
            if (diagram.outputtrans[(left, symbol)]
                    != diagram.outputtrans[(right, symbol)]):
                return prefix + (symbol,)
            next_pair = (diagram.statetrans[(left, symbol)],
                         diagram.statetrans[(right, symbol)])
            if next_pair not in visited:
                visited.add(next_pair)
                queue.append((*next_pair, prefix + (symbol,)))
    return None


def classify_state_pairs(diagram):
    """Return all unordered state pairs with immediate/eventual labels."""
    pairs = []
    for first, second in combinations(sorted(diagram.states), 2):
        sequence = shortest_distinguishing_sequence(diagram, first, second)
        if sequence is None:
            raise ValueError(
                f'States {diagram.state_labels[first]!r} and '
                f'{diagram.state_labels[second]!r} are behaviorally equivalent.')

        immediate_inputs = tuple(
            symbol for symbol in sorted(diagram.X)
            if diagram.outputtrans[(first, symbol)]
            != diagram.outputtrans[(second, symbol)])
        classification = 'immediate' if immediate_inputs else 'eventual'
        if (classification == 'immediate') != (len(sequence) == 1):
            raise AssertionError('Pair label disagrees with its shortest witness.')
        pairs.append({
            'states': (first, second),
            'labels': (diagram.state_labels[first], diagram.state_labels[second]),
            'classification': classification,
            'immediate_inputs': immediate_inputs,
            'shortest_distinguishing_sequence': sequence,
        })
    return pairs


def print_summary(task_name, diagram, cache_path, pairs):
    """Print pair counts, percentages, and shortest distinction lengths."""
    counts = Counter(pair['classification'] for pair in pairs)
    total = len(pairs)
    print(f'{task_name} diagram: {cache_path.name}')
    print(f'States: {len(diagram.states)}')
    print(f'Unordered state pairs: {total}')
    print('Pair distribution:')
    for label in ('immediate', 'eventual'):
        count = counts[label]
        print(f'  {label}: {count} ({100 * count / total:.1f}%)')

    eventual_lengths = Counter(
        len(pair['shortest_distinguishing_sequence'])
        for pair in pairs if pair['classification'] == 'eventual')
    if eventual_lengths:
        print('Shortest distinguishing-sequence lengths among eventual pairs:')
        for length, count in sorted(eventual_lengths.items()):
            print(f'  {length} inputs: {count}')

    print('Immediate distinguishing inputs (counts overlap when multiple inputs work):')
    for symbol in sorted(diagram.X):
        count = sum(symbol in pair['immediate_inputs'] for pair in pairs)
        print(f'  {symbol}: {count} pairs')


def hidden_states_for_samples(model, samples, batch_size):
    """Collect aligned hidden states in the fixed order of the held-out samples."""
    model.eval()
    hidden_batches = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            inputs = torch.stack([sample[0] for sample in batch])
            _, hidden = model(inputs)
            hidden_batches.append(hidden[:, :-1, :].cpu().numpy())
    return np.concatenate(hidden_batches, axis=0)


def cross_validated_pair_accuracies(hidden, labels, groups, pairs, n_folds=N_FOLDS):
    """Score pairs after fold-specific, training-only 2D LDA projection."""
    n_sequences, n_steps, n_features = hidden.shape
    if labels.shape != (n_sequences, n_steps) or len(groups) != n_sequences:
        raise ValueError('Hidden states, state labels, and sequence groups must align.')
    predictors = hidden.reshape(-1, n_features)
    targets = labels.reshape(-1)
    expected_states = np.unique(targets)
    splitter = GroupKFold(n_splits=n_folds)
    available_states = set(expected_states.tolist())
    fold_accuracies = {
        tuple(pair['states']): []
        if set(pair['states']).issubset(available_states) else None
        for pair in pairs
    }

    for fold, (train_idx, test_idx) in enumerate(
            splitter.split(np.zeros((n_sequences, 1)), groups=groups)):
        train_idx = (train_idx[:, None] * n_steps
                     + np.arange(n_steps)).reshape(-1)
        test_idx = (test_idx[:, None] * n_steps
                    + np.arange(n_steps)).reshape(-1)
        for name, indices in (('training', train_idx), ('test', test_idx)):
            found = np.unique(targets[indices])
            if not np.array_equal(found, expected_states):
                raise ValueError(
                    f'Fold {fold + 1} {name} split misses states '
                    f'{np.setdiff1d(expected_states, found).tolist()}.')

        rng = np.random.default_rng(fold)
        balanced_train = train_idx[
            balanced_indices(targets[train_idx], rng)]
        balanced_test = test_idx[
            balanced_indices(targets[test_idx], rng)]
        train_targets = targets[balanced_train]
        test_targets = targets[balanced_test]
        projection = fit_projection(
            'LDA', predictors[balanced_train], train_targets, n_components=2)
        train_projection = projection.transform(predictors[balanced_train])
        test_projection = projection.transform(predictors[balanced_test])
        scaler = StandardScaler().fit(train_projection)
        train_projection = scaler.transform(train_projection)
        test_projection = scaler.transform(test_projection)
        for pair in pairs:
            state_pair = tuple(pair['states'])
            if fold_accuracies[state_pair] is None:
                continue
            first, second = state_pair
            train_mask = ((train_targets == first) | (train_targets == second))
            test_mask = ((test_targets == first) | (test_targets == second))
            classifier = LinearSVC(
                C=1e4, tol=1e-3, max_iter=5000, dual=False)
            classifier.fit(
                train_projection[train_mask], train_targets[train_mask])
            fold_accuracies[state_pair].append(
                classifier.score(test_projection[test_mask],
                                 test_targets[test_mask]))

    return {
        state_pair: float(np.mean(scores)) if scores is not None else float('nan')
        for state_pair, scores in fold_accuracies.items()
    }


def train_and_score(task_name, diagram, pairs, seeds, epochs, batch_size):
    """Train matched networks and score both conditions on the same sequences."""
    config = TASKS[task_name]
    torch.manual_seed(0)
    data = config['generate'](config['train_L'])
    datagenerator = DataGenerator(data=data, cog_diagram=diagram,
                                  batch_size=batch_size)
    test_samples = datagenerator.test

    labels = torch.stack([sample[2][:-1].argmax(dim=-1)
                          for sample in test_samples]).cpu().numpy()
    groups = np.arange(len(test_samples))
    rows = []

    for seed in seeds:
        torch.manual_seed(seed)
        model = ConstrainedRNN(
            input_dim=len(diagram.X), hidden_dim=config['hidden_dim'],
            output_dim=len(diagram.Y), seed=seed, lnorm=config['lnorm'])
        untrained_hidden = hidden_states_for_samples(model, test_samples, batch_size)

        print(f'[{task_name} seed={seed}] training for {epochs} epochs')
        train_model(model, datagenerator, learning_rate=0.001,
                    num_epochs=epochs, loss_tracking_epochs=100)
        trained_hidden = hidden_states_for_samples(model, test_samples, batch_size)

        for training, hidden in (('untrained', untrained_hidden),
                                 ('trained', trained_hidden)):
            accuracies = cross_validated_pair_accuracies(
                hidden, labels, groups, pairs)
            for pair in pairs:
                first, second = pair['states']
                rows.append({
                    'seed': seed,
                    'training': training,
                    'state_1': pair['labels'][0],
                    'state_2': pair['labels'][1],
                    'type': pair['classification'],
                    'accuracy': accuracies[(first, second)],
                })

    return pd.DataFrame(
        rows, columns=COLUMNS)


def cached_results(task_name, diagram, pairs, seeds, epochs, batch_size,
                   result_path, recompute=False):
    """Reuse complete seeds, checkpoint missing seeds, and export requested rows."""
    config = TASKS[task_name]
    settings = dict(
        version=CACHE_VERSION, task=task_name, epochs=epochs, batch_size=batch_size,
        hidden_dim=config['hidden_dim'], train_L=config['train_L'],
        lnorm=config['lnorm'], folds=N_FOLDS, learning_rate=0.001,
        transitions=[
            [state, symbol, diagram.statetrans[state, symbol],
             diagram.outputtrans[state, symbol]]
            for state in sorted(diagram.states) for symbol in sorted(diagram.X)],
        labels=sorted(diagram.state_labels.items()))
    serialized = json.dumps(settings, sort_keys=True)
    signature = hashlib.sha256(serialized.encode()).hexdigest()[:16]
    cache_path = RESULTS_DIR / 'linearity_baseline_cache' / (
        f'{task_name}_{signature}.csv')
    metadata_path = result_path.with_suffix('.json')
    cached = pd.DataFrame(columns=COLUMNS)
    if not recompute:
        source = None
        if cache_path.exists():
            source = cache_path
        elif result_path.exists():
            if metadata_path.exists():
                if json.loads(metadata_path.read_text()) == json.loads(serialized):
                    source = result_path
                else:
                    print(f'{task_name}: output settings changed; not reusing {result_path}.')
            else:
                print(
                    f'WARNING: reusing legacy results {result_path} without '
                    'configuration metadata. Use --recompute if these were '
                    'generated with different training/decoding settings.')
                source = result_path
        if source is not None:
            cached = pd.read_csv(source)
            if list(cached.columns) != COLUMNS:
                raise ValueError(
                    f'Unexpected columns in {source}; expected {COLUMNS}. '
                    'Use --recompute to replace this cache.')
            print(f'{task_name}: loaded cached results from {source}')

    expected = {
        (training, *pair['labels'], pair['classification'])
        for pair in pairs for training in ('untrained', 'trained')
    }

    def complete(rows):
        keys = set(rows[['training', 'state_1', 'state_2', 'type']]
                   .itertuples(index=False, name=None))
        # nback's initial state has no post-input hidden-state observations.
        unavailable = ((rows['state_1'] == 'e') | (rows['state_2'] == 'e')
                       if task_name == 'nback3'
                       else pd.Series(False, index=rows.index))
        valid = rows['accuracy'].between(0, 1)
        valid = valid | (unavailable & rows['accuracy'].isna())
        return len(rows) == len(expected) and keys == expected and valid.all()

    seeds = list(seeds)
    for seed in seeds:
        rows = cached[cached['seed'] == seed]
        if complete(rows):
            print(f'[{task_name} seed={seed}] using cached scores; skipping training.')
        else:
            if not rows.empty:
                print(f'[{task_name} seed={seed}] incomplete cached scores; recomputing seed.')
            rows = train_and_score(
                task_name, diagram, pairs, seeds=[seed], epochs=epochs,
                batch_size=batch_size)
            if not complete(rows):
                raise ValueError(f'{task_name} seed={seed}: incomplete or invalid decoding results.')
            remaining = cached[cached['seed'] != seed]
            cached = (rows if remaining.empty
                      else pd.concat([remaining, rows], ignore_index=True))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix('.csv.tmp')
        cached.to_csv(temporary, index=False)
        temporary.replace(cache_path)

    results = cached[cached['seed'].isin(seeds)].copy()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_suffix('.csv.tmp')
    results.to_csv(temporary, index=False)
    temporary.replace(result_path)
    metadata_path.write_text(serialized + '\n')
    print(f'Saved pairwise decoding results to {result_path}')
    return results


def plot_results(results, figure_name):
    """Plot network means as filled boxplots without markers or outliers."""
    network_means = (results.groupby(['seed', 'training', 'type'], as_index=False)
                     ['accuracy'].mean())
    categories = ['immediate', 'eventual']
    category_labels = ['Immediate', 'Eventual']
    fig, ax = plt.subplots(figsize=(4, 4))
    hue_order = ['untrained', 'trained']
    sns.boxplot(
        data=network_means, x='type', y='accuracy', hue='training',
        order=categories, hue_order=hue_order,
        palette={'untrained': 'lightgray', 'trained': cc.glasbey[3]},
        saturation=1, dodge=True, showfliers=False,
        boxprops={'alpha': 0.35}, ax=ax)

    if ax.get_legend() is not None:
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles[:len(hue_order)], ['Untrained', 'Trained'],
                  frameon=False, title=None)

    ax.set_xticks(np.arange(len(categories)), category_labels)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('')
    save_figure(figure_name)
    plt.close(fig)


def print_accuracy_summary(task_name, results, pairs):
    """Print pair counts, condition means, and paired network improvements."""
    network_means = (results.groupby(['seed', 'training', 'type'])['accuracy']
                     .mean().unstack('training'))
    counts = Counter(pair['classification'] for pair in pairs)
    pair_rows = results.drop_duplicates(['state_1', 'state_2', 'type'])
    scorable_counts = {
        category: int(((pair_rows['type'] == category)
                       & pair_rows['accuracy'].notna()).sum())
        for category in ('immediate', 'eventual')
    }
    print(f'\n{task_name}: cross-validated pairwise accuracy (network mean ± SEM):')
    for category in ('immediate', 'eventual'):
        paired = network_means.xs(category, level='type')
        improvements = paired['trained'] - paired['untrained']
        sem = lambda values: (values.std(ddof=1) / np.sqrt(len(values))
                              if len(values) > 1 else 0.0)
        print(f'  {category}: {counts[category]} state pairs '
              f'({scorable_counts[category]} represented in samples)')
        for condition in ('untrained', 'trained'):
            values = paired[condition]
            print(f'    {condition}: {values.mean():.4f} ± {sem(values):.4f}')
        print('    paired trained − untrained: '
              f'{improvements.mean():+.4f} ± {sem(improvements):.4f} '
              f'(n={len(improvements)} networks)')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tasks', nargs='+', choices=TASK_NAMES,
                        default=list(TASK_NAMES))
    parser.add_argument('--seeds', type=int, default=N_SEEDS,
                        help='Number of matched network initializations (default: 10).')
    parser.add_argument('--epochs', type=int,
                        help='Override task-specific training epochs.')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--recompute', action='store_true',
                        help='Retrain requested seeds instead of reusing cached scores.')
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('--seeds must be at least 1.')
    if args.batch_size < 1:
        parser.error('--batch-size must be at least 1.')

    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be at least 1.')

    for task_name in args.tasks:
        config = TASKS[task_name]
        epochs = args.epochs if args.epochs is not None else config['epochs']
        diagram, cache_path = load_task_diagram(task_name)
        pairs = classify_state_pairs(diagram)
        print_summary(task_name, diagram, cache_path, pairs)
        result_path = RESULTS_DIR / f'linearity_baseline_{task_name}.csv'
        figure_name = f'supp_linearitybaseline_{task_name}'
        if (args.seeds != N_SEEDS or args.epochs is not None
                or args.batch_size != 128):
            suffix = f'_seeds{args.seeds}_epochs{epochs}_batch{args.batch_size}'
            result_path = result_path.with_name(
                f'{result_path.stem}{suffix}{result_path.suffix}')
            figure_name = f'{figure_name}{suffix}'
        results = cached_results(
            task_name, diagram, pairs, seeds=range(args.seeds), epochs=epochs,
            batch_size=args.batch_size, result_path=result_path,
            recompute=args.recompute)
        plot_results(results, figure_name=figure_name)
        print_accuracy_summary(task_name, results, pairs)


if __name__ == '__main__':
    main()
