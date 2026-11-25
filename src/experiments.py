"""Per-(task, seed) experiment runs with a deliberately lean storage policy.

Storage:
  * results/decoding_accuracies_heldout.csv -- one row per (run, test, method).
  * results/history_state_decoding_heldout.csv -- one row per (run, test, k).
    These small scalar tables are kept for every seed.
  * results/hidden_states/<run_key>.pt -- raw hidden-state tensors, kept ONLY
    for the scatter seed of each task (so plots can be re-styled without
    retraining). All other seeds are trained, scored, and discarded.

Legacy tensors can be rescored, but their in-sample accuracies are never reused.
The old decoding CSV is left untouched.
"""
import pickle
import random

import numpy as np
import pandas as pd
import torch

from src.diagrams import build_vocab, load_or_infer_diagram
from src.decoding import state_projection_analysis
from src.history_decoding import history_decoding_scores
from src.lattice import Lattice
from src.paths import (DIAGRAMS_DIR, HIDDEN_STATES_DIR, RESULTS_DIR,
                       ensure_dir)
from src.projections import get_pairwise_decodings
from src.tasks import TASKS, verify_state_labels
from src.training import (ConstrainedRNN, DataGenerator, flatten_states,
                          get_hidden_states, train_model)

METHODS = ['LDA', 'PCA', 'ICA', 'FA']
SHUFFLED_METHOD = 'LDA_shuffled'  # label-shuffled LDA control
DECODING_VERSION = 'heldout_state_v1'
ACCURACIES_CSV = RESULTS_DIR / 'decoding_accuracies_heldout.csv'
CSV_COLUMNS = ['task', 'seed', 'split', 'method', 'accuracy', 'run_key']
HISTORY_ACCURACIES_CSV = RESULTS_DIR / 'history_state_decoding_heldout.csv'
HISTORY_COLUMNS = [
    'task', 'seed', 'split', 'k', 'accuracy', 'n_history_classes',
    'history_train_samples_per_class', 'n_projection_train_samples',
    'n_components', 'n_state_classes', 'n_train_samples', 'n_test_samples',
    'status', 'reason', 'run_key']

# constrained regime (legacy run_constrained.py): strong activity penalties
CONSTRAINED_CSV = RESULTS_DIR / 'constrained_runs.csv'
CONSTRAINED_COLUMNS = ['task', 'seed', 'lnorm', 'final_acc', 'final_loss', 'run_key']
CONSTRAINED_HEATMAPS_DIR = RESULTS_DIR / 'constrained_heatmaps'


# ---------------- cognitive diagram ----------------

def verify_behavior(cd, io_pairs):
    """Assert the diagram reproduces every (input, output) pair exactly."""
    for inp, out in io_pairs:
        _, predicted = cd.apply_transitions(cd.start_state, inp)
        assert predicted == tuple(out), (
            f'Diagram does not reproduce behavior: input {inp} -> '
            f'{predicted}, expected {tuple(out)}')
    print(f'Verified diagram behavior on {len(io_pairs)} io pairs.')


def _diagram_cache_key(task_name):
    cfg = TASKS[task_name]
    sample_tag = ''
    if cfg['diagram_sample'] is not None:
        k, _ = cfg['diagram_sample']
        if len(cfg['generate'](cfg['diagram_L'])) > k:
            sample_tag = f'_sample{k}'
    return f"{task_name}_Ldiag{cfg['diagram_L']}{sample_tag}"


def get_task_diagram(task_name):
    """Infer (or load the cached) minimal cognitive diagram for a task, with
    full behavioral verification and semantic verification of state labels."""
    cfg = TASKS[task_name]
    io_pairs = cfg['generate'](cfg['diagram_L'])
    X, Y = build_vocab(io_pairs)
    inference_pairs = io_pairs
    if cfg['diagram_sample'] is not None:
        k, sample_seed = cfg['diagram_sample']
        if len(io_pairs) > k:
            random.seed(sample_seed)
            inference_pairs = random.sample(io_pairs, k=k)
    cache_key = _diagram_cache_key(task_name)
    cd = load_or_infer_diagram(cache_key, inference_pairs, X, Y,
                               state_labels=cfg['state_labels'])
    try:
        assert len(cd.states) == cfg['expected_states'], (
            f"{task_name}: expected {cfg['expected_states']} states, "
            f"got {len(cd.states)} -- increase diagram_L or the sample size")
        verify_behavior(cd, io_pairs)  # full enumeration at diagram_L
        verify_behavior(cd, cfg['generate'](cfg['train_L']))  # and at train_L
        verify_state_labels(cd, cfg['label_probes'])
    except AssertionError:
        # Never leave an unverified diagram in the cache.
        cache_path = DIAGRAMS_DIR / f'{cache_key}.pkl'
        if cache_path.exists():
            cache_path.unlink()
            print(f'Removed failed diagram cache {cache_path}')
        raise
    return cd


def get_task_lattice(task_name, recompute=False):
    """Lattice of consistent colorings for a task's diagram, with pairwise
    matrices prepared. Cached next to the diagram (the task doesn't change,
    and building the lattice is combinatorial)."""
    cache_path = DIAGRAMS_DIR / f'{_diagram_cache_key(task_name)}_lattice.pkl'
    if cache_path.exists() and not recompute:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    cd = get_task_diagram(task_name)
    cd._prepare_for_lattice()
    print(f'Building coloring lattice for {task_name}...')
    lattice = Lattice(cd)
    lattice.get_pairwise_matrices()
    print(f'Lattice built: {len(lattice.nodes)} consistent colorings')
    ensure_dir(DIAGRAMS_DIR)
    with open(cache_path, 'wb') as f:
        pickle.dump(lattice, f)
    return lattice


# ---------------- projections and scores ----------------

def project_and_score(hidden_train, truths_train, hidden_test, truths_test,
                      keep_states=None, score=True, seed=0, shuffled=False):
    """Train-only projections and optional held-out state-pair classification.

    Returns test projections, test accuracies (or None), and test state IDs.
    Scatter-only state filtering is applied to both splits before balancing.
    """
    train, y_train = flatten_states(hidden_train, truths_train)
    test, y_test = flatten_states(hidden_test, truths_test)
    return state_projection_analysis(
        train.detach().cpu().numpy(), y_train.detach().cpu().numpy(),
        test.detach().cpu().numpy(), y_test.detach().cpu().numpy(),
        keep_states=keep_states, expected_states=np.arange(truths_train.shape[-1]),
        score=score, seed=seed, shuffled=shuffled)


# ---------------- runs and storage ----------------

def run_key(task_name, seed):
    cfg = TASKS[task_name]
    return (f"{task_name}_L{cfg['train_L']}_h{cfg['hidden_dim']}"
            f"_lnorm{cfg['lnorm']:g}_ep{cfg['epochs']}_seed{seed}")


def _hidden_path(task_name, seed):
    return HIDDEN_STATES_DIR / f'{run_key(task_name, seed)}.pt'


def _load_accuracies():
    if ACCURACIES_CSV.exists():
        return pd.read_csv(ACCURACIES_CSV)
    return pd.DataFrame(columns=CSV_COLUMNS)


def _store_run_rows(rows):
    """Idempotent write: replaces any existing rows for the same run_key."""
    df = _load_accuracies()
    df = df[df['run_key'] != rows[0]['run_key']]
    rows = pd.DataFrame(rows)
    df = rows if df.empty else pd.concat([df, rows], ignore_index=True)
    ensure_dir(RESULTS_DIR)
    df.to_csv(ACCURACIES_CSV, index=False)


def _complete_accuracy_rows(rows):
    return (len(rows) == len(METHODS) + 1
            and set(rows['method']) == set(METHODS + [SHUFFLED_METHOD])
            and set(rows['split']) == {'test'})


def _history_lengths(task_name):
    return set(range(1, TASKS[task_name].get('history_max_k', 3) + 1))


def _load_history_accuracies():
    if HISTORY_ACCURACIES_CSV.exists():
        return pd.read_csv(HISTORY_ACCURACIES_CSV)
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def _history_rows(task_name, seed, run):
    scores = history_decoding_scores(
        hidden_train=run['hidden_train'].detach().cpu().numpy(),
        inputs_train=run['inputs_train'].detach().cpu().numpy(),
        truths_train=run['truths_train'].argmax(dim=-1).detach().cpu().numpy(),
        hidden_test=run['hidden_test'].detach().cpu().numpy(),
        inputs_test=run['inputs_test'].detach().cpu().numpy(),
        truths_test=run['truths_test'].argmax(dim=-1).detach().cpu().numpy(),
        seed=seed, max_k=TASKS[task_name].get('history_max_k', 3))
    return pd.DataFrame([dict(task=task_name, seed=seed, split='test',
                              run_key=run_key(task_name, seed), **score)
                         for score in scores])


def _store_history_rows(rows):
    df = _load_history_accuracies()
    selected = ((df['run_key'] == rows.iloc[0]['run_key'])
                & (df['split'] == 'test'))
    remaining = df[~selected]
    df = rows if remaining.empty else pd.concat([remaining, rows], ignore_index=True)
    ensure_dir(RESULTS_DIR)
    df.to_csv(HISTORY_ACCURACIES_CSV, index=False)


def _has_history_inputs(run):
    return all(f'{field}_{split}' in run
               for field in ('hidden', 'truths', 'inputs')
               for split in ('train', 'test'))


def _ensure_decoding(run, seed):
    if run.get('decoding_version') == DECODING_VERSION:
        return run
    arguments = (run['hidden_train'], run['truths_train'],
                 run['hidden_test'], run['truths_test'])
    _, run['accs_test'], _ = project_and_score(*arguments, seed=seed)
    _, shuffled, _ = project_and_score(*arguments, seed=seed, shuffled=True)
    run['acc_lda_shuffled_test'] = shuffled['LDA']
    run['decoding_version'] = DECODING_VERSION
    return run


def _accuracy_rows(task_name, seed, run):
    rk = run_key(task_name, seed)
    rows = []
    for method in METHODS:
        rows.append(dict(task=task_name, seed=seed, split='test',
                         method=method, accuracy=run['accs_test'][method],
                         run_key=rk))
    rows.append(dict(task=task_name, seed=seed, split='test',
                     method=SHUFFLED_METHOD,
                     accuracy=run['acc_lda_shuffled_test'], run_key=rk))
    return rows


def _train_run(task_name, seed, learning_rate=0.001, batch_size=128, decode=True):
    """Train one (task, seed) run in memory: hidden states for both splits,
    decoding accuracies, and the shuffled-LDA control."""
    cfg = TASKS[task_name]
    cd = get_task_diagram(task_name)

    # Legacy RNG discipline: ambient torch seed 0 before the train/test split;
    # the model is built afterwards (seed=0 keeps the ambient stream).
    torch.manual_seed(0)
    io_pairs = cfg['generate'](cfg['train_L'])
    print(f"[{task_name} seed={seed}] training on {len(io_pairs)} sequences "
          f"of length {cfg['train_L']}")
    datagenerator = DataGenerator(data=io_pairs, cog_diagram=cd, batch_size=batch_size)
    model = ConstrainedRNN(input_dim=len(cd.X), hidden_dim=cfg['hidden_dim'],
                           output_dim=len(cd.Y), seed=seed, lnorm=cfg['lnorm'])
    model, track_loss = train_model(model, datagenerator, learning_rate,
                                    cfg['epochs'], loss_tracking_epochs=100)

    run = {'state_labels': cd.state_labels,
           'final_loss': track_loss['loss'][-1],
           'final_acc': track_loss['acc'][-1]}
    for split, is_train in [('train', True), ('test', False)]:
        hidden, truths, inputs, _ = get_hidden_states(model, datagenerator, train=is_train)
        # Drop the last timestep (legacy alignment with next-step inputs).
        run[f'hidden_{split}'] = hidden[:, :-1, :].detach()
        run[f'truths_{split}'] = truths[:, :-1, :].detach()
        # History labels describe h_t, so retain x_t, not the next-step input.
        run[f'inputs_{split}'] = inputs[:, :-1, :].argmax(dim=-1).detach()
    return _ensure_decoding(run, seed) if decode else run


def untrained_run(task_name, seed=0, batch_size=128):
    """Hidden states of the *untrained* network (reviewer baseline).

    Identical setup to _train_run -- same RNG discipline, same initial
    weights -- but without any training. Cheap (one forward pass), so it is
    recomputed on demand and never stored.
    """
    cfg = TASKS[task_name]
    cd = get_task_diagram(task_name)
    torch.manual_seed(0)
    io_pairs = cfg['generate'](cfg['train_L'])
    datagenerator = DataGenerator(data=io_pairs, cog_diagram=cd, batch_size=batch_size)
    model = ConstrainedRNN(input_dim=len(cd.X), hidden_dim=cfg['hidden_dim'],
                           output_dim=len(cd.Y), seed=seed, lnorm=cfg['lnorm'])
    run = {'state_labels': cd.state_labels}
    for split, is_train in [('train', True), ('test', False)]:
        hidden, truths, _, _ = get_hidden_states(model, datagenerator, train=is_train)
        run[f'hidden_{split}'] = hidden[:, :-1, :].detach()
        run[f'truths_{split}'] = truths[:, :-1, :].detach()
    return run


def run_accuracies(task_name, seed, recompute=False, keep_hidden=False):
    """Decoding-accuracy rows for one run (trains it if needed).

    Hidden states are persisted only when keep_hidden=True (the scatter seed);
    otherwise the run is scored and its tensors discarded. Pre-existing .pt
    caches are rescored with train-only projections, then deleted (unless
    keep_hidden). Legacy scalar scores cannot satisfy the new cache lookup.
    """
    rk = run_key(task_name, seed)
    hidden_path = _hidden_path(task_name, seed)
    if not recompute:
        df = _load_accuracies()
        have = df[df['run_key'] == rk]
        if _complete_accuracy_rows(have):
            if hidden_path.exists() and not keep_hidden:
                hidden_path.unlink()  # redundant large cache
                print(f'Removed redundant hidden-state cache {hidden_path}')
            return have

    if hidden_path.exists() and not recompute:
        run = torch.load(hidden_path, weights_only=False)  # harvest legacy cache
        run = _ensure_decoding(run, seed)
    else:
        run = _train_run(task_name, seed)

    _store_run_rows(_accuracy_rows(task_name, seed, run))
    ensure_dir(HIDDEN_STATES_DIR)
    if keep_hidden:
        torch.save(run, hidden_path)
    elif hidden_path.exists():
        hidden_path.unlink()
    df = _load_accuracies()
    return df[df['run_key'] == rk]


def scatter_run(task_name, seed=0, recompute=False):
    """Full run dict (incl. hidden states) for the scatter plots; persisted so
    plot styling changes never require retraining."""
    hidden_path = _hidden_path(task_name, seed)
    if hidden_path.exists() and not recompute:
        return torch.load(hidden_path, weights_only=False)
    run = _train_run(task_name, seed)
    ensure_dir(HIDDEN_STATES_DIR)
    torch.save(run, hidden_path)
    _store_run_rows(_accuracy_rows(task_name, seed, run))
    return run


def history_run_accuracies(task_name, seed, split='test', recompute=False,
                          keep_hidden=False):
    """History-supervised LDA with held-out diagram-state decoding.

    Reuse aligned input IDs when present in a hidden-state cache. Old caches
    contain no input ordering; retrain once rather than guess that ordering.
    As with the main analysis, only the scatter seed keeps its tensors.
    """
    if split != 'test':
        raise ValueError('History decoding is evaluated only on the test split.')
    history_lengths = _history_lengths(task_name)
    rk = run_key(task_name, seed)
    df = _load_history_accuracies()
    selected = (df['run_key'] == rk) & (df['split'] == split)
    have = df[selected & df['k'].isin(history_lengths)]
    if (not recompute and len(have) == len(history_lengths)
            and set(have['k']) == history_lengths):
        return have

    hidden_path = _hidden_path(task_name, seed)
    run = None
    if hidden_path.exists() and not recompute:
        run = torch.load(hidden_path, weights_only=False)
    if run is None or not _has_history_inputs(run):
        print(f'[{task_name} seed={seed}] training for history decoding: '
              'aligned input IDs are unavailable or recomputation was requested.')
        run = _train_run(task_name, seed, decode=False)
    rows = _history_rows(task_name, seed, run)
    _store_history_rows(rows)
    if keep_hidden:
        ensure_dir(HIDDEN_STATES_DIR)
        torch.save(run, hidden_path)
    elif hidden_path.exists():
        hidden_path.unlink()
    return rows


def comparison_run_accuracies(task_name, seed, recompute=False, keep_hidden=False):
    """Return original/control and history rows, sharing one run for missing scores."""
    rk = run_key(task_name, seed)
    main = _load_accuracies()
    main = main[main['run_key'] == rk]
    history = _load_history_accuracies()
    lengths = _history_lengths(task_name)
    history = history[(history['run_key'] == rk) & (history['split'] == 'test')
                      & history['k'].isin(lengths)]
    need_main = recompute or not _complete_accuracy_rows(main)
    need_history = (recompute or len(history) != len(lengths)
                    or set(history['k']) != lengths)
    if not need_main and not need_history:
        return main, history

    hidden_path = _hidden_path(task_name, seed)
    run = None
    if hidden_path.exists() and not recompute:
        run = torch.load(hidden_path, weights_only=False)
    if run is None or (need_history and not _has_history_inputs(run)):
        print(f'[{task_name} seed={seed}] training for combined decoding: '
              'required cached tensors are unavailable or recomputation was requested.')
        run = _train_run(task_name, seed, decode=False)
    if need_main:
        run = _ensure_decoding(run, seed)
        main = pd.DataFrame(_accuracy_rows(task_name, seed, run))
        _store_run_rows(main.to_dict('records'))
    if need_history:
        history = _history_rows(task_name, seed, run)
        _store_history_rows(history)
    if keep_hidden:
        ensure_dir(HIDDEN_STATES_DIR)
        torch.save(run, hidden_path)
    elif hidden_path.exists():
        hidden_path.unlink()
    return main, history


# ---------------- constrained regime (legacy run_constrained.py) ----------------

def constrained_run_key(task_name, seed, lnorm, train_L, hidden_dim, epochs):
    return (f'{task_name}_constrained_L{train_L}_h{hidden_dim}'
            f'_lnorm{lnorm:g}_ep{epochs}_seed{seed}')


def _load_constrained():
    if CONSTRAINED_CSV.exists():
        return pd.read_csv(CONSTRAINED_CSV)
    return pd.DataFrame(columns=CONSTRAINED_COLUMNS)


def constrained_run(task_name, seed, lnorm, train_L=7, hidden_dim=5,
                    epochs=5000, recompute=False, learning_rate=0.001,
                    batch_size=128):
    """One constrained training run (strong activity penalty 1/lnorm).

    Stores a row (final loss/accuracy) in results/constrained_runs.csv and the
    end-of-training pairwise LDA decodability matrix (n_states x n_states) in
    results/constrained_heatmaps/<run_key>.npy. Hidden states are discarded.

    As in the legacy run_constrained.py, the decodability matrix is computed
    on the train split without timestep trimming. Unlike the legacy code, the
    pairwise entries are balanced accuracies, so unequal state visitation
    cannot inflate them (identical representations score 0.5).

    Returns (row_dict, pairwise_matrix).
    """
    rk = constrained_run_key(task_name, seed, lnorm, train_L, hidden_dim, epochs)
    matrix_path = CONSTRAINED_HEATMAPS_DIR / f'{rk}.npy'
    if not recompute:
        df = _load_constrained()
        have = df[df['run_key'] == rk]
        if len(have) > 0 and matrix_path.exists():
            return have.iloc[0].to_dict(), np.load(matrix_path)

    cfg = TASKS[task_name]
    cd = get_task_diagram(task_name)
    torch.manual_seed(0)
    io_pairs = cfg['generate'](train_L)
    print(f'[{task_name} constrained lnorm={lnorm:g} seed={seed}] training on '
          f'{len(io_pairs)} sequences of length {train_L}')
    datagenerator = DataGenerator(data=io_pairs, cog_diagram=cd, batch_size=batch_size)
    model = ConstrainedRNN(input_dim=len(cd.X), hidden_dim=hidden_dim,
                           output_dim=len(cd.Y), seed=seed, lnorm=lnorm)
    model, track_loss = train_model(model, datagenerator, learning_rate,
                                    epochs, loss_tracking_epochs=100)

    hidden, truths, _, _ = get_hidden_states(model, datagenerator, train=True)
    predictor, target = flatten_states(hidden, truths)
    pred_pairwise = get_pairwise_decodings(predictor, target)

    ensure_dir(CONSTRAINED_HEATMAPS_DIR)
    np.save(matrix_path, pred_pairwise)
    row = dict(task=task_name, seed=seed, lnorm=lnorm,
               final_acc=track_loss['acc'][-1], final_loss=track_loss['loss'][-1],
               run_key=rk)
    df = _load_constrained()
    df = df[df['run_key'] != rk]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    ensure_dir(RESULTS_DIR)
    df.to_csv(CONSTRAINED_CSV, index=False)
    return row, pred_pairwise
