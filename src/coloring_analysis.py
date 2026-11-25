"""Cross-entropy of consistent vs. random colorings of a cognitive diagram.

Port of get_cross_entropies from scripts_old/run_limitedstack.py and
run_nback.py, with two deliberate corrections (both flagged in chat and in
comments; use _legacy_avg_cross_entropy to reproduce the old numbers):

  1. START-STATE FIX: the legacy simulation initialized the belief at index
     `state2color[0]` (the COLOR of state 0) in a STATE-indexed vector. For
     canonicalized consistent colorings this happens to equal state 0, but for
     random colorings it frequently points at the wrong state, inflating their
     cross-entropy. The belief now starts at the diagram's start state for
     every coloring.
  2. The length sweep reports consistent colorings PER LENGTH (the legacy code
     reported a running mean over all lengths up to L) and reuses one fixed
     set of random colorings across lengths, so per-coloring slopes are
     well-defined.

The simulation itself is unchanged: propagate a belief over diagram states
under the coloring-smeared transitions (uniform within a color block),
renormalize each step, and accumulate -log2 of the probability assigned to
the true state (cd.trace). Implemented as one matrix-vector product per step;
_parity_check verifies agreement with the legacy implementation.
"""
import pickle
import random

import numpy as np
import pandas as pd

from src.cognitive_diagram import CognitiveDiagram
from src.lattice import Coloring, Lattice, all_partitions, random_partition
from src.paths import DIAGRAMS_DIR, ensure_dir
from src.utils import is_identical

# partition spaces at most this large are enumerated exhaustively (exact null)
ENUMERATION_CAP = 5000


# ---------------- cross-entropy of one coloring ----------------

def _transition_matrices(coloring):
    """Per-input-token state-transition matrices M[tok][from, to]."""
    if not hasattr(coloring, 'statetrans'):
        coloring.get_statetrans()
    n = len(coloring.state2color)
    mats = {}
    for (state, tok), vec in coloring.statetrans.items():
        if tok not in mats:
            mats[tok] = np.zeros((n, n))
        mats[tok][state, :] = vec
    return mats


def avg_cross_entropy(coloring, traced_pairs):
    """Mean sequence cross-entropy of the coloring's state belief.

    Vectorized: belief update is one matvec per step. Belief starts at the
    diagram's start state (see START-STATE FIX in the module docstring).
    """
    mats = _transition_matrices(coloring)
    n = len(coloring.state2color)
    start = coloring.cognitive_diagram.start_state
    total = 0.0
    for inputs, _, trace in traced_pairs:
        c = np.zeros(n)
        c[start] = 1.0
        ce = 0.0
        for tok, true_state in zip(inputs, trace):
            c = c @ mats[tok]
            c = c / c.sum()
            p = c[true_state]
            ce += -np.log2(p) if p > 0 else -np.log2(1e-10)
        total += ce
    return total / len(traced_pairs)


def _legacy_avg_cross_entropy(coloring, traced_pairs):
    """Reference implementation via Coloring.output_probs (legacy semantics,
    including the state2color[0] start index). Kept for parity checks."""
    total = 0.0
    for inputs, outputs, trace in traced_pairs:
        state_vecs, _ = coloring.output_probs(inputs, outputs)
        ce = 0.0
        for idx, state in enumerate(trace):
            p = state_vecs[idx][state]
            ce += -np.log2(p) if p > 0 else -np.log2(1e-10)
        total += ce
    return total / len(traced_pairs)


def _parity_check(coloring, traced_pairs):
    """Fast path must match the legacy implementation on colorings where the
    legacy start index is correct (canonicalized colorings: state2color[0]==0)."""
    assert coloring.state2color[0] == coloring.cognitive_diagram.start_state
    fast = avg_cross_entropy(coloring, traced_pairs)
    slow = _legacy_avg_cross_entropy(coloring, traced_pairs)
    assert np.isclose(fast, slow, atol=1e-8), (fast, slow)


# ---------------- null ensembles of random colorings ----------------

def random_colorings(cd, coloring_size, max_samples):
    """Partitions of the diagram states into coloring_size blocks.

    If the whole partition space has at most ENUMERATION_CAP elements it is
    enumerated exhaustively (exact null distribution -- improvement over the
    legacy with-replacement sampling). Otherwise falls back to the legacy
    random sampling capped at max_samples.
    """
    parts = []
    for p in all_partitions(list(cd.states), coloring_size):
        parts.append(p)
        if len(parts) > ENUMERATION_CAP:
            break
    else:
        return parts, True  # exhausted the generator: exact enumeration
    sampled = []
    for rc in random_partition(list(cd.states), coloring_size,
                               n_samples=max_samples * 1000):
        sampled.append(rc)
        if len(sampled) >= max_samples:
            break
    return sampled, False


# ---------------- sweeps ----------------

def coloring_size_entropies(cd, lattice, coloring_size, io_pairs, max_samples):
    """Entropy of every consistent coloring of the given size and of the
    random-partition null ensemble (exact enumeration when feasible).

    Returns a DataFrame with columns coloring ('consistent'/'random'),
    entropy, coloring_size, exact_null.
    """
    traced = [(inp, out, cd.trace(inp)) for inp, out in io_pairs]
    rows, cc_sets = [], []
    for node in lattice.nodes:
        if len(node.color2state) == coloring_size:
            if not cc_sets:  # one-time safeguard on the fast implementation
                _parity_check(node, traced[:20])
            cc_sets.append(node.get_sets())
            rows.append(dict(coloring='consistent',
                             entropy=np.round(avg_cross_entropy(node, traced), 4),
                             coloring_size=coloring_size))
    nulls, exact = random_colorings(cd, coloring_size, max_samples)
    n_random = 0
    for rc in nulls:
        if any(is_identical(rc, x) for x in cc_sets):
            continue  # identical to a consistent coloring: not a null sample
        rand = Coloring(coloring=rc, transitions=cd.transitions,
                        cognitive_diagram=cd)
        rows.append(dict(coloring='random',
                         entropy=np.round(avg_cross_entropy(rand, traced), 4),
                         coloring_size=coloring_size))
        n_random += 1
        if n_random % 100 == 0:
            print(f'  coloring_size={coloring_size}: random {n_random}')
    df = pd.DataFrame(rows)
    df['exact_null'] = exact
    return df


def length_sweep(cd, lattice, coloring_size, max_length, enumerate_io_pairs,
                 max_samples, num_io_pairs=None):
    """Consistent vs random colorings across sequence lengths 2..max_length.

    Returns a LONG DataFrame (length, kind, coloring_id, entropy): one row per
    coloring per length, with a fixed set of random colorings reused across
    lengths so that per-coloring slopes are well-defined.

    num_io_pairs: optional per-length subsample of the enumeration (for tasks
    whose enumeration explodes; nback uses the full enumeration as legacy).
    """
    cc_nodes = [n for n in lattice.nodes if len(n.color2state) == coloring_size]
    cc_sets = [n.get_sets() for n in cc_nodes]
    nulls, exact = random_colorings(cd, coloring_size, max_samples)
    rand_colorings = []
    for rc in nulls:
        if any(is_identical(rc, x) for x in cc_sets):
            continue
        rand_colorings.append(Coloring(coloring=rc, transitions=cd.transitions,
                                       cognitive_diagram=cd))
        if not exact and len(rand_colorings) >= max_samples:
            break
    print(f'length sweep: {len(cc_nodes)} consistent, {len(rand_colorings)} '
          f'random colorings ({"exact" if exact else "sampled"} null)')

    rows = []
    for length in range(2, max_length + 1):
        print('current io length:', length)
        io_pairs = enumerate_io_pairs(length)
        if num_io_pairs is not None and len(io_pairs) > num_io_pairs:
            io_pairs = random.sample(io_pairs, k=num_io_pairs)
        traced = [(inp, out, cd.trace(inp)) for inp, out in io_pairs]
        for node in cc_nodes:
            rows.append(dict(length=length, kind='consistent',
                             coloring_id=f'cc_{node.id}',
                             entropy=np.round(avg_cross_entropy(node, traced), 4)))
        for i, rand in enumerate(rand_colorings):
            rows.append(dict(length=length, kind='random',
                             coloring_id=f'rand_{i}',
                             entropy=np.round(avg_cross_entropy(rand, traced), 4)))
    return pd.DataFrame(rows)


# ---------------- simplified 3-back machine (legacy run_nback.py) ----------------

def simplified_nback3_diagram():
    """Hand-coded 8-state 3-back machine from the legacy SIMULATE block.

    States are the 2^3 three-token histories with cyclic transitions; the
    transient init states of the inferred 15-state diagram are dropped, so the
    first three outputs differ from the real task (intentional legacy choice).
    """
    cd = CognitiveDiagram(X={'a', 'b'}, Y={'*', '-'})
    cd.states = set(range(8))
    cd.statetrans = {(0, 'a'): 0, (0, 'b'): 1,
                     (1, 'a'): 2, (1, 'b'): 3,
                     (2, 'a'): 4, (2, 'b'): 5,
                     (3, 'a'): 6, (3, 'b'): 7,
                     (4, 'a'): 0, (4, 'b'): 1,
                     (5, 'a'): 2, (5, 'b'): 3,
                     (6, 'a'): 4, (6, 'b'): 5,
                     (7, 'a'): 6, (7, 'b'): 7}
    cd.outputtrans = {(0, 'a'): '*', (0, 'b'): '-',
                      (1, 'a'): '*', (1, 'b'): '-',
                      (2, 'a'): '*', (2, 'b'): '-',
                      (3, 'a'): '*', (3, 'b'): '-',
                      (4, 'a'): '-', (4, 'b'): '*',
                      (5, 'a'): '-', (5, 'b'): '*',
                      (6, 'a'): '-', (6, 'b'): '*',
                      (7, 'a'): '-', (7, 'b'): '*'}
    cd.start_state = 0
    cd._prepare_for_lattice()
    return cd


def get_simplified_nback3_lattice(recompute=False):
    """Lattice over the simplified 8-state 3-back machine (cached).
    Returns (cd, lattice)."""
    cache_path = DIAGRAMS_DIR / 'nback3_simplified8_lattice.pkl'
    if cache_path.exists() and not recompute:
        with open(cache_path, 'rb') as f:
            lattice = pickle.load(f)
        return lattice.cd, lattice
    cd = simplified_nback3_diagram()
    print('Building coloring lattice for simplified 3-back machine...')
    lattice = Lattice(cd)
    print(f'Lattice built: {len(lattice.nodes)} consistent colorings')
    ensure_dir(DIAGRAMS_DIR)
    with open(cache_path, 'wb') as f:
        pickle.dump(lattice, f)
    return cd, lattice
