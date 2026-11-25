"""Task registry for the paper experiments.

Each entry bundles the data generator, diagram-inference settings (including
semantic verification probes), and the RNN hyperparameters used in the paper
(from scripts_old/simulations.sh).

Label probes: the minimal-machine state labels are semantic (e.g. stack
contents). Because state *ids* depend on the inference run, every label is
verified by tracing a canonical input that must land on it; see
verify_state_labels().
"""
from itertools import product as _product

from src.diagrams import (LIMITEDLIST_STATE_LABELS, LIMITEDSET_STATE_LABELS,
                          LIMITEDSTACK_STATE_LABELS, LISTMAX_STATE_LABELS,
                          NBACK_STATE_LABELS)
from src.limitedlist import generate_sequences as limitedlist_sequences
from src.limitedset import generate_sequences as limitedset_sequences
from src.limitedstack import generate_sequences as limitedstack_sequences
from src.listmax import generate_sequences as listmax_sequences
from src.nback import generate_nback_sequences


def _nback_probes(n, tokens=('a', 'b')):
    """label = last min(t, n) tokens; feeding the label itself reaches it."""
    probes = {'e': ()}
    for length in range(1, n + 1):
        for combo in _product(tokens, repeat=length):
            probes[''.join(combo)] = combo
    return probes


def _limitedstack_probes():
    probes = {'e': ()}
    for length in range(1, 4):
        for combo in _product('xy', repeat=length):
            probes[''.join(combo)] = tuple(f'push({c})' for c in combo)
    probes['E'] = tuple(['push(x)'] * 4)
    return probes


def _limitedset_probes():
    probes = {'e': ()}
    for label in ['x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz']:
        probes[label] = tuple(f'add({c})' for c in label)
    return probes


def _limitedlist_probes():
    probes = {'--': ()}
    for c in 'xy':
        probes[f'{c}-'] = (f'write({c},0)',)
        probes[f'-{c}'] = (f'write({c},1)',)
    for c0 in 'xy':
        for c1 in 'xy':
            probes[f'{c0}{c1}'] = (f'write({c0},0)', f'write({c1},1)')
    return probes


TASKS = {
    # 3-back, 2 stimuli
    'nback3': dict(
        generate=lambda L: generate_nback_sequences(L=L, n=3, k=2),
        diagram_L=7, diagram_sample=None,
        state_labels=NBACK_STATE_LABELS, expected_states=15,
        label_probes=_nback_probes(3),
        train_L=10, hidden_dim=8, epochs=400, lnorm=10000.0,
        jitter=True,  # states are near-point clusters; legacy added jitter
        # scatter panels show only the "contentful" full-history states
        # (aaa..bbb); transient states e/a/b/aa/... occur only at t<3
        plot_states=lambda label: len(label) == 3,
    ),
    # running max of a digit sequence
    'listmax': dict(
        generate=listmax_sequences,
        # Full L=3 enumeration (1000 pairs). L=2 and random 200-samples of L=3
        # both fail behavioral verification (insufficient transition coverage).
        diagram_L=3, diagram_sample=None,
        state_labels=LISTMAX_STATE_LABELS, expected_states=10,
        label_probes={str(d): (str(d),) for d in range(10)},
        train_L=4, hidden_dim=5, epochs=200, lnorm=1000.0,
        history_max_k=2,
        jitter=False,
    ),
    # 3-element set with add/remove
    'limitedset': dict(
        generate=limitedset_sequences,
        # Legacy Ldiag=7, random 1000-sample. L=4 is insufficient: transitions
        # out of the full 'xyz' state (3 steps deep) are never probed, so the
        # machine mislinks them (caught by verify_behavior at train_L).
        diagram_L=7, diagram_sample=(1000, 42),
        state_labels=LIMITEDSET_STATE_LABELS, expected_states=8,
        label_probes=_limitedset_probes(),
        train_L=6, hidden_dim=8, epochs=100, lnorm=1000.0,
        jitter=False,
    ),
    # two write/read slots
    'limitedlist': dict(
        generate=limitedlist_sequences,
        # Legacy Ldiag=6: distinguishing destinations needs both slots read
        # after a transition, so shorter inference data risks mislinks.
        diagram_L=6, diagram_sample=(1000, 42),
        state_labels=LIMITEDLIST_STATE_LABELS, expected_states=9,
        label_probes=_limitedlist_probes(),
        train_L=6, hidden_dim=5, epochs=200, lnorm=1000.0,
        jitter=False,
    ),
    # depth-3 stack with error state
    'limitedstack': dict(
        generate=limitedstack_sequences,
        diagram_L=6, diagram_sample=None,  # full enumeration (729 pairs)
        state_labels=LIMITEDSTACK_STATE_LABELS, expected_states=16,
        label_probes=_limitedstack_probes(),
        train_L=9, hidden_dim=5, epochs=500, lnorm=10000.0,
        jitter=False,
        # scatter panels omit the absorbing error state 'E' (it dominates the
        # projection); boxplot accuracies still include it
        plot_states=lambda label: label != 'E',
    ),
}


def verify_state_labels(cd, label_probes):
    """Check that every semantic label sits on the state its probe reaches.

    Raises AssertionError if the hand-assigned label table does not match the
    inferred machine's state numbering.
    """
    for label, inputs in label_probes.items():
        if len(inputs) == 0:
            state = cd.start_state
        else:
            state = cd.trace(inputs)[-1]
        assert cd.state_labels[state] == label, (
            f'State label mismatch: probe {inputs} reaches state {state} '
            f'labeled {cd.state_labels[state]!r}, expected {label!r}')
    assert len(label_probes) == len(cd.state_labels), (
        'Probes do not cover all states')
    print(f'Verified {len(label_probes)} state labels.')
