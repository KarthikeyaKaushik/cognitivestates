"""Cognitive-diagram inference, caching, state labels, and TikZ export.

The cache key must include all task parameters (e.g. nback_n2_k2_Ldiag5) --
the legacy scripts cached per task name only, which silently mixed up e.g.
2-back and 3-back diagrams.
"""
import pickle

from src.cognitive_diagram import veelenturf
from src.paths import DIAGRAMS_DIR, ensure_dir

# State labels keyed by number of states, as hand-assigned in the legacy runners.
NBACK_STATE_LABELS = {
    7: {0: 'e', 1: 'a', 2: 'b', 3: 'aa', 4: 'ab', 5: 'ba', 6: 'bb'},
    15: {0: 'e', 1: 'a', 2: 'b', 3: 'aa', 4: 'ab', 5: 'ba', 6: 'bb',
         7: 'aaa', 8: 'aab', 9: 'aba', 10: 'abb', 11: 'baa', 12: 'bab',
         13: 'bba', 14: 'bbb'},
}
LIMITEDSTACK_STATE_LABELS = {
    16: {0: 'e', 1: 'x', 2: 'y', 3: 'xx', 4: 'xy', 5: 'yx', 6: 'yy',
         7: 'xxx', 8: 'xxy', 9: 'xyx', 10: 'xyy', 11: 'yxx', 12: 'yxy',
         13: 'yyx', 14: 'yyy', 15: 'E'},
}
LIMITEDLIST_STATE_LABELS = {
    9: {0: '--', 1: 'x-', 2: '-x', 3: 'y-', 4: '-y', 5: 'xx', 6: 'xy',
        7: 'yx', 8: 'yy'},
}
LIMITEDSET_STATE_LABELS = {
    8: {0: 'e', 1: 'x', 2: 'y', 3: 'z', 4: 'xy', 5: 'xz', 6: 'yz', 7: 'xyz'},
}
LISTMAX_STATE_LABELS = {
    10: {i: str(i) for i in range(10)},
}


def build_vocab(io_pairs):
    """Input/output token sets from a list of (inputs, outputs) pairs."""
    X = set(token for inp, _ in io_pairs for token in inp)
    Y = set(token for _, out in io_pairs for token in out)
    return X, Y


def label_order(cd):
    """State labels in state-id order -- the canonical hue order for plots."""
    return [cd.state_labels[k] for k in sorted(cd.state_labels)]


def load_or_infer_diagram(cache_key, io_pairs, X, Y, state_labels=None,
                          force=False):
    """Load results/diagrams/<cache_key>.pkl, or run veelenturf and cache it.

    state_labels: dict keyed by number of states (e.g. NBACK_STATE_LABELS);
    the matching entry is attached as cd.state_labels.
    """
    ensure_dir(DIAGRAMS_DIR)
    cache_path = DIAGRAMS_DIR / f'{cache_key}.pkl'
    if cache_path.exists() and not force:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f'Inferring cognitive diagram "{cache_key}" from {len(io_pairs)} io pairs...')
    cd = veelenturf(io_pairs, X, Y)
    print(f'Learned cognitive diagram with {len(cd.states)} states.')
    if state_labels is not None:
        if len(cd.states) not in state_labels:
            raise ValueError(
                f'No state labels for a {len(cd.states)}-state diagram '
                f'(have {sorted(state_labels)}); check the inference data.')
        cd.state_labels = state_labels[len(cd.states)]
    with open(cache_path, 'wb') as f:
        pickle.dump(cd, f)
    return cd


def save_tikz(cd, diagram_name):
    """Write a standalone TikZ .tex of the diagram; returns the path."""
    ensure_dir(DIAGRAMS_DIR)
    latex_document = (
        "\\documentclass{standalone}\n"
        "\\usepackage{tikz}\n"
        "\\begin{document}\n"
        f"{cd.to_tikz()}\n"
        "\\end{document}\n"
    )
    tex_path = DIAGRAMS_DIR / f'{diagram_name}.tex'
    with open(tex_path, 'w') as f:
        f.write(latex_document)
    print(f'Saved TikZ diagram to {tex_path}')
    return tex_path
