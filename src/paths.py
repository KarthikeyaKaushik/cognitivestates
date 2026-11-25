"""Repository-relative paths. Import these instead of hardcoding 'results/...'."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
DIAGRAMS_DIR = RESULTS_DIR / 'diagrams'          # cached cognitive-diagram pickles + TikZ
HIDDEN_STATES_DIR = RESULTS_DIR / 'hidden_states'  # cached trained-run artifacts
FIGURES_DIR = ROOT / 'figures'                   # final paper figures


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if needed; returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
