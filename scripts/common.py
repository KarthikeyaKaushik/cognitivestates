"""Shared figure-script utilities: path bootstrap, styling, saving."""
import sys
from pathlib import Path

# Make `src` importable when running `python scripts/figN_*.py` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from src.paths import FIGURES_DIR, ensure_dir  # noqa: E402


def save_figure(name, dpi=300):
    """Save the current matplotlib figure to figures/<name>.png."""
    ensure_dir(FIGURES_DIR)
    out_path = FIGURES_DIR / f'{name}.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    print(f'Saved figure to {out_path}')
    return out_path
