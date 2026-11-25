"""Statistics helpers: empirical (permutation-style) tests, effect sizes,
multiple-comparison correction. Used for the stats reported in the paper."""
import numpy as np


def empirical_comparison(value, null_values, alternative='less'):
    """Compare one observed value against an empirical null sample.

    Returns a dict with:
      n_null, null_mean, null_sd
      z           -- (value - null_mean) / null_sd  (standardized effect size)
      percentile  -- % of null values strictly below the observed value
      empirical_p -- (1 + #null as-or-more-extreme) / (1 + n_null)
      p_floor     -- smallest p obtainable with this null sample size
    """
    null_values = np.asarray(null_values, dtype=float)
    n = len(null_values)
    mean = null_values.mean()
    sd = null_values.std(ddof=1) if n > 1 else np.nan
    if alternative == 'less':
        extreme = np.sum(null_values <= value)
    elif alternative == 'greater':
        extreme = np.sum(null_values >= value)
    else:  # two-sided: distance from the null mean
        extreme = np.sum(np.abs(null_values - mean) >= abs(value - mean))
    return dict(n_null=int(n),
                null_mean=float(mean),
                null_sd=float(sd),
                z=float((value - mean) / sd) if sd and sd > 0 else np.nan,
                percentile=float(100.0 * np.mean(null_values < value)),
                empirical_p=float((1 + extreme) / (1 + n)),
                p_floor=float(1.0 / (1 + n)))


def holm_correction(pvals):
    """Holm-Bonferroni adjusted p-values, returned in the original order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted
