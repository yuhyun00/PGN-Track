"""Paired per-sequence bootstrap of a mean metric difference (the intervals reported in the paper)."""
import numpy as np

RESAMPLES = 10000
SEED = 0


def paired_bootstrap(a, b, resamples=RESAMPLES, seed=SEED) -> dict:
    """Per-sequence values a, b (same sequence order) -> mean of a - b and its 95% percentile interval."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (resamples, len(d)))].mean(1)
    return dict(mean=float(d.mean()), lo=float(np.percentile(means, 2.5)), hi=float(np.percentile(means, 97.5)),
                up=int((d > 1e-9).sum()), down=int((d < -1e-9).sum()), n=len(d))
