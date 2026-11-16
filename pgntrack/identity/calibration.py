"""Label-free calibration and decision: kappa from pair pseudo-splits, EM prior, MAP rule (Eqs. 2-4).

Evidence of one video (a dict, JSON-serialisable): {'seq', 'inter': [{'a', 'b', 's', 'e', 'dec', 'D'}, ...] sorted by
(s, e, a, b), 'd0': [D0 of each kept pair pseudo-split, in draw order]}.  'D' is None for a tested interaction
without evidence (no valid visible view on one side).
"""
import json

import numpy as np

from .params import EM_ITERS, EM_PI0, PI_MAX, PI_MIN


def sig(x):
    """Logistic function (argument clipped to +-50)."""
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def logit(p):
    """log(p / (1 - p))."""
    return np.log(p / (1 - p))


def kappa_of(d0, scale=1.0) -> float:
    """kappa = -2 mu0 / sigma0^2 of the label-free H0 sample D0 (Eq. 2)."""
    d0 = np.asarray(d0, dtype=np.float64)
    return scale * (-2 * float(d0.mean()) / float(d0.var()))


def em_prior(lam) -> float:
    """EM fixed point of the exchange prior under fixed likelihood ratios lam (Eq. 3), clipped to [1e-6, 1/2]."""
    lam = np.asarray(lam, dtype=np.float64)
    pi = EM_PI0
    if len(lam):
        for _ in range(EM_ITERS):
            pi = float(np.clip(sig(logit(pi) + lam).mean(), PI_MIN, PI_MAX))
    return pi


def fit(videos, prior='em', kappa_scale=1.0):
    """(kappa, pi) from the evidence of several videos (pooled in the given order)."""
    kappa = kappa_of([d for v in videos for d in v['d0']], kappa_scale)
    if prior != 'em':
        return kappa, float(prior)
    lam = [kappa * x['D'] for v in videos for x in v['inter'] if x['D'] is not None]
    return kappa, em_prior(lam)


def exchange(D, kappa, pi) -> bool:
    """MAP rule: exchange the two labels if logit(pi) + kappa D > 0 (Eq. 4)."""
    return bool(logit(pi) + kappa * D > 0)


def decide(video, kappa, pi):
    """Exchanges [(dec + 1, a, b, record)] of one video's evidence."""
    return [(x['dec'] + 1, x['a'], x['b'], x) for x in video['inter']
            if x['D'] is not None and exchange(x['D'], kappa, pi)]


def calibrate(videos, warm, mode, prior='em', kappa_scale=1.0):
    """Per video (kappa, pi) of a split: online (warm start + earlier videos), batch (the split), cross (warm only)."""
    if mode == 'batch':
        return [fit(videos, prior, kappa_scale)] * len(videos)
    if mode == 'cross':
        return [fit(warm, prior, kappa_scale)] * len(videos)
    if mode == 'online':
        return [fit(list(warm) + list(videos[:k]), prior, kappa_scale) for k in range(len(videos))]
    raise ValueError(f'unknown calibration mode {mode!r}')


class Calibration:
    """Evidence pool of the in-loop layer: the warm start, plus each finished video in online calibration."""

    def __init__(self, videos=(), prior='em', kappa_scale=1.0):
        self.videos, self.prior, self.kappa_scale = list(videos), prior, kappa_scale
        self._fit = None

    @classmethod
    def load(cls, path, prior='em', kappa_scale=1.0):
        """Warm start from an evidence file (tools/identity_offline.py)."""
        with open(path) as f:
            return cls(json.load(f), prior, kappa_scale)

    def add(self, video):
        """Add the evidence of a finished video (online calibration)."""
        self.videos.append(video)
        self._fit = None

    def params(self):
        """Current (kappa, pi)."""
        if self._fit is None:
            self._fit = fit(self.videos, self.prior, self.kappa_scale)
        return self._fit
