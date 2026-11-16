"""The identity layer on saved tracker outputs (how the paper's accuracy numbers are produced).

1. evidence of each video: the layer replayed over the result file (no decisions), then its pair pseudo-splits;
2. (kappa, pi) per video: online, batch or cross calibration (calibration.calibrate);
3. exchanges (dec + 1, a, b) by the MAP rule, applied to the result file as a running permutation of labels.
For fixed kappa and pi, these are the in-loop plugin's outputs.
"""
import numpy as np

from .calibration import calibrate, decide
from .layer import IdentityLayer
from .plugin import dedupe


def read_rows(path):
    """Rows (frame, id, x, y, w, h, score) of a result file, in file order (one per id and frame)."""
    a = np.loadtxt(path, delimiter=',', ndmin=2)
    return dedupe([(int(r[0]), int(r[1])) + tuple(float(v) for v in r[2:7]) for r in a])


def sequence_evidence(rows, size, params, encode, seq, seed):
    """Evidence of one video (interactions and pair pseudo-splits); size: (width, height) of its frames."""
    layer = IdentityLayer(params, size, encode)
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r[0], []).append(r)
    for t in range(1, max(by_frame, default=0) + 1):
        fr = by_frame.get(t, [])
        layer.frame(t, [r[1] for r in fr], [r[2:6] for r in fr])
    layer.finish()
    return layer.evidence(seq, layer.pair_d0(seed))


def relabel(rows, exchanges):
    """Rows with their output labels; exchanges [(f0, a, b)] swap the labels of tracks a, b from frame f0 on."""
    ev = sorted(exchanges)
    cur, k, labels = {}, 0, [None] * len(rows)
    for i in sorted(range(len(rows)), key=lambda i: rows[i][0]):
        f, q = rows[i][0], rows[i][1]
        while k < len(ev) and ev[k][0] <= f:
            _, a, b = ev[k]
            cur[a], cur[b] = cur.get(b, b), cur.get(a, a)
            k += 1
        labels[i] = cur.get(q, q)
    return [(r[0], label) + tuple(r[2:]) for r, label in zip(rows, labels)]


def decide_split(videos, warm, params):
    """Per video: (kappa, pi, [(dec + 1, a, b, record)]) under the calibration of `params.type`."""
    fits = calibrate(videos, warm, params.type, params.prior, params.kappa_scale)
    return [(kappa, pi, decide(v, kappa, pi)) for v, (kappa, pi) in zip(videos, fits)]
