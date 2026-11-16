"""Identity inference inside the tracker loop (supp. Alg. S1, steps 8-10): a tracker plugin of the hook API.

The plugin must come last: it consumes the rows emitted by the tracker and the plugins before it, one complete frame
at a time.  A plugin that holds rows back (McByte++'s provisional ids) exposes `held_from(tracker)`, the first frame
of a row it still holds (None if none), and releases a row as soon as its id is settled; frame f is processed once
no plugin holds a row of a frame <= f, with the frame image kept until then.  The crops of visible views are
buffered, each needed view is embedded as it arrives (pre views when the interaction reaches the test overlap, post
views on arrival; one embedding call per frame), and the rows of frame t are emitted with the current labels.
kappa and pi are fixed during a video; with online calibration, the pair pseudo-splits of the finished video (views
read back from the frame files) and its interactions are added to the calibration for the next video in the same
process.
"""
import time

from .calibration import Calibration, exchange
from .embed import CropEncoder, FrameEncoder
from .layer import IdentityLayer
from .params import PAIR_SEED, IdentityParams


def r2(v) -> float:
    """A value as written to (and read back from) a result file."""
    return float(f'{v:.2f}')


def dedupe(rows):
    """One row per (frame, id): the higher-score one (the first on ties), in the given order."""
    best = {}
    for i, r in enumerate(rows):
        k = (r[0], r[1])
        if k not in best or r2(r[6]) > r2(rows[best[k]][6]):
            best[k] = i
    return [rows[i] for i in sorted(best.values())]


class IdentityPlugin:
    """Relabels the output rows of one video; per-frame cost in `times` [s], calibration update in `calib_time`."""

    def __init__(self, params, calibration, embed, seq='', frames=(), seed=PAIR_SEED, update=False):
        self.p, self.calibration, self.embed = params, calibration, embed
        self.seq, self.frames, self.seed, self.update = seq, frames, seed, update
        self.kappa, self.pi = calibration.params()
        self.layer = None
        self.stream, self.images = {}, {}
        self.done = 0
        self.times, self.calib_time = [], 0.0

    def begin_frame(self, tracker, frame, dets):
        """Keep the frame until its rows are complete."""
        if self.layer is None:
            self.layer = IdentityLayer(self.p, (frame.shape[1], frame.shape[0]), CropEncoder(self.embed),
                                       lambda D: exchange(D, self.kappa, self.pi))
        self.images[tracker.frame_id] = frame

    def output(self, tracker, tracks, rows):
        """Rows of the frames that became complete, with their output labels."""
        t0 = time.perf_counter()
        for r in rows:
            if r[0] <= self.done:
                raise RuntimeError(f'row of frame {r[0]} after the frame was processed: a plugin before the identity '
                                   'plugin holds rows back without held_from (see DESIGN.md)')
            self.stream.setdefault(int(r[0]), []).append(r)
        upstream = tracker.plugins[:tracker.plugins.index(self)]
        held = [h for h in (p.held_from(tracker) for p in upstream if hasattr(p, 'held_from')) if h is not None]
        out = self._advance(min(held) - 1 if held else tracker.frame_id)
        self.times.append(time.perf_counter() - t0)
        return out

    def finish(self, tracker):
        """Rows of the remaining frames; online: add this video to the calibration."""
        out = self._advance(tracker.frame_id)
        if self.layer is not None:
            self.layer.finish()
            if self.update:
                t0 = time.perf_counter()
                d0 = self.layer.pair_d0(self.seed, FrameEncoder(self.embed, self.frames))
                self.calibration.add(self.layer.evidence(self.seq, d0))
                self.calib_time = time.perf_counter() - t0
        return out

    def _advance(self, upto):
        """Process the complete frames up to `upto`; returns their rows with the output labels."""
        out = []
        while self.done < upto:
            f = self.done + 1
            rows = dedupe(self.stream.pop(f, []))
            labels, _ = self.layer.frame(f, [r[1] for r in rows], [[r2(v) for v in r[2:6]] for r in rows],
                                         self.images.pop(f, None))
            out += [(f, label) + tuple(r[2:]) for r, label in zip(rows, labels)]
            self.done = f
        return out


_CALIBRATIONS = {}


def build_identity_plugin(cfg, dataset, seq, embedder):
    """Identity plugin of one sequence from the `identity` section (None if off); embedder: the re-ID embedder."""
    # online / cross: kappa and pi from identity.calib (evidence of the warm-start dataset, tools/identity_offline.py);
    # online also adds each finished video to the calibration of this process.  batch needs every video of the
    # split before any decision: offline only.
    p = IdentityParams.from_config(cfg.get('identity'))
    if p.type == 'none':
        return None
    if p.type == 'batch':
        raise ValueError('batch calibration is offline only (tools/identity_offline.py)')
    if not p.calib:
        raise ValueError('identity.calib: evidence file of the warm-start dataset (tools/identity_offline.py)')
    key = (p.calib, p.prior, p.kappa_scale)
    if key not in _CALIBRATIONS:
        _CALIBRATIONS[key] = Calibration.load(p.calib, p.prior, p.kappa_scale)
    return IdentityPlugin(p, _CALIBRATIONS[key], getattr(embedder, 'embed', embedder), seq, dataset.frames(seq),
                          PAIR_SEED + dataset.seqs.index(seq), update=p.type == 'online')
