"""The identity layer over the tracker's output, one complete output frame at a time (Sec. 4, supp. Alg. S1).

Interactions: pairs of output boxes with IoU > 0.2; overlaps at most 3 frames apart form one interaction [t_s, t_e],
closed 4 frames after its last overlap and tested if its IoU reaches `test_iou`.  Visible view: an output box whose
IoU with every other box of its frame is below `visible_iou`.  Prototypes: normalised mean embedding of the last M
visible views of a track in the L frames before t_s (fixed when the interaction opens) and of its first M visible
views in the L frames after t_e.  Decision: as soon as both post-prototypes are complete, else at t_e + L
(dec = the frame of the later M-th view, else t_e + L); an exchange relabels the two physical tracks from dec + 1 on.
Physical tracks are the ids of the consumed output; the output labels are a running permutation of them.
"""
from collections import deque
from typing import NamedTuple

import numpy as np

from .params import BUSY_MARGIN, INTERACT_IOU, MAX_GAP, MIN_CROP, PAIR_DRAWS, PAIR_GAPS, PAIR_MARGIN


class View(NamedTuple):
    """One output box of a track: the frame, the track, its box (tlwh as written to the result file)."""
    frame: int
    track: int
    box: tuple
    valid: bool                  # crop inside the image at least MIN_CROP x MIN_CROP px
    crop: object = None          # BGR crop (in-loop) or None (cut from the frame file when embedded)


def iou(a, b):
    """IoU matrix of tlwh boxes."""
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - inter + 1e-9)


def crop_window(box, size):
    """Integer crop window (x0, y0, x1, y1) of a tlwh box clipped to an image of size (width, height)."""
    x, y, w, h = box
    return int(max(0, x)), int(max(0, y)), int(min(size[0], x + w)), int(min(size[1], y + h))


def make_view(frame, track, box, size, image=None) -> View:
    """View of a box; its crop is cut from `image` if one is given."""
    x0, y0, x1, y1 = crop_window(box, size)
    valid = x1 - x0 >= MIN_CROP and y1 - y0 >= MIN_CROP
    crop = image[y0:y1, x0:x1].copy() if valid and image is not None else None
    return View(frame, track, tuple(box), valid, crop)


def double_difference(pa, pb, qa, qb) -> float:
    """D = psi_ab + psi_ba - psi_aa - psi_bb of pre-prototypes (pa, pb) and post-prototypes (qa, qb) (Eq. 1)."""
    return float(pa @ qb) + float(pb @ qa) - float(pa @ qa) - float(pb @ qb)


class Interaction:
    """An interaction of physical tracks a < b and its views."""

    def __init__(self, a, b, s, pre_a, pre_b):
        self.a, self.b, self.s, self.e = a, b, s, s
        self.maxov, self.armed = 0.0, False
        self.pre, self.post = (pre_a, pre_b), ([], [])
        self.dec = None


class IdentityLayer:
    """Online identity layer of one video (size: (width, height) of its frames)."""

    def __init__(self, params, size, encode, decide=None):
        # encode(views) -> (n, D) L2-normalised embeddings; decide(D) -> bool, or None for evidence only
        self.p, self.size, self.encode, self.decide = params, size, encode, decide
        self.buf = {}                  # track -> last visible views
        self.open = {}                 # (a, b) -> open interaction
        self.post = []                 # closed tested interactions collecting post views
        self.cache = {}                # (track, frame) -> embedding
        self.pending = {}              # (track, frame) -> view to embed at the end of the frame
        self.cur = {}                  # physical track -> output label
        self.records, self.tested = [], []         # records: every tested interaction (D or None)
        self.visible, self.life = {}, {}           # (track, frame) -> visible view; track -> (first, last) frame
        self.n_embedded = 0                        # views embedded (embedding load)

    # ------------------------------------------------------------------ embeddings
    def _need(self, views):
        for v in views:
            if v.valid and (v.track, v.frame) not in self.cache:
                self.pending[(v.track, v.frame)] = v

    def _embed_pending(self):
        if self.pending:
            keys, views = list(self.pending), list(self.pending.values())
            self.pending = {}
            for k, f in zip(keys, np.asarray(self.encode(views), dtype=np.float64)):
                self.cache[k] = f
            self.n_embedded += len(keys)

    def prototype(self, views):
        """Normalised mean embedding of the valid views (None if there is none)."""
        f = [self.cache[(v.track, v.frame)] for v in views if v.valid]
        if not f:
            return None
        m = np.mean(f, 0)
        return m / max(np.linalg.norm(m), 1e-12)

    # ------------------------------------------------------------------ interactions
    def _close(self, it):
        if it.maxov < self.p.test_iou:
            return
        self.tested.append(it)
        if it.pre[0] and it.pre[1]:
            it.post = tuple([v for v in self.buf.get(x, ()) if v.frame > it.e][:self.p.views] for x in (it.a, it.b))
            self._need(it.post[0] + it.post[1])
            self.post.append(it)
        else:
            self._record(it, None)

    def _record(self, it, D):
        self.records.append(dict(a=it.a, b=it.b, s=it.s, e=it.e, dec=it.dec, D=D))

    def _due(self, t, final=False):
        """Interactions whose post-prototypes are complete, or that reached t_e + L (all of them at the end)."""
        M, L = self.p.views, self.p.window
        due = []
        for it in list(self.post):
            full = len(it.post[0]) == M and len(it.post[1]) == M
            if final or full or t >= it.e + L:
                self.post.remove(it)
                if it.post[0] and it.post[1]:
                    it.dec = int(max(it.post[0][-1].frame if len(it.post[0]) == M else it.e + L,
                                     it.post[1][-1].frame if len(it.post[1]) == M else it.e + L, it.e + MAX_GAP + 1))
                    due.append(it)
                else:
                    self._record(it, None)
        return due

    def _decide(self, due):
        swaps = []
        for it in due:
            P = [self.prototype(v) for v in it.pre + it.post]
            D = None if any(x is None for x in P) else double_difference(*P)
            self._record(it, D)
            if D is not None and self.decide is not None and self.decide(D):
                swaps.append((it.dec + 1, it.a, it.b))
        return sorted(swaps)

    # ------------------------------------------------------------------ one frame
    def frame(self, t, ids, boxes, image=None):
        """Output labels of frame t's rows and the exchanges decided at t (applied from t + 1)."""
        # ids, boxes: physical ids and tlwh boxes of the complete output frame t (as in the result file);
        # image: the BGR frame in the loop (crops are buffered), None offline (crops are read when embedded)
        M, L = self.p.views, self.p.window
        T = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        ids = [int(q) for q in ids]
        o = iou(T, T)
        np.fill_diagonal(o, 0)
        new = []
        for i, q in enumerate(ids):
            first, _ = self.life.get(q, (t, t))
            self.life[q] = (first, t)
            if len(ids) == 1 or o[i].max() < self.p.visible_iou:
                v = make_view(t, q, T[i], self.size, image)
                new.append(v)
                self.visible[(q, t)] = v._replace(crop=None)
        for i, j in zip(*np.nonzero(np.triu(o) > INTERACT_IOU)):
            a, b = sorted((ids[i], ids[j]))
            it = self.open.get((a, b))
            if it is not None and t - it.e > MAX_GAP + 1:      # a frame was skipped: the old interaction has ended
                self._close(self.open.pop((a, b)))
                it = None
            if it is None:
                pre = [[v for v in reversed(self.buf.get(x, ())) if v.frame >= t - L][:M] for x in (a, b)]
                it = self.open[(a, b)] = Interaction(a, b, t, *pre)
            it.e = t
            it.maxov = max(it.maxov, float(o[i, j]))
            if not it.armed and it.maxov >= self.p.test_iou:   # pre views are embedded when the test overlap is reached
                it.armed = True
                if it.pre[0] and it.pre[1]:
                    self._need(it.pre[0] + it.pre[1])
        for v in new:
            self.buf.setdefault(v.track, deque(maxlen=self.p.buffer)).append(v)
        for it in self.post:                                    # post views as they arrive
            for v in new:
                if v.track in (it.a, it.b) and it.e < t <= it.e + L:
                    w = it.post[0] if v.track == it.a else it.post[1]
                    if len(w) < M:
                        w.append(v)
                        self._need([v])
        for key in [k for k, it in self.open.items() if t - it.e >= MAX_GAP + 1]:
            self._close(self.open.pop(key))
        due = self._due(t)
        self._embed_pending()
        swaps = self._decide(due)
        labels = [self.cur.get(q, q) for q in ids]
        self._apply(swaps)
        return labels, swaps

    def finish(self):
        """End of the video: close every interaction and decide the rest (their exchanges change no output)."""
        for key in list(self.open):
            self._close(self.open.pop(key))
        due = self._due(None, final=True)
        for it in due:
            self._need(it.pre[0] + it.pre[1] + it.post[0] + it.post[1])
        self._embed_pending()
        swaps = self._decide(due)
        self._apply(swaps)
        self.buf, self.open, self.post = {}, {}, []
        return swaps

    def _apply(self, swaps):
        for _, a, b in swaps:
            self.cur[a], self.cur[b] = self.cur.get(b, b), self.cur.get(a, a)

    # ------------------------------------------------------------------ calibration samples of the finished video
    def pair_splits(self, seed):
        """Kept pair pseudo-splits (t, u, tau, gap, [pre_t, pre_u, post_t, post_u]) of the finished video."""
        # 200 draws (default_rng(seed)): a track t living more than 2 L + 32 frames, tau at least L frames from both
        # ends of its life, gap uniform in 1..29, and a track u alive over [tau - L, tau + gap + L); both must be free
        # of tested interactions (+-2 frames) over this window and have a valid visible view in all four windows
        M, L = self.p.views, self.p.window
        busy = {(x, f) for it in self.tested for x in (it.a, it.b)
                for f in range(it.s - BUSY_MARGIN, it.e + BUSY_MARGIN + 1)}
        ids = sorted(self.life)
        rng = np.random.default_rng(seed)

        def views(x, frames):
            return [self.visible[(x, f)] for f in frames if (x, f) in self.visible][:M]
        out = []
        for _ in range(PAIR_DRAWS if ids else 0):
            t = int(rng.choice(ids))
            first, last = self.life[t]
            if last - first <= 2 * L + PAIR_MARGIN:
                continue
            tau = int(rng.integers(first + L, last - L))
            gap = int(rng.integers(*PAIR_GAPS))
            lo, hi = tau - L, tau + gap + L
            if any((t, f) in busy for f in range(lo, hi)):
                continue
            others = [u for u in ids if u != t and self.life[u][0] <= lo and self.life[u][1] >= hi - 1
                      and not any((u, f) in busy for f in range(lo, hi))]
            if not others:
                continue
            u = int(rng.choice(others))
            V = [views(x, range(tau - 1, tau - L - 1, -1)) for x in (t, u)] + \
                [views(x, range(tau + gap, tau + gap + L)) for x in (t, u)]
            if all(V) and all(any(v.valid for v in w) for w in V):
                out.append((t, u, tau, gap, V))
        return out

    def pair_d0(self, seed, encode=None):
        """D0 = psi_tu + psi_ut - psi_tt - psi_uu of each pair pseudo-split (views not embedded yet use `encode`)."""
        draws = self.pair_splits(seed)
        if encode is not None:
            self.encode = encode
        self._need([v for d in draws for w in d[4] for v in w])
        self._embed_pending()
        return [double_difference(*[self.prototype(w) for w in d[4]]) for d in draws]

    def evidence(self, seq, d0=None):
        """Evidence of the video: tested interactions sorted by (t_s, t_e, a, b) and the pair samples D0."""
        inter = sorted(self.records, key=lambda x: (x['s'], x['e'], x['a'], x['b']))
        return dict(seq=seq, inter=inter, d0=list(d0 or []))
