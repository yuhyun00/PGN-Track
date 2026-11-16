# Adapted from ByteTrack (MIT) in the configuration of McByte (MIT): thresholds, Kalman filter, and output of the
# last matched detection box.  The camera-motion step is a pluggable motion stage, and plugins (masks, re-ID,
# identity inference) attach through the hooks documented in DESIGN.md ("Hook API").
import time
from dataclasses import dataclass, field

import numpy as np

from . import matching
from .kalman import KalmanFilter
from .track import Track, TrackState, multi_predict, tlbr_to_tlwh

LOW_SCORE = 0.1          # detections at or below this score are ignored (ByteTrack)
BIRTH_MARGIN = 0.1       # a new track needs score >= track_thresh + BIRTH_MARGIN


@dataclass
class FrameEvents:
    """What happened to the tracks in one frame (passed to `end_frame` of the plugins)."""
    frame_id: int
    activated: list = field(default_factory=list)   # tracked/unconfirmed tracks matched, and new tracks
    refound: list = field(default_factory=list)     # lost tracks matched again
    lost: list = field(default_factory=list)        # tracks that became lost
    removed: list = field(default_factory=list)     # unconfirmed tracks dropped and lost tracks expired
    new: list = field(default_factory=list)         # tracks born in this frame
    confirmed: list = field(default_factory=list)   # unconfirmed tracks matched in this frame (McByte's new tracks)
    matched: list = field(default_factory=list)     # high-score matches (first and unconfirmed steps), then births


class Tracker:
    """ByteTrack: high-score association, low-score association, unconfirmed tracks, births."""

    STEPS = ('first', 'second', 'unconfirmed')

    def __init__(self, motion, track_thresh=0.6, track_buffer=30, frame_rate=30, match_thresh=(0.9, 0.5, 0.7),
                 min_box_area=None, aspect_ratio_thresh=None, plugins=()):
        self.motion = motion
        self.plugins = list(plugins)
        self.track_thresh = track_thresh
        self.det_thresh = track_thresh + BIRTH_MARGIN
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.match_thresh = tuple(match_thresh)      # maximum cost per step (first, second, unconfirmed)
        self.min_box_area = min_box_area             # ByteTrack's output filter (None = off)
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.kalman_filter = KalmanFilter()
        self.tracked, self.lost, self.removed = [], [], []
        self.frame_id = 0
        self._last_id = 0
        self.frame_times = []                        # per frame: (motion stage, association and bookkeeping after it) [s]

    # ------------------------------------------------------------------ hooks
    def _hook(self, name, *args):
        for p in self.plugins:
            fn = getattr(p, name, None)
            if fn is not None:
                fn(self, *args)

    def _assign(self, step, tracks, detections, cost):
        """Plugins may change the cost of an association step; then linear assignment with its maximum cost."""
        thresh = self.match_thresh[self.STEPS.index(step)]
        for p in self.plugins:
            fn = getattr(p, 'association_cost', None)
            if fn is not None:
                cost = fn(self, step, tracks, detections, cost, thresh)
        return matching.linear_assignment(cost, thresh)

    def _new_id(self):
        self._last_id += 1
        return self._last_id

    # ------------------------------------------------------------------ one frame
    def update(self, dets, frame):
        """dets: (N, 5) x1, y1, x2, y2, score in pixels; frame: BGR image. Returns the confirmed tracked tracks."""
        self.frame_id += 1
        ev = FrameEvents(self.frame_id)
        self._hook('begin_frame', frame, dets)

        scores, boxes = dets[:, 4], dets[:, :4]
        high = scores > self.track_thresh
        low = np.logical_and(scores > LOW_SCORE, scores < self.track_thresh)
        dets_high = boxes[high]
        detections = [Track(tlbr_to_tlwh(b), s) for b, s in zip(dets_high, scores[high])]

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in self.tracked if t.is_activated]
        pool = joint_tracks(tracked, self.lost)
        multi_predict(pool, self.kalman_filter)

        # motion stage: camera compensation and/or photometric measurements of the predicted tracks
        t1 = time.perf_counter()
        self.motion.step(frame, boxes[scores > LOW_SCORE], dets_high, pool, unconfirmed, self.frame_id)
        t2 = time.perf_counter()

        # first association: tracked and lost tracks with high-score detections
        cost = matching.fuse_score(matching.iou_distance(pool, detections), detections)
        matches, u_track, u_det = self._assign('first', pool, detections, cost)
        for it, idet in matches:
            trk, det = pool[it], detections[idet]
            if trk.state == TrackState.Tracked:
                trk.update(det, self.frame_id)
                ev.activated.append(trk)
            else:
                trk.re_activate(det, self.frame_id)
                ev.refound.append(trk)
            ev.matched.append(trk)

        # second association: remaining tracked tracks with low-score detections
        detections_low = [Track(tlbr_to_tlwh(b), s) for b, s in zip(boxes[low], scores[low])]
        remain = [pool[i] for i in u_track if pool[i].state == TrackState.Tracked]
        cost = matching.iou_distance(remain, detections_low)
        matches, u_remain, _ = self._assign('second', remain, detections_low, cost)
        for it, idet in matches:
            remain[it].update(detections_low[idet], self.frame_id)
            ev.activated.append(remain[it])
        for it in u_remain:                                  # tracked tracks without any detection become lost
            remain[it].mark_lost()
            ev.lost.append(remain[it])

        # unconfirmed tracks (one frame old) with the remaining high-score detections
        detections = [detections[i] for i in u_det]
        cost = matching.fuse_score(matching.iou_distance(unconfirmed, detections), detections)
        matches, u_unconf, u_det = self._assign('unconfirmed', unconfirmed, detections, cost)
        for it, idet in matches:
            unconfirmed[it].update(detections[idet], self.frame_id)
            ev.activated.append(unconfirmed[it])
            ev.matched.append(unconfirmed[it])
            ev.confirmed.append(unconfirmed[it])
        for it in u_unconf:
            unconfirmed[it].mark_removed()
            ev.removed.append(unconfirmed[it])

        # births
        for i in u_det:
            trk = detections[i]
            if trk.score < self.det_thresh:
                continue
            trk.activate(self.kalman_filter, self.frame_id, self._new_id())
            ev.activated.append(trk)
            ev.new.append(trk)

        # lost tracks beyond the buffer are removed
        for trk in self.lost:
            if self.frame_id - trk.end_frame > self.max_time_lost:
                trk.mark_removed()
                ev.removed.append(trk)

        self.tracked = [t for t in self.tracked if t.state == TrackState.Tracked]
        self.tracked = joint_tracks(self.tracked, ev.activated)
        self.tracked = joint_tracks(self.tracked, ev.refound)
        self.lost = sub_tracks(self.lost, self.tracked)
        self.lost.extend(ev.lost)
        self.lost = sub_tracks(self.lost, self.removed)
        self.removed.extend(ev.removed)

        ev.matched += ev.new
        self.motion.post_update(frame, ev.matched, ev.removed, self.frame_id)
        self._hook('end_frame', frame, ev)
        self.frame_times.append((t2 - t1, time.perf_counter() - t2))
        return [t for t in self.tracked if t.is_activated]

    # ------------------------------------------------------------------ output
    def output(self, tracks):
        """Rows (frame, id, x, y, w, h, score) of this frame: last matched detection box, output filter, plugins."""
        rows = []
        for t in tracks:
            x, y, w, h = t.last_det_tlwh
            if self.min_box_area is not None and not (
                    w * h > self.min_box_area and
                    (self.aspect_ratio_thresh is None or w / max(h, 1e-9) <= self.aspect_ratio_thresh)):
                continue
            rows.append((self.frame_id, t.track_id, x, y, w, h, t.score))
        for p in self.plugins:
            fn = getattr(p, 'output', None)
            if fn is not None:
                rows = fn(self, tracks, rows)
        return rows

    def finish(self):
        """Rows that plugins still hold at the end of the sequence (passed through the `output` of later plugins)."""
        rows = []
        for p in self.plugins:
            if rows and getattr(p, 'output', None) is not None:
                rows = p.output(self, [], rows)
            fn = getattr(p, 'finish', None)
            if fn is not None:
                rows = rows + fn(self)
        return rows


def joint_tracks(a, b):
    """a followed by the tracks of b that are not in a (by id)."""
    seen = {t.track_id for t in a}
    out = list(a)
    for t in b:
        if t.track_id not in seen:
            seen.add(t.track_id)
            out.append(t)
    return out


def sub_tracks(a, b):
    """Tracks of a whose id is not in b."""
    out = {t.track_id: t for t in a}
    for t in b:
        out.pop(t.track_id, None)
    return list(out.values())
