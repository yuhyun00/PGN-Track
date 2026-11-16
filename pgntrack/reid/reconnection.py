# Adapted from McByte++ (Apache-2.0), yolox/tracker/mcbyteplusplus_tracker__with_reid.py (re-ID features, identity
# bank, reconnection, export ids) and tools/demo_track__with_reid.py (held-back output of provisional tracks).
"""McByte++ online re-ID: a young track takes the identity of a terminated track when their OSNet features agree."""
from collections import deque

import numpy as np

from ..tracker.track import TrackState


class ReIDState:
    """Per-track re-ID state (track.data['reid'])."""

    def __init__(self, track_id: int, final: bool, max_feats: int):
        self.feats = deque(maxlen=max_feats)    # L2-normalised OSNet features
        self.avg = None                         # normalised mean of feats
        self.last_frame = -1                    # frame of the last feature
        self.updates = 0                        # update counter (two per feature, as McByte++ counts)
        self.export_id = track_id               # id written to the output
        self.final = final                      # identity settled: no more reconnection, rows are written

    def add(self, feat: np.ndarray, frame_id: int):
        """Add a feature and recompute the normalised mean."""
        self.feats.append(feat)
        stack = np.stack(self.feats, axis=0)
        norms = np.linalg.norm(stack, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        stack = stack / norms
        self.avg = stack.mean(axis=0)
        n = np.linalg.norm(self.avg)
        if n > 0:
            self.avg = self.avg / n
        self.last_frame = frame_id
        self.updates += 1


class Reconnection:
    """Plugin: McByte++'s re-ID reconnection with its output buffering (end_frame, output, finish).

    Each frame, after the state update: (1) tracks that expired from the lost list leave their identity (mean
    feature, if they have >= min_updates updates) in a bank; (2) confirmed tracked tracks add a feature every
    min_frame_gap frames if score >= min_score, area >= min_area, w >= min_w, h >= min_h and the box keeps
    `border` px from the image border; (3) provisional tracks at most new_max_len frames old (with >= min_updates
    updates) and banked identities that are mutual nearest neighbours with cosine similarity >= sim_thresh are
    joined: the track takes the bank's export id; (4) duplicate export ids among tracked tracks are resolved;
    (5) tracks older than new_max_len frames are settled.  Rows of provisional tracks are held back and written
    with the final export id once the identity is settled (or with their own id at the end of the sequence).
    The official code counts each feature twice, so min_updates 2 means one feature.
    release_removed: rows of a track removed while provisional are released at its removal (with its own id) instead
    of at the end of the sequence (for an in-loop plugin after this one; same rows, the order of rows within a frame
    can differ from the official file).
    """

    def __init__(self, embedder, sim_thresh=0.8, max_feats=50, min_frame_gap=5, min_score=0.6, min_area=32 * 32,
                 min_w=25, min_h=50, border=4, min_updates=2, new_max_len=10, release_removed=False):
        self.embedder = embedder
        self.sim_thresh = sim_thresh
        self.max_feats = max_feats
        self.min_frame_gap = min_frame_gap
        self.min_score = min_score
        self.min_area = min_area
        self.min_w, self.min_h = min_w, min_h
        self.border = border
        self.min_updates = min_updates
        self.new_max_len = new_max_len
        self.release_removed = release_removed
        self.ready = []                         # released rows not yet emitted (release_removed)
        self.bank = {}                          # export id -> mean feature of a terminated track (insertion order)
        self.pending = {}                       # track id -> held-back rows (insertion order)
        self.reconnections = []                 # (frame, track id, export id, similarity)

    # ------------------------------------------------------------------ hooks
    def end_frame(self, tracker, frame, events):
        fid = tracker.frame_id
        for t in events.new:
            t.data['reid'] = ReIDState(t.track_id, fid == 1, self.max_feats)
        for t in events.removed:                # lost tracks that expired (unconfirmed tracks are never activated)
            if t.is_activated:
                self._stash(t)
            if self.release_removed and t.track_id in self.pending:
                self.ready += self._flush(t.track_id, t.track_id)
        self._update_features(tracker, frame)
        self._reconnect(tracker)
        self._unique_export_ids(tracker)
        self._settle(tracker)

    def output(self, tracker, tracks, rows):
        """Write rows of settled tracks with their export id; hold back rows of provisional tracks."""
        by_id = {t.track_id: t for t in tracks}
        out, self.ready = self.ready, []
        for row in rows:
            tid = row[1]
            if tid not in by_id:                # a row released by an earlier plugin's finish: already final
                out.append(row)
                continue
            st = by_id[tid].data['reid']
            if not st.final:
                self.pending.setdefault(tid, []).append(row)
                continue
            if tid in self.pending:
                out += self._flush(tid, st.export_id)
            out.append((row[0], int(st.export_id)) + tuple(row[2:]))
        return out

    def held_from(self, tracker):
        """First frame of a row still held back (None if none)."""
        return min(rows[0][0] for rows in self.pending.values()) if self.pending else None

    def finish(self, tracker):
        """Rows still held at the end are written with the track's own id."""
        out, self.ready = self.ready, []
        for tid in list(self.pending):
            out += self._flush(tid, tid)
        return out

    # ------------------------------------------------------------------ steps
    def _flush(self, tid, export_id):
        return [(r[0], int(export_id)) + tuple(r[2:]) for r in self.pending.pop(tid, [])]

    def _stash(self, track):
        st = track.data['reid']
        if st.avg is None or st.updates < self.min_updates:
            return
        self.bank[st.export_id] = st.avg.copy()

    def _update_features(self, tracker, frame):
        img_h, img_w = frame.shape[:2]
        m = self.border
        todo = []
        for t in tracker.tracked:
            if not t.is_activated or t.state != TrackState.Tracked:
                continue
            st = t.data['reid']
            if st.last_frame >= 0 and tracker.frame_id - st.last_frame < self.min_frame_gap:
                continue
            x, y, w, h = t.last_det_tlwh
            if w * h < self.min_area:
                continue
            if w < self.min_w or h < self.min_h:
                continue
            if t.score < self.min_score:
                continue
            if (x <= m) or (y <= m) or ((x + w) >= (img_w - m)) or ((y + h) >= (img_h - m)):
                continue                        # touches the border: likely truncated
            todo.append(t)
        if not todo:
            return
        feats = self.embedder.embed(frame, [t.last_det_tlwh for t in todo])     # per-box dtype kept
        for t, f in zip(todo, feats):
            if f.any():                         # zero row: empty crop
                t.data['reid'].add(f, tracker.frame_id)
                t.data['reid'].updates += 1

    def _active(self, tracker):
        return [t for t in tracker.tracked if t.is_activated and t.state == TrackState.Tracked]

    def _reconnect(self, tracker):
        if not self.bank:
            return
        reserved = {t.data['reid'].export_id for t in tracker.tracked + tracker.lost if 'reid' in t.data}
        young = []
        for t in self._active(tracker):
            st = t.data['reid']
            if st.final or st.avg is None or st.updates < self.min_updates:
                continue
            if tracker.frame_id - t.start_frame > self.new_max_len or st.export_id != t.track_id:
                continue
            young.append(t)
        if not young:
            return
        ids = list(self.bank.items())
        S = np.full((len(young), len(ids)), -1.0, dtype=float)
        for i, t in enumerate(young):
            for j, (eid, feat) in enumerate(ids):
                if eid in reserved:             # never join an id that is in use
                    continue
                sim = float(np.dot(t.data['reid'].avg, feat))
                if sim < self.sim_thresh:
                    continue
                S[i, j] = sim
        if np.all(S < 0):
            return
        best_j = np.argmax(S, axis=1)
        best_sim = S[np.arange(len(young)), best_j]
        best_i = np.argmax(S, axis=0)
        for i in range(len(young)):             # mutual nearest neighbours above the threshold
            j = int(best_j[i])
            sim = float(best_sim[i])
            if sim < self.sim_thresh or S[i, j] < 0 or best_i[j] != i:
                continue
            eid = ids[j][0]
            if eid in reserved:
                continue
            st = young[i].data['reid']
            st.export_id, st.final = eid, True
            reserved.add(eid)
            self.reconnections.append((tracker.frame_id, young[i].track_id, eid, sim))
            del self.bank[eid]

    def _unique_export_ids(self, tracker):
        """Among tracked tracks an export id has one owner: settled, then longer, then higher score wins."""
        def key(t):
            return (1 if t.data['reid'].final else 0, int(t.tracklet_len), float(t.score))
        owner = {}
        for t in self._active(tracker):
            eid = t.data['reid'].export_id
            if eid not in owner:
                owner[eid] = t
                continue
            cur = owner[eid]
            if key(t) > key(cur):
                loser, owner[eid] = cur, t
            else:
                loser = t
            loser.data['reid'].export_id, loser.data['reid'].final = loser.track_id, False

    def _settle(self, tracker):
        """Joined tracks, and tracks older than new_max_len frames, are settled."""
        for t in self._active(tracker):
            st = t.data['reid']
            if st.final:
                continue
            if st.export_id != t.track_id or tracker.frame_id - t.start_frame > self.new_max_len:
                st.final = True
