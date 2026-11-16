# Adapted from ByteTrack (MIT) / McByte (MIT): track state with a Kalman filter on (cx, cy, w, h).
import numpy as np


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class Track:
    """A detection, or a track once activated; outputs the box of its last matched detection (McByte)."""

    def __init__(self, tlwh, score):
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.last_det_tlwh = np.asarray(tlwh).copy()   # dtype of the detections until the first update (McByte)
        self.track_id = 0
        self.state = TrackState.New
        self.frame_id = 0               # last frame with a matched detection
        self.start_frame = 0
        self.data = {}                  # per-track state of plugins (see DESIGN.md, Hook API)

    @property
    def end_frame(self):
        return self.frame_id

    def activate(self, kalman_filter, frame_id, track_id):
        """Start a new track from this detection (confirmed at once only in the first frame)."""
        self.kalman_filter = kalman_filter
        self.track_id = track_id
        self.mean, self.covariance = kalman_filter.initiate(tlwh_to_xywh(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, det, frame_id):
        """A lost track matched again."""
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, tlwh_to_xywh(det.tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = det.score
        self.last_det_tlwh = det.tlwh

    def update(self, det, frame_id):
        """A tracked (or unconfirmed) track matched to a detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, tlwh_to_xywh(det.tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = det.score
        self.last_det_tlwh = det.tlwh

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @property
    def tlwh(self):
        """Current box (Kalman state, or the detection box before activation)."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f'Track_{self.track_id}_({self.start_frame}-{self.end_frame})'


def tlwh_to_xywh(tlwh):
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    return ret


def tlbr_to_tlwh(tlbr):
    ret = np.asarray(tlbr).copy()
    ret[2:] -= ret[:2]
    return ret


def multi_predict(tracks, kalman_filter):
    """Kalman prediction of a list of tracks; tracks that are not tracked keep their size (zero size velocity)."""
    if not tracks:
        return
    mean = np.asarray([t.mean.copy() for t in tracks])
    cov = np.asarray([t.covariance for t in tracks])
    for i, t in enumerate(tracks):
        if t.state != TrackState.Tracked:
            mean[i][6] = 0
            mean[i][7] = 0
    mean, cov = kalman_filter.multi_predict(mean, cov)
    for t, m, c in zip(tracks, mean, cov):
        t.mean, t.covariance = m, c


def warp_tracks(tracks, H):
    """Apply a 2x3 camera warp to the Kalman states (as McByte / BoT-SORT do)."""
    if not tracks:
        return
    R = H[:2, :2]
    R8x8 = np.kron(np.eye(4, dtype=float), R)
    t = H[:2, 2]
    for trk in tracks:
        mean = R8x8.dot(trk.mean)
        mean[:2] += t
        trk.mean = mean
        trk.covariance = R8x8.dot(trk.covariance).dot(R8x8.T)
