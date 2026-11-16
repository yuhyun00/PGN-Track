"""Interface of the motion stage, which runs between the Kalman prediction and the association."""
import numpy as np


class MotionStage:
    """Moves the predicted tracks into the current frame (camera) and/or measures them (photometric updates).

    step()        called once per frame after the Kalman prediction of `tracks` (tracked and lost tracks);
                  `unconfirmed` tracks are not predicted (ByteTrack).  Edits track.mean / track.covariance.
    post_update() called after association with the high-score matches and births of the frame (`matched`)
                  and the tracks removed in the frame.
    warps         the 2x3 camera warp applied in each frame (identity when none), for plugins and diagnostics.
    """

    def __init__(self):
        self.warps = []

    def step(self, frame, dets_all, dets_high, tracks, unconfirmed, frame_id):
        """frame: BGR image; dets_all: (n, 4) boxes with score > 0.1; dets_high: (m, 4) high-score boxes."""
        raise NotImplementedError

    def post_update(self, frame, matched, removed, frame_id):
        pass


class NoMotion(MotionStage):
    """ByteTrack: no camera motion compensation."""

    def step(self, frame, dets_all, dets_high, tracks, unconfirmed, frame_id):
        self.warps.append(np.eye(2, 3))
