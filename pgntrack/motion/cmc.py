# Camera motion compensation (CMC) baselines with ORB keypoints.
#   mcbyte   : ORB at 1/2 resolution, adapted from BoT-SORT's gmc.py as shipped with McByte (MIT), with guards
#              against degenerate inputs (empty masks, single knn neighbours, too few matches);
#              strict: McByte's public code as is (unclipped box masks; where it raises, the tracker skips CMC
#              for the frame and keeps the reference frame).
#   mcbytepp : conditional ORB at 1/4 resolution, adapted from McByte++ (Apache-2.0), yolox/tracker/gmc.py and the
#              warp check of its tracker; the control flow, including its failure paths, is kept.
import warnings

import numpy as np
import cv2

from ..tracker.track import warp_tracks
from .base import MotionStage


class _ORB:
    """FAST keypoints + ORB descriptors outside the detection boxes, matched to the previous frame."""

    def __init__(self, downscale):
        self.downscale = max(1, int(downscale))
        self.detector = cv2.FastFeatureDetector_create(20)
        self.extractor = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.prev_kps, self.prev_desc = None, None
        self.initialized = False

    def _gray(self, raw_frame):
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width, height = width // self.downscale, height // self.downscale
        return frame, width, height

    def _store(self, kps, desc):
        self.prev_kps, self.prev_desc = kps, desc


class McByteORB(_ORB):
    """McByte's CMC: ORB at 1/2, applied in every frame."""

    def __init__(self, downscale=2, strict=False):
        super().__init__(downscale)
        self.strict = strict

    def apply(self, raw_frame, detections):
        """2x3 warp from the previous to the current frame (identity if it cannot be estimated)."""
        if self.strict:
            return self._apply_strict(raw_frame, detections)
        frame, width, height = self._gray(raw_frame)
        H = np.eye(2, 3)
        mask = np.zeros_like(frame)
        mask[int(0.02 * height): int(0.98 * height), int(0.02 * width): int(0.98 * width)] = 255
        for det in detections:
            b = (det[:4] / self.downscale).astype(np.int_)
            mask[max(b[1], 0):max(b[3], 0), max(b[0], 0):max(b[2], 0)] = 0
        kps = self.detector.detect(frame, mask)
        kps, desc = self.extractor.compute(frame, kps)
        if not self.initialized:
            self._store(kps, desc)
            self.initialized = True
            return H
        if self.prev_desc is None or desc is None or len(kps) < 2 or len(self.prev_kps) < 2:
            self._store(kps, desc)
            return H
        matches, dists = [], []
        max_dist = 0.25 * np.array([width, height])
        for mn in self.matcher.knnMatch(self.prev_desc, desc, 2):
            if len(mn) < 2:
                continue
            m, n = mn
            if m.distance < 0.9 * n.distance:
                p0, p1 = self.prev_kps[m.queryIdx].pt, kps[m.trainIdx].pt
                d = (p0[0] - p1[0], p0[1] - p1[1])
                if abs(d[0]) < max_dist[0] and abs(d[1]) < max_dist[1]:
                    dists.append(d)
                    matches.append(m)
        if len(matches) > 4:
            dists = np.asarray(dists)
            inl = (dists - np.mean(dists, 0)) < 2.5 * np.std(dists, 0)
            keep = [i for i in range(len(matches)) if inl[i, 0] and inl[i, 1]]
            prev_pts = np.array([self.prev_kps[matches[i].queryIdx].pt for i in keep])
            curr_pts = np.array([kps[matches[i].trainIdx].pt for i in keep])
            if len(prev_pts) > 4:
                Hr, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, cv2.RANSAC)
                if Hr is not None:
                    H = Hr
                    H[0, 2] *= self.downscale
                    H[1, 2] *= self.downscale
        self._store(kps, desc)
        return H

    def _apply_strict(self, raw_frame, detections):
        """McByte's public gmc.applyFeatures: may raise (single knn neighbour, no descriptors, RANSAC failure)."""
        frame, width, height = self._gray(raw_frame)
        H = np.eye(2, 3)
        mask = np.zeros_like(frame)
        mask[int(0.02 * height): int(0.98 * height), int(0.02 * width): int(0.98 * width)] = 255
        for det in detections:
            b = (det[:4] / self.downscale).astype(np.int_)
            mask[b[1]:b[3], b[0]:b[2]] = 0              # unclipped slice
        kps = self.detector.detect(frame, mask)
        kps, desc = self.extractor.compute(frame, kps)
        if not self.initialized:
            self._store(kps, desc)
            self.initialized = True
            return H
        knn = self.matcher.knnMatch(self.prev_desc, desc, 2)
        if len(knn) == 0:
            self._store(kps, desc)
            return H
        matches, dists = [], []
        max_dist = 0.25 * np.array([width, height])
        for m, n in knn:
            if m.distance < 0.9 * n.distance:
                p0, p1 = self.prev_kps[m.queryIdx].pt, kps[m.trainIdx].pt
                d = (p0[0] - p1[0], p0[1] - p1[1])
                if (np.abs(d[0]) < max_dist[0]) and (np.abs(d[1]) < max_dist[1]):
                    dists.append(d)
                    matches.append(m)
        with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)       # empty match list: NaN statistics, no inliers
            inl = (dists - np.mean(dists, 0)) < 2.5 * np.std(dists, 0)
        keep = [i for i in range(len(matches)) if inl[i, 0] and inl[i, 1]]
        prev_pts = np.array([self.prev_kps[matches[i].queryIdx].pt for i in keep])
        curr_pts = np.array([kps[matches[i].trainIdx].pt for i in keep])
        if np.size(prev_pts, 0) > 4:
            H, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, cv2.RANSAC)
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        self._store(kps, desc)
        return H


class McBytePPORB(_ORB):
    """McByte++'s CMC: ORB at 1/4 with a minimum number of matches; returns None where the official code fails."""

    MIN_GOOD_MATCHES = 12

    def __init__(self, downscale=4):
        super().__init__(downscale)

    def apply(self, raw_frame, detections):
        """2x3 warp, or None (the official code raises there: no CMC in this frame, reference frame kept)."""
        frame, width, height = self._gray(raw_frame)
        H = np.eye(2, 3)
        mask = np.zeros_like(frame)
        mask[int(0.02 * height): int(0.98 * height), int(0.02 * width): int(0.98 * width)] = 255
        for det in detections:
            b = (det[:4] / self.downscale).astype(np.int_)
            mask[b[1]:b[3], b[0]:b[2]] = 0              # unclipped slice, as in the official code
        kps = self.detector.detect(frame, mask)
        kps, desc = self.extractor.compute(frame, kps)
        if desc is None or kps is None or len(kps) == 0:
            self._store(kps, desc)
            return np.eye(2, 3, dtype=np.float32)
        if not self.initialized:
            self._store(kps, desc)
            self.initialized = True
            return H
        if self.prev_desc is None or len(self.prev_kps) == 0:
            self._store(kps, desc)
            return np.eye(2, 3, dtype=np.float32)
        knn = self.matcher.knnMatch(self.prev_desc, desc, 2)
        if len(knn) == 0:
            self._store(kps, desc)
            return H
        if any(len(mn) < 2 for mn in knn):             # official: tuple unpacking raises
            return None
        matches, dists = [], []
        max_dist = 0.25 * np.array([width, height])
        for m, n in knn:
            if m.distance < 0.9 * n.distance:
                p0, p1 = self.prev_kps[m.queryIdx].pt, kps[m.trainIdx].pt
                d = (p0[0] - p1[0], p0[1] - p1[1])
                if np.abs(d[0]) < max_dist[0] and np.abs(d[1]) < max_dist[1]:
                    dists.append(d)
                    matches.append(m)
        if len(dists) < self.MIN_GOOD_MATCHES:
            self._store(kps, desc)
            return np.eye(2, 3, dtype=np.float32)
        dists = np.asarray(dists, dtype=np.float32)
        inl = np.abs(dists - np.mean(dists, axis=0)) < 2.5 * (np.std(dists, axis=0) + 1e-6)
        keep = [i for i in range(len(matches)) if inl[i, 0] and inl[i, 1]]
        prev_pts = np.array([self.prev_kps[matches[i].queryIdx].pt for i in keep])
        curr_pts = np.array([kps[matches[i].trainIdx].pt for i in keep])
        if np.size(prev_pts, 0) > 4 and np.size(prev_pts, 0) == np.size(curr_pts, 0):
            H, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, cv2.RANSAC)
            if H is None:                              # official: scaling None raises
                return None
            if self.downscale > 1:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        self._store(kps, desc)
        return H


def warp_looks_safe(H):
    """McByte++'s check of a warp before it is applied: finite, column norms in [0.3, 3]."""
    if H is None:
        return False
    H = np.asarray(H, dtype=np.float32)
    if H.shape != (2, 3) or not np.isfinite(H).all():
        return False
    sx = float(np.linalg.norm(H[:, 0]))
    sy = float(np.linalg.norm(H[:, 1]))
    return 0.3 <= sx <= 3.0 and 0.3 <= sy <= 3.0


class CMCStage(MotionStage):
    """Warp all predicted tracks with the estimated camera motion."""

    VARIANTS = ('mcbyte', 'mcbytepp')

    def __init__(self, variant='mcbytepp', strict=False):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f'cmc.variant must be one of {self.VARIANTS}, got {variant!r}')
        if strict and variant != 'mcbyte':
            raise ValueError('cmc.strict applies to variant mcbyte (mcbytepp always keeps its failure paths)')
        self.variant = variant
        self.strict = strict
        self.orb = McByteORB(strict=strict) if variant == 'mcbyte' else McBytePPORB()

    def step(self, frame, dets_all, dets_high, tracks, unconfirmed, frame_id):
        if self.variant == 'mcbyte' and self.strict:
            try:                                        # McByte skips CMC for a frame on any error
                H = self.orb.apply(frame, dets_high)
            except Exception:
                H = None
        elif self.variant == 'mcbyte':
            H = self.orb.apply(frame, dets_high)
        else:
            try:                                        # McByte++ skips CMC for a frame on any error
                H = self.orb.apply(frame, dets_high)
            except Exception:
                H = None
            if not warp_looks_safe(H):
                H = None
        if H is None:
            self.warps.append(np.eye(2, 3))
            return
        warp_tracks(tracks, H)
        warp_tracks(unconfirmed, H)
        self.warps.append(np.asarray(H, dtype=np.float64))
