# Adapted from ByteTrack (MIT): IoU cost and linear assignment.  IoU uses the +1 pixel convention of
# cython_bbox, as in ByteTrack / McByte, so that association decisions are the same.
import lap
import numpy as np


def linear_assignment(cost, thresh):
    """Minimum-cost assignment with a maximum cost; returns (matches (n, 2), unmatched rows, unmatched cols)."""
    if cost.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost.shape[0])), tuple(range(cost.shape[1]))
    _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=thresh)
    matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
    return np.asarray(matches, dtype=int).reshape(-1, 2), np.where(x < 0)[0], np.where(y < 0)[0]


def bbox_ious(a, b):
    """IoU of tlbr boxes a (N, 4) and b (K, 4) -> (N, K), +1 area convention."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]) + 1
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]) + 1
    iw = np.clip(iw, 0, None)
    ih = np.clip(ih, 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    area_b = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(inter > 0, inter / np.maximum(union, 1e-9), 0.0)


def iou_distance(tracks, detections):
    """Cost 1 - IoU between the tracks' predicted boxes and the detections."""
    return 1 - bbox_ious([t.tlbr for t in tracks], [d.tlbr for d in detections])


def fuse_score(cost, detections):
    """ByteTrack's score fusion: 1 - IoU * detection score."""
    if cost.size == 0:
        return cost
    iou_sim = 1 - cost
    scores = np.array([d.score for d in detections])[None, :]
    return 1 - iou_sim * scores
