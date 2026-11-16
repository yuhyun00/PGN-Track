# Adapted from McByte (MIT) and McByte++ (Apache-2.0): the conditioned assignment of their association steps
# and the box-overlap test that decides whether a mask may be created for a track.
"""McByte's conditioned assignment: isolated pairs are fixed, mask cues re-weight ambiguous pairs."""
import numpy as np

MIN_MM1 = 0.9                # mask coverage by the box ("mc"): below it the mask cue is not used
MIN_MM2 = 0.05               # box fill by the mask ("mf"): the mask cue needs at least this much
ISOLATED_PENALTY = 10        # added to the other entries of an isolated pair's row and column


class ConditionedAssignment:
    """Cost hook of McByte / McByte++ for the three association steps (rows: tracks, columns: detections).

    A pair whose cost is within the step's maximum cost and is the only such entry in its row and column is
    isolated: the rest of its row and column get +10 (McByte++ applies this with and without masks).  The other
    admissible pairs are ambiguous; with masks, the mask fill ratio mf of the detection box is subtracted from
    their cost when mf >= 0.05 and mc >= 0.9 (see `mask_cue` of the mask plugins).
    """

    def association_cost(self, tracker, step, tracks, detections, cost, max_cost):
        valid = cost <= max_cost
        if not valid.any():
            return cost
        isolated = valid & (valid.sum(1) == 1)[:, None] & (valid.sum(0) == 1)[None, :]
        out = np.copy(cost)
        out[isolated.any(1), :] += ISOLATED_PENALTY
        out[:, isolated.any(0)] += ISOLATED_PENALTY
        out[isolated] = cost[isolated]
        frame_hw = getattr(self, 'frame_hw', None)
        for i, j in zip(*np.nonzero(valid & ~isolated)):          # row-major, as the official double loop
            mf = self.mask_cue(tracks[i], detections[j], frame_hw)
            if mf is not None:
                out[i, j] -= mf
        return out

    def mask_cue(self, track, det, frame_hw):
        """Value subtracted from an ambiguous pair's cost, or None (no masks)."""
        return None


def box_bounds(tlwh, img_h, img_w):
    """Integer box (x, y, right, bottom) clipped as McByte does for the mask ratios."""
    x, y, w, h = tlwh
    x = int(x)
    if x < 0:
        x = 0
    y = int(y)
    if y < 0:
        y = 0
    w = int(w)
    hor = img_w if x + w > img_w else x + w
    h = int(h)
    ver = img_h if y + h > img_h else y + h
    return x, y, hor, ver


def boxes_below(tlwh, online_tlwhs):
    """Boxes whose bottom edge is lower in the image than the bottom edge of tlwh."""
    bottom = tlwh[1] + tlwh[3]
    return [ot for ot in online_tlwhs if ot[1] + ot[3] > bottom]


def overlap_with_boxes_below(tlwh, online_tlwhs):
    """Largest intersection with a box below, divided by the area of tlwh (McByte's overlap variant 1)."""
    x, y, w, h = tlwh[0], tlwh[1], tlwh[2], tlwh[3]
    best = 0
    for lb in boxes_below(tlwh, online_tlwhs):
        x_dist = min(x + w, lb[0] + lb[2]) - max(x, lb[0])
        y_dist = min(y + h, lb[1] + lb[3]) - max(y, lb[1])
        area = 0 if (x_dist < 0 or y_dist < 0) else x_dist * y_dist
        part = area / (w * h)
        if best < part:
            best = part
            if best == 1:
                break
    return best
