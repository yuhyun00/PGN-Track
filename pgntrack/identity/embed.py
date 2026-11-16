"""View encoders around an embedding function embed(frame_bgr, boxes_tlwh) -> (N, D) L2-normalised (the re-ID OSNet)."""
import cv2
import numpy as np


class CropEncoder:
    """In-loop: the buffered crops of the views, packed side by side into one image and embedded in one call."""

    def __init__(self, embed):
        self.embed = embed

    def __call__(self, views):
        crops = [v.crop for v in views]
        canvas = np.zeros((max(c.shape[0] for c in crops), sum(c.shape[1] for c in crops), 3), dtype=np.uint8)
        boxes, x = [], 0
        for c in crops:                      # integer boxes: the embedder cuts exactly these pixels back out
            h, w = c.shape[:2]
            canvas[:h, x:x + w] = c
            boxes.append((x, 0, w, h))
            x += w
        return self.embed(canvas, np.asarray(boxes, dtype=np.float64))


class FrameEncoder:
    """Offline (and the pseudo-splits after a video): views cut from the frame files, one call per frame."""

    def __init__(self, embed, frames):
        self.embed, self.frames = embed, frames          # frames[k]: image path of frame k + 1

    def __call__(self, views):
        by_frame = {}
        for i, v in enumerate(views):
            by_frame.setdefault(v.frame, []).append(i)
        out = [None] * len(views)
        for f in sorted(by_frame):
            idx = by_frame[f]
            boxes = np.asarray([views[i].box for i in idx], dtype=np.float64)
            feats = self.embed(cv2.imread(self.frames[f - 1]), boxes)
            for i, x in zip(idx, feats):
                out[i] = x
        return np.asarray(out)


def make_embedder(cfg, device=None, batch_size=16):
    """embed(frame_bgr, boxes_tlwh) of the re-ID network of the `reid` section (McByte++'s sports OSNet)."""
    from ..reid.embedder import OSNetEmbedder        # imported lazily: needs torch / torchreid
    return OSNetEmbedder.from_config(cfg['reid'], device=device, batch_size=batch_size).embed
