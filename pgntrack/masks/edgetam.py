# Adapted from McByte++ (Apache-2.0), mask_propagation/mask_manager__edgetam.py: EdgeTAM mask propagation with
# the official defaults (no mask removal, no mask refresh).  EdgeTAM (the `sam2` package of the EdgeTAM repository,
# Apache-2.0) is an external dependency.
"""McByte++ masks: EdgeTAM seeded with track boxes and propagated frame by frame, used as an association cue."""
import time

import numpy as np

from .association import ConditionedAssignment, MIN_MM1, MIN_MM2, box_bounds, overlap_with_boxes_below

CREATE_OVERLAP = 0.1          # no mask for a box overlapped >= 10% by a box below it (it waits for a later frame)
START_FRAME = 1               # first frame of the tracker; masks are seeded from its output in the next frame


class EdgeTAMMasks(ConditionedAssignment):
    """Plugin: EdgeTAM masks of the confirmed tracks (begin_frame) and McByte's mask cue (association_cost).

    frame_dir: the sequence's JPEG folder (EdgeTAM loads the video itself); offload: keep the decoded frames and
    the memory bank in host memory (same tensors, moved to the GPU on use; lets several sequences share a GPU).
    """

    def __init__(self, frame_dir: str, checkpoint: str, model_cfg: str = 'configs/edgetam.yaml', offload: bool = False,
                 device: str = 'cuda'):
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        self.torch = torch
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            self.state = self.predictor.init_state(video_path=frame_dir, offload_video_to_cpu=offload,
                                                   offload_state_to_cpu=offload)
        self.awaiting = []                       # track ids whose mask creation was postponed (overlap)
        self.init_delay = 0
        self.iterator = None                     # EdgeTAM propagation generator
        self.last_ids, self.last_logits = None, None
        self.track_mask = {}                     # track id -> mask (object) id; equal ids in McByte++
        self.prediction = None                   # (object ids, mask logits) of the current frame
        self.track_mask_now = {}                 # copy used by the association of the current frame
        self.frame_hw = None
        self._prev = None                        # tracker output of the previous frame
        self.times = []                          # mask time per frame [s]

    # ------------------------------------------------------------------ hooks
    def begin_frame(self, tracker, frame, dets):
        self.frame_hw = frame.shape[:2]
        self.prediction, self.track_mask_now = None, {}
        if tracker.frame_id > 1:
            t0 = time.perf_counter()
            self._update(tracker.frame_id, *self._prev)
            if self.torch.cuda.is_available():
                self.torch.cuda.synchronize()
            self.times.append(time.perf_counter() - t0)

    def end_frame(self, tracker, frame, events):
        online = [t for t in tracker.tracked if t.is_activated]
        self._prev = ([t.last_det_tlwh for t in online], [t.track_id for t in online], list(events.confirmed))

    def mask_cue(self, track, det, frame_hw):
        """Mask fill ratio mf of the detection box when the track's mask covers it (mf >= 0.05, mc >= 0.9)."""
        if self.prediction is None or track.track_id not in self.track_mask_now:
            return None
        mask_ids, logits = self.prediction
        pos = mask_ids.index(self.track_mask_now[track.track_id])
        all_px = (logits[pos][0] > 0.0).sum()
        if not all_px > 0:
            return None
        x, y, hor, ver = box_bounds(det.tlwh, frame_hw[0], frame_hw[1])
        in_box = (logits[pos][0][y:ver, x:hor] > 0.0).sum()
        mc = in_box / all_px
        mf = in_box / ((ver - y) * (hor - x))           # torch tensors, as in the official code
        if mf >= MIN_MM2 and not mc < MIN_MM1:
            return mf
        return None

    # ------------------------------------------------------------------ mask manager
    def _update(self, frame_id, online_tlwhs, online_ids, new_tracks):
        torch = self.torch
        ids, logits = None, None
        if frame_id == START_FRAME + 1 + self.init_delay:
            out = self._initialize(frame_id, online_tlwhs, online_ids)
            if out is not None:
                ids, logits = out
        elif frame_id > START_FRAME + 1 + self.init_delay:
            self._add_new(frame_id, online_tlwhs, online_ids, new_tracks)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                _, ids, logits = next(self.iterator)
        self.last_ids, self.last_logits = ids, logits
        self.prediction = None if ids is None else (ids, logits)
        self.track_mask_now = dict(self.track_mask)

    def _initialize(self, frame_id, online_tlwhs, online_ids):
        """Seed masks from the first frame's confirmed tracks (boxes overlapped from below wait)."""
        torch = self.torch
        boxes, ids = [], []
        for i, ot in enumerate(online_tlwhs):
            if overlap_with_boxes_below(ot, online_tlwhs) >= CREATE_OVERLAP:
                self.awaiting.append(online_ids[i])
                continue
            boxes.append(np.array([ot[0], ot[1], ot[0] + ot[2], ot[1] + ot[3]], dtype=np.float32))
            ids.append(online_ids[i])
        if not boxes:
            self.init_delay += 1              # the official code intends to postpone here (it would fail instead)
            return None
        out_ids, out_logits = None, None
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for tid, box in zip(ids, boxes):
                # frame_id - 2: the previous frame, 0-based in EdgeTAM
                self.predictor.add_new_points_or_box(inference_state=self.state, frame_idx=frame_id - 2, obj_id=tid,
                                                     box=box)
            self.iterator = self.predictor.propagate_in_video(self.state)
            try:
                _, out_ids, out_logits = next(self.iterator)      # previous frame
                _, out_ids, out_logits = next(self.iterator)      # this frame
            except StopIteration:
                pass
        self.track_mask = dict(zip(ids, ids))
        return out_ids, out_logits

    def _add_new(self, frame_id, online_tlwhs, online_ids, new_tracks):
        """Masks for postponed tracks and tracks confirmed in the previous frame: reseed EdgeTAM at that frame."""
        torch = self.torch
        if not new_tracks and not self.awaiting:
            return
        boxes, ids = [], []
        for tid in self.awaiting:
            if tid not in online_ids:
                continue
            tlwh = online_tlwhs[online_ids.index(tid)]
            if overlap_with_boxes_below(tlwh, online_tlwhs) < CREATE_OVERLAP:
                boxes.append(np.array([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]], dtype=np.float32))
                ids.append(tid)
        for tid in ids:
            self.awaiting.remove(tid)
        for nt in new_tracks:
            tlwh = nt.last_det_tlwh
            if overlap_with_boxes_below(tlwh, online_tlwhs) >= CREATE_OVERLAP:
                self.awaiting.append(nt.track_id)
                continue
            boxes.append(np.array([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]], dtype=np.float32))
            ids.append(nt.track_id)
        if not boxes:
            return
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            existing = {}
            for i, oid in enumerate(self.last_ids):
                m = (self.last_logits[i] > 0.0).detach().cpu().numpy()
                existing[oid] = m[0] if m.ndim == 3 else m
            self._reseed(frame_id - 2, existing, boxes, ids)
            self.iterator = self.predictor.propagate_in_video(self.state)
            next(self.iterator)                                   # the reseeded (previous) frame

    def _reseed(self, frame_idx, existing, boxes, ids):
        """Restart EdgeTAM at frame_idx with the existing masks and the new boxes."""
        self.track_mask = {}
        self.predictor.reset_state(self.state)
        for oid, m in existing.items():
            self.predictor.add_new_mask(inference_state=self.state, frame_idx=frame_idx, obj_id=oid,
                                        mask=m.astype(np.float32))
            self.track_mask[oid] = oid
        for tid, box in zip(ids, boxes):
            self.predictor.add_new_points_or_box(inference_state=self.state, frame_idx=frame_idx, obj_id=tid, box=box)
            self.track_mask[tid] = tid
