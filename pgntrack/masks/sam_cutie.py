# Adapted from McByte (MIT), mask_propagation/mask_manager.py and tools/demo_track.py: SAM masks seeded from track
# boxes, propagated by Cutie.  SAM (`segment_anything`, Apache-2.0) and McByte's copy of Cutie (MIT; the `cutie` and
# `gui` packages of McByte's mask_propagation/Cutie, which adds object purging) are external dependencies.
"""McByte masks: SAM (ViT-B) creates masks from track boxes, Cutie propagates them, used as an association cue."""
import os
import time

import numpy as np

from .association import ConditionedAssignment, MIN_MM1, MIN_MM2, box_bounds, overlap_with_boxes_below

CREATE_OVERLAP = 0.6          # no mask for a box overlapped >= 60% by a box below it (it waits for a later frame)
MIN_MASK_AVG_CONF = 0.6       # mean Cutie probability over a mask's pixels needed to use its cue
START_FRAME = 1


class McByteMasks(ConditionedAssignment):
    """Plugin: SAM + Cutie masks of the confirmed tracks (begin_frame) and McByte's mask cue (association_cost).

    All mask work runs under torch.inference_mode() and CUDA autocast, as McByte's demo runs its whole loop.
    """

    def __init__(self, sam_checkpoint: str, cutie_weights: str, device: str = 'cuda'):
        import torch
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import open_dict
        from segment_anything import SamPredictor, sam_model_registry
        import cutie
        from cutie.inference.inference_core import InferenceCore
        from cutie.inference.utils.args_utils import get_dataset_cfg
        from cutie.model.cutie import CUTIE
        from gui.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask
        self.torch = torch
        self._to_torch, self._to_mask = image_to_torch, torch_prob_to_numpy_mask
        self._one_hot = index_numpy_to_one_hot_torch
        self.device = device
        self.awaiting = []
        self.init_delay = 0
        self.track_mask = None                   # track id -> mask number (1, 2, ...; shifted on removal)
        self.track_mask_now = None               # copy used by the association of the current frame
        self.objects = []                        # Cutie object ids
        self.last_object = 0
        self.num_objects = 0
        self.prev_mask = None                    # index mask of the previous frame
        self.prediction, self.avg_prob, self.uniq = None, None, []
        self.frame_hw = None
        self._prev = None                        # (frame, tracker output) of the previous frame
        self.times = []
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            np.random.seed(0)                    # as McByte (SAM)
            sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
            sam.to(device=device)
            self.sam = sam
            self.sam_predictor = SamPredictor(sam)
            config_dir = os.path.join(list(cutie.__path__)[0], 'config')
            GlobalHydra.instance().clear()       # another package (e.g. sam2) may have initialised hydra
            with initialize_config_dir(version_base='1.3.2', config_dir=config_dir, job_name='eval_config'):
                cfg = compose(config_name='eval_config')
            with open_dict(cfg):
                cfg['weights'] = cutie_weights
            get_dataset_cfg(cfg)                 # adds entries to cfg (needed by Cutie)
            self.cutie = CUTIE(cfg).to(device).eval()
            self.cutie.load_weights(torch.load(cfg.weights, map_location=device))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.processor = InferenceCore(self.cutie, cfg=cfg)

    # ------------------------------------------------------------------ hooks
    def begin_frame(self, tracker, frame, dets):
        torch = self.torch
        self.frame_hw = frame.shape[:2]
        self.prediction = None
        if tracker.frame_id > 1:
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
                self._update(tracker.frame_id, frame, *self._prev)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.times.append(time.perf_counter() - t0)

    def end_frame(self, tracker, frame, events):
        online = [t for t in tracker.tracked if t.is_activated]
        self._prev = (frame, [t.last_det_tlwh for t in online], [t.track_id for t in online], list(events.confirmed),
                      [t.track_id for t in events.removed])

    def mask_cue(self, track, det, frame_hw):
        """Mask fill ratio mf of the detection box when the confident track mask covers it (mf >= 0.05, mc >= 0.9)."""
        if self.track_mask_now is None or track.track_id not in self.track_mask_now:
            return None
        mid = self.track_mask_now[track.track_id]
        if mid not in self.uniq or not self.avg_prob[mid] >= MIN_MASK_AVG_CONF:
            return None
        pred = self.prediction
        x, y, hor, ver = box_bounds(det.tlwh, frame_hw[0], frame_hw[1])
        in_box = (pred[y:ver, x:hor] == mid).sum()
        mc = in_box / (pred == mid).sum()
        mf = in_box / ((ver - y) * (hor - x))
        if mf >= MIN_MM2 and not mc < MIN_MM1:
            return mf
        return None

    # ------------------------------------------------------------------ mask manager
    def _update(self, frame_id, frame, prev_frame, online_tlwhs, online_ids, new_tracks, removed_ids):
        cur = self._to_torch(frame, device=self.device)
        prev = self._to_torch(prev_frame, device=self.device)
        prob = None
        if frame_id == START_FRAME + 1 + self.init_delay:
            prob = self._initialize(cur, prev, prev_frame, online_tlwhs, online_ids)
        elif frame_id > START_FRAME + 1 + self.init_delay:
            self._add_new(prev_frame, online_tlwhs, online_ids, new_tracks)
            self._remove(removed_ids)
            prob = self.processor.step(cur)
        self.avg_prob, self.uniq = None, []
        if prob is not None:
            self.avg_prob = self._mask_avg_prob(prob)
            self.prediction = self._to_mask(prob)             # argmax index mask (uint8)
            self.prev_mask = self.prediction.copy()
            self.uniq = list(np.unique(self.prediction))[1:]  # mask numbers present (first entry dropped, as McByte)
        self.track_mask_now = None if self.track_mask is None else dict(self.track_mask)

    def _sam_masks(self, image_bgr, boxes):
        """SAM masks (one per box) on the previous frame; McByte passes the BGR image as is."""
        torch = self.torch
        self.sam_predictor.set_image(image_bgr)
        t = torch.tensor(boxes, device=self.sam.device)
        t = self.sam_predictor.transform.apply_boxes_torch(t, image_bgr.shape[:2])
        masks, _, _ = self.sam_predictor.predict_torch(point_coords=None, point_labels=None, boxes=t,
                                                       multimask_output=False)
        return masks

    def _initialize(self, cur, prev, prev_frame, online_tlwhs, online_ids):
        """SAM masks of the first frame's confirmed tracks, Cutie memory, propagation to this frame."""
        boxes, ids = [], []
        for i, ot in enumerate(online_tlwhs):
            if overlap_with_boxes_below(ot, online_tlwhs) >= CREATE_OVERLAP:
                self.awaiting.append(online_ids[i])
                continue
            boxes.append([ot[0], ot[1], ot[0] + ot[2], ot[1] + ot[3]])
            ids.append(online_ids[i])
        if not boxes:
            self.init_delay += 1              # McByte's check for this case never fires (it would fail instead)
            return None
        masks = self._sam_masks(prev_frame, boxes)
        mask = np.zeros(masks[0].shape)
        for mi in range(len(masks)):         # earlier masks keep their pixels
            m = masks[mi].cpu().numpy().astype(int)
            m[m > 0] = mi + 1
            mask += m * (mask == 0).astype(int)
        mask = mask.squeeze(0)
        self.num_objects = len(masks)
        self.objects = list(range(1, self.num_objects + 1))
        self.last_object = max(self.objects, default=0)
        self.track_mask = dict(zip(ids, range(1, self.num_objects + 1)))
        one_hot = self._one_hot(mask, self.num_objects + 1).to(self.device)
        self.processor.step(prev, one_hot[1:], idx_mask=False)
        return self.processor.step(cur)

    def _add_new(self, prev_frame, online_tlwhs, online_ids, new_tracks):
        """SAM masks for postponed tracks and tracks confirmed in the previous frame, added to Cutie's memory."""
        if not new_tracks and not self.awaiting:
            return
        boxes, ids = [], []
        for tid in self.awaiting:
            if tid not in online_ids:
                continue
            tlwh = online_tlwhs[online_ids.index(tid)]
            if overlap_with_boxes_below(tlwh, online_tlwhs) < CREATE_OVERLAP:
                boxes.append([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]])
                ids.append(tid)
        for tid in ids:
            self.awaiting.remove(tid)
        for nt in new_tracks:
            tlwh = nt.last_det_tlwh
            if overlap_with_boxes_below(tlwh, online_tlwhs) >= CREATE_OVERLAP:
                self.awaiting.append(nt.track_id)
                continue
            boxes.append([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]])
            ids.append(nt.track_id)
        if not boxes:
            return
        masks = self._sam_masks(prev_frame, boxes)
        extra = np.zeros(masks[0].shape)
        top = max(self.track_mask.values(), default=0)
        numbers, objects = [], []
        for mi in range(len(masks)):
            m = masks[mi].cpu().numpy().astype(int)
            m[m > 0] = top + mi + 1
            numbers.append(top + mi + 1)
            self.last_object += 1
            objects.append(self.last_object)
            extra += m * (extra == 0).astype(int)
        extra = extra.squeeze(0)
        prev = self._to_torch(prev_frame, device=self.device)
        self.prev_mask[extra > 0] = extra[extra > 0]          # new masks take priority
        self.num_objects += len(ids)
        one_hot = self._one_hot(self.prev_mask, self.num_objects + 1).to(self.device)
        self.objects.extend(objects)
        self.processor.step(prev, one_hot, objects=self.objects, idx_mask=False)
        for k, v in zip(ids, numbers):
            self.track_mask[k] = v

    def _remove(self, removed_ids):
        """Purge the masks of removed tracks from Cutie; mask numbers above them shift down."""
        if not removed_ids:
            return
        gone = [self.track_mask[i] for i in self.track_mask.keys() if i in removed_ids]
        purge, _, keep = self.processor.object_manager.purge_selected_objects(gone)
        if purge:
            self.processor.memory.purge_except(keep)
        self.objects = keep
        self.num_objects = len(self.objects)
        drop, shift = [], {}
        for k in self.track_mask.keys():
            if self.track_mask[k] in gone:
                drop.append(k)
            else:
                for r in gone:
                    if self.track_mask[k] > r:
                        shift[k] = shift.get(k, 0) + 1
        for k in drop:
            del self.track_mask[k]
        for k, s in shift.items():
            self.track_mask[k] -= s

    def _mask_avg_prob(self, prob):
        """Mean probability of each track mask over the pixels where it wins the argmax (NaN entries dropped)."""
        out = {}
        winner = self.torch.max(prob, dim=0).indices
        for v in self.track_mask.values():
            s = (prob[v][winner == v]).mean().item()
            if s is not None and not np.isnan(s):
                out[v] = s
        return out
