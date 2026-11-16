"""PGN-Track motion stage (host side).

Per frame, after the Kalman prediction: pack the exclusion boxes, the participating tracks (predicted box, prior
covariance, template slot), the previous camera motion and the queued template refreshes into one packet; solve
camera and track displacements jointly on the device; accept or reject the camera estimate; warp all Kalman
states with it (plus its uncertainty); and feed each track's photometric pseudo-measurement z_k with covariance
R_k = R0_k + F_k to its Kalman filter (the bridge).  Association then runs on the updated predictions.
"""
import numpy as np

from ...tracker.kalman import KalmanFilter
from ...tracker.track import TrackState, warp_tracks
from ..base import MotionStage
from .params import PGNParams
from .solver import StaticSolver
from .warp import BASIS, DOF, full_warp


def inv2(M):
    """Batched 2x2 inverse (host)."""
    a, b, c, d = M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1]
    det = a * d - b * c
    out = np.empty_like(M)
    out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1] = d, -b, -c, a
    return out / det[:, None, None]


class PGNStage(MotionStage):
    """Joint photometric Gauss-Newton camera + per-track displacement, bridged into the Kalman filter."""

    def __init__(self, params: PGNParams):
        super().__init__()
        self.prm = params
        self.solver = None                                  # built on the first frame (needs the image size)
        self.prev_dets = np.zeros((0, 4))
        self.p_prev = np.zeros(DOF)                         # camera motion applied in the previous frame
        self.free_slots = list(range(params.k_pool))[::-1]
        self.templates = {}                                 # track id -> template slot
        self.pending = []                                   # template refreshes for the next solve: (tlwh, slot)
        self.stats = dict(frames=0, camera_rejected=0, cuts=0, measurements=0)

    def _dilated(self, boxes):
        ctr = 0.5 * (boxes[:, :2] + boxes[:, 2:])
        half = 0.5 * self.prm.box_dilate * (boxes[:, 2:] - boxes[:, :2])
        return np.concatenate([ctr - half, ctr + half], 1)

    def _participants(self, tracks, unconfirmed, frame_id, W0, H0):
        """Tracked, then unconfirmed, then lost tracks (most recent first) with a template, inside the image."""
        cand = [t for t in tracks if t.state == TrackState.Tracked] + list(unconfirmed)
        cand += sorted([t for t in tracks if t.state != TrackState.Tracked], key=lambda t: -t.end_frame)
        parts = []
        for trk in cand:
            if trk.track_id not in self.templates:
                continue
            if trk.state == TrackState.Lost and frame_id - trk.end_frame > self.prm.max_lost:
                continue
            if not (0 <= trk.mean[0] < W0 and 0 <= trk.mean[1] < H0):
                continue
            parts.append(trk)
            if len(parts) == self.prm.k_max:
                break
        return parts

    def _pack(self, dets_all, all_tracks, parts):
        """Fill the host packet of the solver."""
        prm, sv = self.prm, self.solver
        pk, v = sv.host_packet()
        boxes = [dets_all[:, :4], self.prev_dets]           # current and previous detections, predicted tracks
        if all_tracks:
            boxes.append(np.asarray([t.tlbr for t in all_tracks]))
        boxes = self._dilated(np.concatenate(boxes, 0))[:sv.Bm]
        v['boxes'][:] = np.array([0, 0, -1, -1], dtype=np.float32)    # empty boxes
        v['boxes'][:len(boxes)] = boxes
        tk = v['tracks']
        tk[:, 8] = prm.k_pool
        for i, trk in enumerate(parts):
            tk[i, :4] = trk.mean[:4]
            P = trk.covariance[:2, :2].copy()
            if prm.prior == 'PF':                           # prior of the template centre: P_k + F_k
                P[0, 0] += (prm.alpha * trk.mean[2]) ** 2
                P[1, 1] += (prm.alpha * trk.mean[3]) ** 2
            tk[i, 4:8] = P.reshape(-1)
            tk[i, 8] = self.templates[trk.track_id]
            tk[i, 9] = 1
        v['p_prev'][:] = self.p_prev
        tu = v['refresh']
        tu[:, 2:4] = 1
        tu[:, 4] = prm.k_pool
        for i, (tlwh, slot) in enumerate(self.pending[:sv.T]):
            tu[i, :4] = tlwh
            tu[i, 4] = slot
        self.pending = []
        return pk

    def step(self, frame, dets_all, dets_high, tracks, unconfirmed, frame_id):
        prm = self.prm
        H0, W0 = frame.shape[:2]
        all_tracks = list(tracks) + list(unconfirmed)
        first = self.solver is None
        if first:
            self.c_np = np.array([(W0 - 1) / 2.0, (H0 - 1) / 2.0])
            self.s = W0 / 2.0
            self.solver = StaticSolver(W0, H0, prm, prm.device)
        parts = [] if (prm.camera_only or first) else self._participants(tracks, unconfirmed, frame_id, W0, H0)
        out = self.solver.split_output(self.solver.run(frame, self._pack(dets_all, all_tracks, parts)))
        self.prev_dets = dets_all[:, :4].copy()
        if first:                                           # the first frame only fills the pyramid
            self.warps.append(np.eye(2, 3))
            return

        # camera: accept, or fall back
        p_fin, S, sig0, sig1 = out['p'], out['S'], out['sig0'], out['sig1']
        with np.errstate(all='ignore'):
            ev = np.linalg.eigvalsh(0.5 * (S + S.T)) if np.isfinite(S).all() else np.array([0.0])
        cond = ev[-1] / ev[0] if ev[0] > 0 else np.inf
        cut = sig1 > prm.tau_cut
        cam_ok = bool(np.isfinite(p_fin).all() and cond < prm.kappa_max and sig1 <= prm.tau_res * sig0
                      and not cut and out['n_bg'] > prm.min_bg)
        if cam_ok:
            p_apply = p_fin
        elif cut or prm.fallback == 'zero':
            p_apply = np.zeros(DOF)
        else:
            p_apply = self.p_prev.copy()
        A = full_warp(p_apply, self.c_np, self.s)
        self.warps.append(A)
        self.p_prev = np.zeros(DOF) if cut else p_apply
        self.stats['frames'] += 1
        self.stats['camera_rejected'] += int(not cam_ok)
        self.stats['cuts'] += int(cut)

        warp_tracks(tracks, A)
        warp_tracks(unconfirmed, A)
        if cam_ok and all_tracks:                           # camera uncertainty into the Kalman prior
            Sinv = np.linalg.inv(0.5 * (S + S.T))
            xt = (np.asarray([t.mean[:2] for t in all_tracks]) - self.c_np) / self.s
            G = np.zeros((len(all_tracks), 2, 6))
            G[:, 0, 0], G[:, 0, 1], G[:, 0, 4] = xt[:, 0], xt[:, 1], 1
            G[:, 1, 2], G[:, 1, 3], G[:, 1, 5] = xt[:, 0], xt[:, 1], 1
            Jc = (G * self.s) @ BASIS
            Pc = Jc @ Sinv @ np.transpose(Jc, (0, 2, 1))
            for trk, pc in zip(all_tracks, Pc):
                trk.covariance[:2, :2] += pc
        if parts and cam_ok:
            self.stats['measurements'] += self._measure(parts, out)

    def _measure(self, parts, out):
        """Photometric measurements of the participants; returns the number of tracks updated."""
        prm = self.prm
        K = len(parts)
        dl, dd = out['delta'][:K], out['delta_data'][:K]
        Hd, fv = out['H_d'][:K], out['frac_valid'][:K]
        a, b, e = Hd[:, 0, 0], 0.5 * (Hd[:, 0, 1] + Hd[:, 1, 0]), Hd[:, 1, 1]
        lam_min = 0.5 * (a + e) - np.sqrt(np.maximum(0.25 * (a - e) ** 2 + b * b, 0))
        ok = (lam_min > prm.tau_info) & (fv > prm.min_visible) & np.isfinite(dd).all(1) & np.isfinite(dl).all(1)
        idx = np.nonzero(ok)[0]
        if not len(idx):
            return 0
        sel = [parts[j] for j in idx]
        if prm.bridge == 'box':                             # ablation: MAP displacement added to the predicted box
            for j, trk in zip(idx, sel):
                trk.mean[:2] += dl[j]
            return len(sel)
        # Kalman bridge: z_k = predicted centre + data-only displacement, R_k = R0_k + F_k, chi2 gate
        mean = np.stack([t.mean for t in sel])
        cov = np.stack([t.covariance for t in sel])
        R = inv2(Hd[idx] + 1e-9 * np.eye(2))
        if prm.alpha > 0:
            wh = mean[:, 2:4] * prm.alpha
            R[:, 0, 0] += wh[:, 0] ** 2
            R[:, 1, 1] += wh[:, 1] ** 2
        off = dd[idx]
        z = mean[:, :2] + off
        m2 = np.einsum('ki,kij,kj->k', off, inv2(cov[:, :2, :2] + R), off)
        keep = m2 < prm.maha_gate
        if keep.any():
            nm, nc = KalmanFilter.multi_update_position(mean[keep], cov[keep], z[keep], R[keep])
            for trk, m_, c_ in zip([t for t, k in zip(sel, keep) if k], nm, nc):
                trk.mean, trk.covariance = m_, c_
        return int(keep.sum())

    def post_update(self, frame, matched, removed, frame_id):
        """Free the template slots of removed tracks; queue new templates at the matched detections."""
        for trk in removed:
            slot = self.templates.pop(trk.track_id, None)
            if slot is not None:
                self.free_slots.append(slot)
        for trk in matched:
            if len(self.pending) >= self.prm.t_max:
                break
            slot = self.templates.get(trk.track_id)
            if slot is None:
                if not self.free_slots:
                    continue
                slot = self.templates[trk.track_id] = self.free_slots.pop()
            self.pending.append((np.asarray(trk.last_det_tlwh, dtype=np.float32), slot))
