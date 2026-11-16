"""Static-shape PGN solve: one host->device packet, one CUDA-graph replay, one device->host copy per frame.

Participants are padded to K_max with an activity flag (padding rows get zero data weight and an identity prior,
so they do not change the Schur complement), exclusion boxes to B_max, template refreshes to T_max.  On the CPU
the same body runs eagerly.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .energy import Level, TrackTerm, background_sigma, gn_step, sample, with_gradients
from .warp import BASIS, DOF

BGR_TO_GRAY = (0.114, 0.587, 0.299)


class StaticSolver:
    """Pyramids, template pool and the layered Gauss-Newton solve on fixed-size device buffers."""

    def __init__(self, W0, H0, prm, device):
        self.prm = prm
        self.dev = torch.device(device)
        dev = self.dev
        self.d = DOF
        self.B = torch.tensor(BASIS, dtype=torch.float32, device=dev)
        Ph, Pw = prm.grid                                     # template samples on the inner part of the box
        ys = torch.linspace(-prm.inner / 2, prm.inner / 2, Ph)
        xs = torch.linspace(-prm.inner / 2, prm.inner / 2, Pw)
        uy, ux = torch.meshgrid(ys, xs, indexing='ij')
        self.u = torch.stack([ux.reshape(-1), uy.reshape(-1)], -1).to(dev)
        self.M = self.u.shape[0]
        self.K, self.Bm, self.T, self.k_pool = prm.k_max, prm.b_max, prm.t_max, prm.k_pool
        self.levels = [Level(f, W0, H0) for f in prm.levels]
        self.f_fine = self.levels[-1].f
        self.W4, self.H4 = W0 // 4, H0 // 4
        self.c = torch.tensor([(W0 - 1) / 2.0, (H0 - 1) / 2.0], device=dev)
        self.s = W0 / 2.0
        # background samples: stride bg_stride on the 1/4 level, margin bg_margin
        st, mg = prm.bg_stride, prm.bg_margin
        ii = torch.arange(mg, self.W4 - mg, st, dtype=torch.float32)
        jj = torch.arange(mg, self.H4 - mg, st, dtype=torch.float32)
        gy, gx = torch.meshgrid(jj, ii, indexing='ij')
        self.bg_pts = torch.stack([gx.reshape(-1) * 4 + 1.5, gy.reshape(-1) * 4 + 1.5], -1).to(dev)
        self.Lam_p = torch.eye(self.d, device=dev) * (prm.lam_p * self.s ** 2)
        self.eye2 = torch.eye(2, device=dev)
        self.bgr_w = torch.tensor(BGR_TO_GRAY, device=dev)
        k1 = torch.tensor([1., 2., 1.], device=dev) / 4
        self.blur_k = (k1[:, None] * k1[None, :])[None, None]
        kx = (torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0).to(dev)
        self.sobel = torch.stack([kx, kx.t()], 0)[:, None].contiguous()

        # packet layout: exclusion boxes | participants | previous camera | template refreshes
        self.o_trk = 4 * self.Bm
        self.o_pp = self.o_trk + 10 * self.K
        self.o_tu = self.o_pp + self.d
        self.P_len = self.o_tu + 5 * self.T
        self.O_len = self.d + self.d * self.d + 3 + 9 * self.K
        self.packet = torch.zeros(self.P_len, device=dev)
        self.in_boxes = self.packet[:self.o_trk].view(self.Bm, 4)               # x1 y1 x2 y2
        self.in_tracks = self.packet[self.o_trk:self.o_pp].view(self.K, 10)     # cx cy w h | P (2x2) | slot | active
        self.in_p_prev = self.packet[self.o_pp:self.o_tu]
        self.in_refresh = self.packet[self.o_tu:].view(self.T, 5)               # tlwh | slot
        self.out = torch.zeros(self.O_len, device=dev)
        self.img = torch.zeros(H0, W0, 3, dtype=torch.uint8, device=dev)
        self.img_h = torch.zeros(H0, W0, 3, dtype=torch.uint8)
        self.packet_h = torch.zeros(self.P_len)
        self.out_h = torch.zeros(self.O_len)
        if self.dev.type == 'cuda':
            self.img_h, self.packet_h, self.out_h = (x.pin_memory() for x in (self.img_h, self.packet_h, self.out_h))
        # previous-frame pyramid and template pool (slot k_pool is a scratch row for padding)
        self.prev = {lv.f: torch.zeros(1, 3, lv.H, lv.W, device=dev) for lv in self.levels}
        self.pool = {lv.f: torch.zeros(self.k_pool + 1, self.M, device=dev) for lv in self.levels}
        self.pool_valid = {lv.f: torch.zeros(self.k_pool + 1, self.M, device=dev) for lv in self.levels}
        self.pool_g = {lv.f: torch.zeros(self.k_pool + 1, self.M, 2, device=dev) for lv in self.levels}
        self.use_graph = prm.graph and self.dev.type == 'cuda'
        self.graph = None

    def host_packet(self):
        """A zeroed float32 host packet and named views onto it."""
        pk = np.zeros(self.P_len, dtype=np.float32)
        views = dict(boxes=pk[:self.o_trk].reshape(self.Bm, 4), tracks=pk[self.o_trk:self.o_pp].reshape(self.K, 10),
                     p_prev=pk[self.o_pp:self.o_tu], refresh=pk[self.o_tu:].reshape(self.T, 5))
        return pk, views

    def split_output(self, res):
        """Host copy of the output -> dict of named arrays (track rows padded to K_max)."""
        d, K = self.d, self.K
        o = d + d * d + 3
        return dict(p=res[:d], S=res[d:d + d * d].reshape(d, d), sig0=res[o - 3], sig1=res[o - 2], n_bg=res[o - 1],
                    delta=res[o:o + 2 * K].reshape(K, 2), delta_data=res[o + 2 * K:o + 4 * K].reshape(K, 2),
                    H_d=res[o + 4 * K:o + 8 * K].reshape(K, 2, 2), frac_valid=res[o + 8 * K:o + 9 * K])

    # ------------------------------------------------------------------ device body
    def _refresh_templates(self):
        """Write the queued templates, sampled from the previous frame (where their tracks were matched)."""
        tu = self.in_refresh
        slots = tu[:, 4].long()
        ctr = tu[:, :2] + 0.5 * tu[:, 2:4]
        pts = (ctr[:, None, :] + self.u[None] * tu[:, None, 2:4]).reshape(-1, 2)
        for lv in self.levels:
            g, inside = lv.to_grid(pts)
            v3 = sample(self.prev[lv.f], g)
            self.pool[lv.f].index_copy_(0, slots, v3[0].view(self.T, self.M))
            self.pool_g[lv.f].index_copy_(0, slots, v3[1:].t().reshape(self.T, self.M, 2))
            self.pool_valid[lv.f].index_copy_(0, slots, inside.view(self.T, self.M).float())

    def _pyramid(self):
        """Grey 1/4-resolution image, normalised, then per level: average pooling, smoothing, gradients."""
        gray = (self.img[:self.H4 * 4, :self.W4 * 4].float() * self.bgr_w).sum(-1)
        t = F.avg_pool2d(gray[None, None], 4)
        t = (t - t.mean()) / (t.std() + 1e-6)
        cur = {}
        for lv in self.levels:
            img = t if lv.f == 4 else F.avg_pool2d(t, lv.f // 4)[..., :lv.H, :lv.W]
            for _ in range(self.prm.blur):
                img = F.conv2d(F.pad(img, (1, 1, 1, 1), mode='replicate'), self.blur_k)
            cur[lv.f] = with_gradients(img, self.sobel)
        return cur

    def _visibility(self, ctr, wh, act, pts):
        """Depth-ordered layers: a sample of track k inside the predicted box of a participant j whose box bottom
        is lower in the image (j in front of k) is hidden. -> (K, M) weights."""
        lo, hi = ctr - 0.5 * wh, ctr + 0.5 * wh
        front = (hi[None, :, 1] > hi[:, None, 1]) & (act[None, :] > 0)                # front[k, j]
        px, py = pts[..., 0:1], pts[..., 1:2]
        inb = ((px >= lo[None, None, :, 0]) & (px <= hi[None, None, :, 0]) &
               (py >= lo[None, None, :, 1]) & (py <= hi[None, None, :, 1]))           # (K, M, K)
        return (~(inb & front[:, None, :]).any(-1)).to(pts.dtype)

    def _body(self):
        prm = self.prm
        self._refresh_templates()
        cur = self._pyramid()
        # background: previous-frame samples, textured and outside every (dilated) box
        refs = {}
        for lv in self.levels:
            g, _ = lv.to_grid(self.bg_pts)
            refs[lv.f] = sample(self.prev[lv.f], g)
        w_bg0 = (refs[self.f_fine][1:].norm(dim=0) > prm.tex_thresh).float()
        bx, P = self.in_boxes, self.bg_pts
        inb = ((P[:, None, 0] >= bx[None, :, 0]) & (P[:, None, 0] <= bx[None, :, 2]) &
               (P[:, None, 1] >= bx[None, :, 1]) & (P[:, None, 1] <= bx[None, :, 3])).any(1)
        w_bg0 = w_bg0 * (~inb).float()
        # participants: predicted boxes, prior information, template slots
        tk = self.in_tracks
        act = tk[:, 9]
        ctr, wh = tk[:, :2], tk[:, 2:4]
        Pk = torch.where(act[:, None, None] > 0, tk[:, 4:8].view(self.K, 2, 2), self.eye2)
        a, b, c_, d_ = Pk[:, 0, 0], Pk[:, 0, 1], Pk[:, 1, 0], Pk[:, 1, 1]
        det = a * d_ - b * c_
        Lam_k = torch.stack([torch.stack([d_, -b], -1), torch.stack([-c_, a], -1)], -2) / det[:, None, None]
        slots = tk[:, 8].long()
        trk_pts = ctr[:, None, :] + self.u[None] * wh[:, None, :]
        vis = self._visibility(ctr, wh, act, trk_pts) if prm.layers else None
        area = (prm.inner * wh[:, 0] / self.f_fine) * (prm.inner * wh[:, 1] / self.f_fine)
        nu = torch.clamp(area / self.M, max=1.0) * act        # effective number of independent samples / M
        # coarse-to-fine: one Gauss-Newton step per level
        p_prev = self.in_p_prev
        p0 = p_prev.clone() if prm.warm_start else torch.zeros_like(p_prev)
        p = p0
        delta = torch.zeros(self.K, 2, device=self.dev)
        out = None
        for lv in self.levels:
            trk = None
            if not prm.camera_only:
                valid = self.pool_valid[lv.f][slots]
                if vis is not None:
                    valid = valid * vis
                trk = TrackTerm(pts=trk_pts, tmpl=self.pool[lv.f][slots], valid=valid, delta0=delta, Lam=Lam_k, nu=nu,
                                grad=self.pool_g[lv.f][slots] if prm.esm else None)
            out = gn_step(lv, cur[lv.f], refs[lv.f][0], w_bg0, self.bg_pts, p, p_prev, self.B, self.Lam_p, self.c,
                          self.s, prm.huber_k, ref_grad=refs[lv.f][1:] if prm.esm else None, trk=trk)
            p = out['p']
            if trk is not None:
                delta = out['delta']
        fine = self.levels[-1]
        sig = background_sigma(fine, cur[fine.f], refs[fine.f][0], w_bg0, self.bg_pts, torch.stack([p0, p]), self.B,
                               self.c, self.s)
        if prm.camera_only:
            z = torch.zeros(self.K * 9, device=self.dev)
        else:
            z = torch.cat([out['delta'].reshape(-1), out['delta_data'].reshape(-1), out['H_d'].reshape(-1),
                           out['frac_valid']])
        self.out.copy_(torch.cat([p, out['S'].reshape(-1), sig, out['n_bg'].reshape(1), z]))
        for lv in self.levels:
            self.prev[lv.f].copy_(cur[lv.f])

    # ------------------------------------------------------------------ execution
    def _capture(self):
        """Warm up on a side stream, capture the body as a CUDA graph, and restore the state the warm-up touched."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        saved = {f: (self.prev[f].clone(), self.pool[f].clone(), self.pool_valid[f].clone(), self.pool_g[f].clone())
                 for f in self.prev}
        with torch.cuda.stream(s):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()
        for f, (a, b, c, e) in saved.items():
            self.prev[f].copy_(a)
            self.pool[f].copy_(b)
            self.pool_valid[f].copy_(c)
            self.pool_g[f].copy_(e)
        torch.cuda.synchronize()

    def run(self, img_np, packet_np):
        """Upload the frame and the packet, solve, return the output as float64 (the only sync of the frame)."""
        self.img_h.numpy()[...] = img_np
        self.packet_h.numpy()[...] = packet_np
        self.img.copy_(self.img_h, non_blocking=True)
        self.packet.copy_(self.packet_h, non_blocking=True)
        if self.use_graph:
            if self.graph is None:
                self._capture()
            self.graph.replay()
        else:
            self._body()
        self.out_h.copy_(self.out, non_blocking=True)
        if self.dev.type == 'cuda':
            torch.cuda.current_stream().synchronize()
        return self.out_h.numpy().astype(np.float64)
