"""Layered photometric energy and one Gauss-Newton step (camera p and per-track displacements d_k).

E(p, d) = sum_bg rho(r) / sigma^2 + sum_k nu_k sum_u v_ku rho(r_ku) / sigma^2
          + sum_k d_k^T Lam_k d_k + (p - p_prev)^T Lam_p (p - p_prev)
The Gauss-Newton Hessian is an arrowhead (tracks couple only through p); the track blocks are eliminated
exactly with a Schur complement.  All operations are batched tensor ops without host synchronisation, so the
solve can be captured as one CUDA graph.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .warp import jac_affine, warp_points


class Level:
    """Geometry of one pyramid level (downsampling factor f w.r.t. full resolution)."""

    def __init__(self, f, W0, H0):
        self.f = f
        self.W = W0 // f
        self.H = H0 // f
        self.off = (f - 1) / 2.0

    def to_grid(self, q_full):
        """Full-resolution points -> grid_sample coordinates and a mask of points inside the level (1 px margin)."""
        x = (q_full[..., 0] - self.off) / self.f
        y = (q_full[..., 1] - self.off) / self.f
        gx = 2 * x / (self.W - 1) - 1
        gy = 2 * y / (self.H - 1) - 1
        inside = (x >= 1) & (x <= self.W - 2) & (y >= 1) & (y <= self.H - 2)
        return torch.stack([gx, gy], -1), inside


def sample(stack, grid):
    """Bilinear samples of a (1, C, H, W) stack at grid points (P, 2) -> (C, P)."""
    return F.grid_sample(stack, grid[None, None], mode='bilinear', padding_mode='border', align_corners=True)[0, :, 0]


def with_gradients(img, sobel):
    """(1, 1, H, W) image -> (1, 3, H, W) stack [I, dI/dx, dI/dy] (Sobel / 8, level-pixel units)."""
    g = F.conv2d(F.pad(img, (1, 1, 1, 1), mode='replicate'), sobel)
    return torch.cat([img, g], 1)


def huber_weight(r, sigma, k):
    """IRLS weight of the Huber loss at residuals r with scale sigma."""
    a = r.abs() / (sigma + 1e-6)
    return torch.clamp(k / (a + 1e-9), max=1.0)


def inv2x2(M):
    """Batched 2x2 inverse with a floored determinant."""
    a, b, c, d = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
    det = a * d - b * c
    det = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
    out = torch.stack([torch.stack([d, -b], -1), torch.stack([-c, a], -1)], -2)
    return out / det[..., None, None]


@dataclass
class TrackTerm:
    """Inputs of the per-track term at one level (K padded participants, M template samples each)."""
    pts: torch.Tensor                     # (K, M, 2) samples on the predicted boxes, frame t-1, full resolution
    tmpl: torch.Tensor                    # (K, M) template intensities
    valid: torch.Tensor                   # (K, M) sample weight: template inside the image and visible
    delta0: torch.Tensor                  # (K, 2) current displacements
    Lam: torch.Tensor                     # (K, 2, 2) prior information of the displacements
    nu: torch.Tensor                      # (K,) information scale (0 for padding rows)
    grad: Optional[torch.Tensor] = None   # (K, M, 2) template gradients (ESM)


def gn_step(lvl, cur, ref_bg, w_bg0, bg_pts, p0, p_prev, B, Lam_p, c, s, huber_k, ref_grad=None, trk=None):
    """One Gauss-Newton step at one level, linearised at camera p0 and displacements trk.delta0.

    cur (1, 3, H, W) current [I, gx, gy]; ref_bg (N,) previous-frame intensities at bg_pts; w_bg0 (N,) 0/1
    background mask; ref_grad (2, N) previous-frame gradients (ESM); trk: TrackTerm or None (camera only).
    """
    v6 = B @ p0
    q_bg, xt_bg = warp_points(v6, bg_pts, c, s)
    N = bg_pts.shape[0]
    if trk is not None:
        K, M, _ = trk.pts.shape
        q_tr, xt_tr = warp_points(v6, trk.pts.reshape(-1, 2), c, s)
        q_tr = q_tr + trk.delta0.repeat_interleave(M, 0)
        grid, inside = lvl.to_grid(torch.cat([q_bg, q_tr], 0))
    else:
        grid, inside = lvl.to_grid(q_bg)
    smp = sample(cur, grid)                                     # (3, N [+ K M])
    g_all = smp[1:].t() / lvl.f                                 # gradient per full-resolution pixel
    if ref_grad is not None:                                    # ESM: mean of current and reference gradients
        g_ref = ref_grad.t() / lvl.f
        if trk is not None:
            g_ref = torch.cat([g_ref, trk.grad.reshape(-1, 2) / lvl.f], 0)
        g_all = 0.5 * (g_all + g_ref)

    # background: camera only
    r_bg = smp[0, :N] - ref_bg
    valid_bg = w_bg0 * inside[:N].to(r_bg.dtype)
    r_abs = torch.where(valid_bg > 0, r_bg.abs(), torch.full_like(r_bg, float('nan')))
    sigma = 1.4826 * torch.nanmedian(r_abs)
    sigma = torch.nan_to_num(sigma, nan=1.0).clamp(min=1e-3)
    inv_s2 = 1.0 / (sigma * sigma)
    w_bg = huber_weight(r_bg, sigma, huber_k) * valid_bg
    J_bg = jac_affine(g_all[:N], xt_bg, s) @ B                 # (N, d)
    Jw = J_bg * w_bg[:, None]
    H_bg = inv_s2 * (Jw.t() @ J_bg)
    b_bg = -inv_s2 * (Jw.t() @ r_bg)
    H_AA = H_bg + Lam_p
    b_A = b_bg - Lam_p @ (p0 - p_prev)
    out = {'sigma': sigma, 'n_bg': valid_bg.sum()}
    if trk is None:
        dp = torch.linalg.solve_ex(H_AA, b_A[:, None])[0][:, 0]
        out.update(p=p0 + dp, S=H_AA)
        return out

    # tracks: template residuals at the displaced, camera-warped samples
    d = B.shape[1]
    r_tr = smp[0, N:].view(K, M) - trk.tmpl
    valid_tr = trk.valid * inside[N:].view(K, M).to(r_tr.dtype)
    w_tr = huber_weight(r_tr, sigma, huber_k) * valid_tr * trk.nu[:, None]
    g_tr = g_all[N:].view(K, M, 2)
    J_tr = (jac_affine(g_tr.reshape(-1, 2), xt_tr, s) @ B).view(K, M, d)
    H_d = inv_s2 * torch.einsum('kmi,km,kmj->kij', g_tr, w_tr, g_tr)       # data information of d_k
    H_kA = inv_s2 * torch.einsum('kmi,km,kmj->kij', g_tr, w_tr, J_tr)      # (K, 2, d)
    b_d = -inv_s2 * torch.einsum('kmi,km,km->ki', g_tr, w_tr, r_tr)        # (K, 2)
    H_kk = H_d + trk.Lam
    b_k = b_d - torch.einsum('kij,kj->ki', trk.Lam, trk.delta0)
    Hkk_inv = inv2x2(H_kk)

    # Schur complement of the track blocks
    H_AA = H_AA + inv_s2 * torch.einsum('kmi,km,kmj->ij', J_tr, w_tr, J_tr)
    b_A = b_A - inv_s2 * torch.einsum('kmi,km,km->i', J_tr, w_tr, r_tr)
    S = H_AA - torch.einsum('kia,kij,kjb->ab', H_kA, Hkk_inv, H_kA)
    bt = b_A - torch.einsum('kia,kij,kj->a', H_kA, Hkk_inv, b_k)
    dp = torch.linalg.solve_ex(S, bt[:, None])[0][:, 0]
    rhs = b_k - torch.einsum('kia,a->ki', H_kA, dp)
    delta = trk.delta0 + torch.einsum('kij,kj->ki', Hkk_inv, rhs)
    # data-only displacement (the pseudo-measurement: the Kalman prior is not counted twice)
    H_d_reg = H_d + 1e-9 * torch.eye(2, device=H_d.device, dtype=H_d.dtype)
    delta_data = trk.delta0 + torch.einsum('kij,kj->ki', inv2x2(H_d_reg), b_d - torch.einsum('kia,a->ki', H_kA, dp))
    out.update(p=p0 + dp, S=S, delta=delta, delta_data=delta_data, H_d=H_d, frac_valid=valid_tr.mean(1))
    return out


def background_sigma(lvl, cur, ref_bg, w_bg0, bg_pts, ps, B, c, s):
    """Robust residual scale of the background at several camera motions ps (P, d) -> (P,)."""
    P = ps.shape[0]
    qs = [warp_points(B @ ps[i], bg_pts, c, s)[0] for i in range(P)]
    grid, ins = lvl.to_grid(torch.cat(qs, 0))
    v = sample(cur[:, :1], grid)[0].view(P, -1)
    valid = (w_bg0[None] > 0) & ins.view(P, -1)
    r = torch.where(valid, (v - ref_bg[None]).abs(), torch.full_like(v, float('nan')))
    return 1.4826 * torch.nanmedian(r, dim=1).values
