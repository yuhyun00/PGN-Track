"""Similarity camera motion in normalised image coordinates.

x~ = (x - c) / s with c the image centre and s = W / 2 (full-resolution pixels); a motion moves x~ to
(I + A) x~ + t.  The 6-D affine increment v = (a11, a12, a21, a22, tx, ty) is v = B p with the similarity
parameters p = (a, b, tx, ty): a11 = a22 = a, a21 = -a12 = b.
"""
import numpy as np
import torch

BASIS = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0],
                  [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
DOF = BASIS.shape[1]


def full_warp(p, c, s):
    """Similarity parameters p -> full-resolution 2x3 warp A with x_t = A [x_{t-1}; 1]."""
    v = BASIS @ np.asarray(p, dtype=np.float64)
    L = np.array([[1 + v[0], v[1]], [v[2], 1 + v[3]]])
    c = np.asarray(c, dtype=np.float64)
    t = (np.eye(2) - L) @ c + s * v[4:6]
    return np.concatenate([L, t[:, None]], 1)


def warp_points(v6, pts, c, s):
    """Apply the affine increment v6 (6,) to full-resolution points (..., 2); returns (warped points, x~)."""
    xt = (pts - c) / s
    x, y = xt[..., 0], xt[..., 1]
    qx = (1 + v6[0]) * x + v6[1] * y + v6[4]
    qy = v6[2] * x + (1 + v6[3]) * y + v6[5]
    return torch.stack([qx, qy], -1) * s + c, xt


def jac_affine(g, xt, s):
    """Jacobian of the warped intensity w.r.t. the affine increment: gradient g (..., 2), points x~ -> (..., 6)."""
    gx, gy = g[..., 0], g[..., 1]
    x, y = xt[..., 0], xt[..., 1]
    return s * torch.stack([gx * x, gx * y, gy * x, gy * y, gx, gy], -1)
