"""Options of the PGN-Track motion stage: the switches of the paper's ablations and the fixed hyper-parameters."""
from dataclasses import dataclass, fields

from ...config import check_keys


@dataclass
class PGNParams:
    # ---- switches (configs/ablations)
    layers: bool = True             # depth-ordered visibility of template samples; False = one layer, all visible
    prior: str = 'PF'               # prior information of d_k: 'PF' = (P_k + F_k)^-1, 'P' = P_k^-1
    alpha: float = 0.05             # offset term F_k = diag((alpha w)^2, (alpha h)^2) in R_k and the prior; 0 = off
    bridge: str = 'kalman'          # 'kalman': Kalman update with (z_k, R_k); 'box': add d_k to the predicted centre
    camera_only: bool = False       # no track term (the camera from the background alone, no measurements)
    esm: bool = True                # efficient second-order minimisation: mean of current and reference gradients
    levels: tuple = (8, 4)          # pyramid levels (downsampling factors, ending at 4), one Gauss-Newton step each
    warm_start: bool = True         # start the solve from the previous camera motion (else from zero)
    fallback: str = 'previous'      # motion applied when the estimate is rejected: 'previous' | 'zero'
    graph: bool = True              # capture the solve as a CUDA graph (GPU only)
    device: str = 'cuda'            # cuda | cpu (same outputs)
    # ---- fixed hyper-parameters (supplementary table of hyper-parameters)
    grid: tuple = (16, 8)           # template samples per box, rows x columns
    inner: float = 0.7              # the samples cover the central 70% of the box
    blur: int = 1                   # [1 2 1]^2 / 16 smoothing passes per level before differentiation
    huber_k: float = 1.345          # Huber loss, scale = 1.4826 median |r| of the background residuals
    bg_stride: int = 2              # background samples: stride and margin on the 1/4 level
    bg_margin: int = 2
    box_dilate: float = 1.2         # dilation of the boxes excluded from the background
    tex_thresh: float = 0.02        # gradient-magnitude gate of background samples
    lam_p: float = 1e-2             # camera prior Lam_p = lam_p s^2 I (px^-2)
    tau_info: float = 1e-2          # measurement needs lambda_min(H_data) > tau_info (px^-2)
    min_visible: float = 0.5        # ... and more than this fraction of samples inside the image and visible
    maha_gate: float = 9.21         # chi2_2(0.99) gate on the innovation of z_k
    kappa_max: float = 1e6          # camera rejected if cond(S) >= kappa_max,
    tau_res: float = 1.0            # or if the robust scale grows beyond tau_res times its value at the start,
    tau_cut: float = 0.6            # or if it exceeds tau_cut (shot cut: zero motion, no fallback)
    min_bg: int = 50                # or if fewer background samples are valid
    max_lost: int = 30              # lost tracks take part for at most this many frames
    k_max: int = 64                 # static buffer sizes: participants, exclusion boxes, template refreshes, slots
    b_max: int = 256
    t_max: int = 64
    k_pool: int = 512

    @classmethod
    def from_dict(cls, d: dict) -> 'PGNParams':
        """Build from a config section, rejecting unknown keys."""
        check_keys(d, [f.name for f in fields(cls)], 'motion.pgn')
        p = cls(**d)
        p.levels = tuple(int(f) for f in p.levels)
        p.grid = tuple(p.grid)
        p.alpha = float(p.alpha)
        if p.prior not in ('PF', 'P'):
            raise ValueError("motion.pgn.prior must be 'PF' or 'P'")
        if p.bridge not in ('kalman', 'box'):
            raise ValueError("motion.pgn.bridge must be 'kalman' or 'box'")
        if p.fallback not in ('previous', 'zero'):
            raise ValueError("motion.pgn.fallback must be 'previous' or 'zero'")
        if p.levels[-1] != 4 or list(p.levels) != sorted(p.levels, reverse=True):
            raise ValueError('motion.pgn.levels must be decreasing and end at 4')
        return p
