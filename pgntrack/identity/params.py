"""Constants and switches of the identity layer (supp. table of identity constants)."""
from dataclasses import dataclass

from ..config import check_keys

INTERACT_IOU = 0.2      # two output boxes interact while their IoU is above 0.2 ...
MAX_GAP = 3             # ... and overlaps at most 3 frames apart form one interaction
MIN_CROP = 4            # a view is embedded only if its crop inside the image is at least 4 x 4 px
PAIR_DRAWS = 200        # pair pseudo-split draws per sequence
PAIR_SEED = 10000       # sequence k (sorted order) draws with numpy default_rng(PAIR_SEED + k)
PAIR_GAPS = (1, 30)     # split gap uniform in 1..29 frames
PAIR_MARGIN = 32        # a drawn track lives more than 2 L + 32 frames
BUSY_MARGIN = 2         # a track is busy from t_s - 2 to t_e + 2 of each of its tested interactions
EM_PI0, EM_ITERS = 0.05, 500
PI_MIN, PI_MAX = 1e-6, 0.5
TYPES = ('none', 'online', 'batch', 'cross')


@dataclass(frozen=True)
class IdentityParams:
    """Switches of the `identity` config section (defaults = the paper)."""
    type: str = 'online'        # online | batch | cross calibration (none: no identity layer)
    views: int = 5              # M: visible views per prototype
    window: int = 45            # L: frames searched before t_s and after t_e
    test_iou: float = 0.3       # interactions whose IoU reaches this value are decided
    visible_iou: float = 0.3    # a visible view has IoU below this with every other output box of its frame
    prior: object = 'em'        # 'em' (EM prior) or a fixed exchange prior pi (sensitivity rows)
    kappa_scale: float = 1.0    # scale of the label-free kappa (sensitivity rows)
    calib: str = ''             # in-loop only: evidence file of the warm-start dataset

    @classmethod
    def from_config(cls, cfg):
        """Parameters from the `identity` config section."""
        cfg = dict(cfg or {})
        check_keys(cfg, cls.__dataclass_fields__, 'identity')
        p = cls(**cfg)
        if p.type not in TYPES:
            raise ValueError(f'identity.type must be one of {TYPES}, got {p.type!r}')
        if p.prior != 'em' and not 0 < float(p.prior) < 1:
            raise ValueError(f"identity.prior must be 'em' or a probability, got {p.prior!r}")
        return p

    @property
    def buffer(self) -> int:
        """Visible views kept per track: M, and every view between the last overlap and the closing frame."""
        return max(self.views, MAX_GAP + 1)
