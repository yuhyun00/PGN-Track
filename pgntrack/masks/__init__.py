"""Mask cues: McByte's conditioned assignment, McByte (SAM + Cutie) and McByte++ (EdgeTAM) mask propagation."""
import os

from ..config import check_keys
from .association import ConditionedAssignment


def build_masks(cfg: dict, dataset, seq: str):
    """Plugin from the `masks` config section (None for plain ByteTrack association).

    type none       no masks; `conditioned: true` keeps McByte's conditioned assignment (McByte++ w/o masks)
    type mcbytepp   EdgeTAM masks (checkpoint under $WEIGHTS_ROOT; offload: frames and memory in host memory)
    type mcbyte     SAM + Cutie masks (sam_checkpoint, cutie_weights under $WEIGHTS_ROOT)
    The mask types always use the conditioned assignment.
    """
    cfg = dict(cfg or {})
    check_keys(cfg, ('type', 'conditioned', 'checkpoint', 'offload', 'sam_checkpoint', 'cutie_weights'), 'masks')
    kind = cfg.get('type', 'none')
    if kind == 'none':
        return ConditionedAssignment() if cfg.get('conditioned', False) else None
    from ..data import resolve
    if kind == 'mcbytepp':
        from .edgetam import EdgeTAMMasks
        return EdgeTAMMasks(os.path.join(dataset.root, seq, 'img1'),
                            resolve(cfg.get('checkpoint', 'edgetam.pt'), 'WEIGHTS_ROOT'),
                            offload=bool(cfg.get('offload', False)))
    if kind == 'mcbyte':
        from .sam_cutie import McByteMasks
        return McByteMasks(resolve(cfg.get('sam_checkpoint', 'sam_vit_b_01ec64.pth'), 'WEIGHTS_ROOT'),
                           resolve(cfg.get('cutie_weights', 'cutie-base-mega.pth'), 'WEIGHTS_ROOT'))
    raise ValueError(f'masks.type must be none | mcbyte | mcbytepp, got {kind!r}')


__all__ = ['ConditionedAssignment', 'build_masks']
