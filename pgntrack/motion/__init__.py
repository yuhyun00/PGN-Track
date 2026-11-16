"""Motion stages: none (ByteTrack), ORB camera motion compensation, PGN-Track."""
from ..config import check_keys
from .base import MotionStage, NoMotion


def build_motion(cfg: dict) -> MotionStage:
    """Motion stage from the `motion` config section: {type: none | cmc | pgn, cmc: {...}, pgn: {...}}."""
    cfg = dict(cfg or {})
    check_keys(cfg, ('type', 'cmc', 'pgn'), 'motion')
    kind = cfg.get('type', 'none')
    if kind == 'none':
        return NoMotion()
    if kind == 'cmc':
        from .cmc import CMCStage
        sub = cfg.get('cmc') or {}
        check_keys(sub, ('variant', 'strict'), 'motion.cmc')
        return CMCStage(**sub)
    if kind == 'pgn':
        from .pgn import PGNParams, PGNStage
        return PGNStage(PGNParams.from_dict(cfg.get('pgn') or {}))
    raise ValueError(f'motion.type must be none | cmc | pgn, got {kind!r}')
