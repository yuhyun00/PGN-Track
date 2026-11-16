"""Re-identification: OSNet embedder and McByte++'s online reconnection plugin."""
import inspect

from ..config import check_keys
from .embedder import OSNetEmbedder
from .reconnection import Reconnection, ReIDState


def build_reid(cfg: dict):
    """Plugin from the `reid` config section: {type: none | reconnection, weights, device, <Reconnection args>}."""
    cfg = dict(cfg or {})
    kind = cfg.pop('type', 'none')
    if kind == 'none':
        return None
    if kind != 'reconnection':
        raise ValueError(f'reid.type must be none | reconnection, got {kind!r}')
    args = [p for p in inspect.signature(Reconnection).parameters if p not in ('embedder', 'release_removed')]
    check_keys(cfg, ['weights', 'device'] + args, 'reid')
    embedder = OSNetEmbedder.from_config({'weights': cfg.pop('weights'), 'device': cfg.pop('device', 'cuda')})
    return Reconnection(embedder, **cfg)


__all__ = ['OSNetEmbedder', 'Reconnection', 'ReIDState', 'build_reid']
