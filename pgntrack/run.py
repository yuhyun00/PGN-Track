"""Run a tracker configuration on one sequence and write MOT-format rows."""
import os

import cv2
import numpy as np

from .config import check_keys
from .motion import build_motion
from .tracker import Tracker


def single_thread():
    """One CPU thread for OpenCV and PyTorch (the paper's setting; also makes runs reproducible)."""
    cv2.setNumThreads(1)
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def build_plugins(cfg: dict, dataset, seq: str) -> list:
    """Plugins of one sequence from the `masks`, `reid` and `identity` config sections, in this order."""
    from .masks import build_masks
    from .reid import build_reid
    identity = (cfg.get('identity') or {}).get('type', 'none') != 'none'
    reid = build_reid(cfg.get('reid'))
    if reid is not None and identity:
        reid.release_removed = True             # the identity plugin needs settled rows promptly
    plugins = [p for p in (build_masks(cfg.get('masks'), dataset, seq), reid) if p is not None]
    if identity:
        from .identity import build_identity_plugin
        from .reid import OSNetEmbedder
        plugin = build_identity_plugin(cfg, dataset, seq, OSNetEmbedder.from_config(cfg['reid']).embed)
        if plugin is not None:
            plugins.append(plugin)
    return plugins


def build_tracker(cfg: dict, dataset, seq: str, plugins=()) -> Tracker:
    """Tracker for one sequence from the `tracker`, `motion` and `output` config sections."""
    t = dict(cfg.get('tracker') or {})
    check_keys(t, ('track_thresh', 'track_buffer', 'frame_rate', 'match_thresh', 'det_format'), 'tracker')
    out = dict(cfg.get('output') or {})
    check_keys(out, ('min_box_area', 'aspect_ratio_thresh'), 'output')
    frame_rate = t.get('frame_rate', 30)
    if frame_rate == 'seqinfo':                          # per-sequence frame rate (lost-track buffer in frames)
        frame_rate = dataset.frame_rate(seq)
    return Tracker(build_motion(cfg.get('motion')), track_thresh=t.get('track_thresh', 0.6),
                   track_buffer=t.get('track_buffer', 30), frame_rate=frame_rate,
                   match_thresh=t.get('match_thresh', (0.9, 0.5, 0.7)),
                   min_box_area=out.get('min_box_area'), aspect_ratio_thresh=out.get('aspect_ratio_thresh'),
                   plugins=plugins)


def run_sequence(cfg: dict, dataset, seq: str, max_frames=None, plugins=None):
    """Track one sequence; returns (rows, tracker).  Rows: (frame, id, x, y, w, h, score), stably sorted by frame
    (plugins may emit held-back rows late).  plugins=None builds them from the config (build_plugins)."""
    frames = dataset.frames(seq)[:max_frames]
    dets = dataset.detections(seq, (cfg.get('tracker') or {}).get('det_format', 'float64'))
    if plugins is None:
        plugins = build_plugins(cfg, dataset, seq)
    tracker = build_tracker(cfg, dataset, seq, plugins)
    empty = np.zeros((0, 5))
    rows = []
    for fid, path in enumerate(frames, 1):
        img = cv2.imread(path)
        online = tracker.update(dets.get(fid, empty), img)
        rows += tracker.output(online)
    rows += tracker.finish()
    rows.sort(key=lambda r: r[0])
    return rows, tracker


def write_mot(path: str, rows):
    """MOTChallenge text file: frame, id, x, y, w, h, score, -1, -1, -1."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.writelines(f'{fr},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{s:.2f},-1,-1,-1\n' for fr, tid, x, y, w, h, s in rows)
