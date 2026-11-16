"""Datasets in MOTChallenge layout: sequences, frames, cached detections, ground truth.

A split directory holds one folder per sequence with `img1/*.jpg`, `seqinfo.ini` and (if annotated) `gt/gt.txt`.
Detections come from a cache (`$DET_ROOT/<dir>/<seq>.npy`, rows frame, x1, y1, x2, y2, score) or from the
sequence's own `det/det.txt` (MOT rows frame, id, left, top, width, height, conf; SoccerNet ships its
ground-truth boxes there).

`det_format` (tracker config) selects the precision the tracker sees:
  float64        the cached values (ByteTrack, PGN-Track presets)
  mcbytepp_text  as McByte++'s demo reads its detection text files: frame,x1,y1,x2,y2,score written with
                 %.2f / %.5f, parsed into float32 (McByte++ presets and the re-ID rows of the paper)
  mcbyte_text    as McByte's demo reads MOT text files: frame,-1,x,y,w,h,score written with %.2f / %.4f,
                 x2 = x + w in float64, parsed into float32 (McByte preset)
MOT sources (`det/det.txt`) are read directly by both demos: their values only become float32.
"""
import configparser
import os
from dataclasses import dataclass, field

import numpy as np

from .config import check_keys


def resolve(path: str, env: str) -> str:
    """Join a relative path to the root given by an environment variable (absolute paths are kept)."""
    if os.path.isabs(path):
        return path
    root = os.environ.get(env)
    if not root:
        raise EnvironmentError(f'set ${env} (needed for {path!r})')
    return os.path.join(root, path)


@dataclass
class Dataset:
    """One split: where its frames, detections and ground truth are."""
    name: str
    root: str                      # split directory (frames and ground truth)
    det_source: str                # 'npy' | 'mot'
    det_dir: str = ''              # directory of the .npy caches (det_source 'npy')
    benchmark: str = ''            # TrackEval benchmark name ('' = no ground truth)
    preproc: bool = True           # TrackEval DO_PREPROC
    seqs: list = field(default_factory=list)

    def frames(self, seq: str) -> list:
        """Sorted JPEG paths of a sequence; frame ids are 1, 2, ... in this order."""
        d = os.path.join(self.root, seq, 'img1')
        return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith('.jpg') and not f.startswith('.')]

    def frame_rate(self, seq: str) -> int:
        """frameRate of the sequence's seqinfo.ini."""
        c = configparser.ConfigParser()
        c.read(os.path.join(self.root, seq, 'seqinfo.ini'))
        return int(c['Sequence']['frameRate'])

    def detections(self, seq: str, det_format: str = 'float64') -> dict:
        """Detections per frame id: {frame: (n, 5) array of x1, y1, x2, y2, score} (float32 for the text formats)."""
        if det_format not in DET_FORMATS:
            raise ValueError(f'det_format must be one of {DET_FORMATS}, got {det_format!r}')
        if self.det_source == 'npy':
            raw = np.load(os.path.join(self.det_dir, seq + '.npy'))
            rows = raw.astype(np.float64) if det_format == 'float64' else text_rows(raw, det_format)
        else:
            a = np.loadtxt(os.path.join(self.root, seq, 'det', 'det.txt'), delimiter=',', ndmin=2)
            rows = np.zeros((len(a), 6), dtype=np.float64)
            rows[:, 0] = a[:, 0]
            rows[:, 1] = a[:, 2]
            rows[:, 2] = a[:, 3]
            rows[:, 3] = a[:, 2] + a[:, 4]
            rows[:, 4] = a[:, 3] + a[:, 5]
            rows[:, 5] = np.where(a[:, 6] > 0, a[:, 6], 1.0)     # unscored boxes get confidence 1
            if det_format != 'float64':
                rows = rows.astype(np.float32)
        by_frame = {}
        for r in rows:
            by_frame.setdefault(int(r[0]), []).append(r[1:6])
        return {f: np.asarray(v, dtype=rows.dtype).reshape(-1, 5) for f, v in by_frame.items()}


DET_FORMATS = ('float64', 'mcbytepp_text', 'mcbyte_text')


def text_rows(raw: np.ndarray, det_format: str) -> np.ndarray:
    """Cached float32 rows -> the float32 rows the McByte++ / McByte demos parse from their text exports."""
    out = np.empty((len(raw), 6), dtype=np.float32)
    for k, (fr, x1, y1, x2, y2, s) in enumerate(raw):
        if det_format == 'mcbytepp_text':          # "%d,%.2f,%.2f,%.2f,%.2f,%.5f": frame, x1, y1, x2, y2, score
            vals = (float(f'{x1:.2f}'), float(f'{y1:.2f}'), float(f'{x2:.2f}'), float(f'{y2:.2f}'), float(f'{s:.5f}'))
        else:                                      # "%d,-1,%.2f,%.2f,%.2f,%.2f,%.4f": frame, x, y, w, h, score
            x, y = float(f'{x1:.2f}'), float(f'{y1:.2f}')
            w, h = float(f'{x2 - x1:.2f}'), float(f'{y2 - y1:.2f}')
            vals = (x, y, w + x, h + y, float(f'{s:.4f}'))
        out[k] = (int(fr),) + vals
    return out


def load_dataset(cfg: dict) -> Dataset:
    """Build a Dataset from the `data` and `eval` sections of a dataset config."""
    data = cfg['data']
    check_keys(data, ('name', 'root', 'detections', 'seqs'), 'data')
    det = data.get('detections', {'source': 'npy', 'dir': data['name']})
    check_keys(det, ('source', 'dir'), 'data.detections')
    ev = cfg.get('eval') or {}
    check_keys(ev, ('benchmark', 'preproc'), 'eval')
    ds = Dataset(name=data['name'], root=resolve(data['root'], 'DATA_ROOT'), det_source=det['source'],
                 benchmark=ev.get('benchmark', ''), preproc=bool(ev.get('preproc', True)))
    if ds.det_source == 'npy':
        ds.det_dir = resolve(det['dir'], 'DET_ROOT')
    elif ds.det_source != 'mot':
        raise ValueError(f"data.detections.source must be npy | mot, got {ds.det_source!r}")
    ds.seqs = list(data['seqs']) if data.get('seqs') else sorted(
        s for s in os.listdir(ds.root) if not s.startswith('.') and os.path.isdir(os.path.join(ds.root, s)))
    return ds
