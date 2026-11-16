"""Identity inference (Sec. 4) on saved tracker outputs: how the paper's accuracy numbers are produced.

The calibration comes from `identity.type` of the method config: online (the paper's default; video k uses kappa
and pi of the warm start plus videos 1..k-1 of the sorted split), batch (the split itself), cross (warm start only).
The warm start is the evidence of the other dataset: SoccerNet for the SportsMOT splits, SMV-u for SoccerNet.

  # evidence of each split (batch calibration needs no warm start)
  python tools/identity_offline.py --method configs/methods/ours_batch.yaml \
      --data configs/datasets/soccernet_test.yaml --base results/pgn_track_reid/soccernet_test \
      --out results/ours_batch/soccernet_test
  python tools/identity_offline.py --method configs/methods/ours_batch.yaml --data configs/datasets/smvu.yaml \
      --base results/pgn_track_reid/sportsmot_valtr --out results/ours_batch/sportsmot_valtr
  # online (and cross): reuse the split's evidence, warm start from the other dataset
  python tools/identity_offline.py --method configs/methods/ours.yaml --data configs/datasets/smvu.yaml \
      --base results/pgn_track_reid/sportsmot_valtr --evidence results/ours_batch/sportsmot_valtr/evidence.json \
      --warm results/ours_batch/soccernet_test/evidence.json --out results/ours/sportsmot_valtr

Writes <out>/<seq>.txt, <out>/evidence.json (reusable as --evidence, --warm and identity.calib) and
<out>/identity.json (kappa, pi and the exchanges of every video).  Sensitivity rows: --set identity.prior=0.01,
identity.kappa_scale=0.7, identity.views=3, ... (the evidence of both splits must then be rebuilt).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pgntrack.config import apply_overrides, load_config, merge  # noqa: E402
from pgntrack.data import load_dataset  # noqa: E402
from pgntrack.identity.embed import FrameEncoder, make_embedder  # noqa: E402
from pgntrack.identity.offline import decide_split, read_rows, relabel, sequence_evidence  # noqa: E402
from pgntrack.identity.params import PAIR_SEED, IdentityParams  # noqa: E402
from pgntrack.run import write_mot  # noqa: E402

_EMBED = None


def _init_worker(cfg, device):
    global _EMBED
    cv2.setNumThreads(1)
    _EMBED = make_embedder(cfg, device)


def _evidence(job):
    """Evidence of one sequence (worker)."""
    cfg, seq, k, base = job
    ds = load_dataset(cfg)
    frames = ds.frames(seq)
    h, w = cv2.imread(frames[0]).shape[:2]
    rows = read_rows(os.path.join(base, seq + '.txt'))
    return sequence_evidence(rows, (w, h), IdentityParams.from_config(cfg['identity']), FrameEncoder(_EMBED, frames),
                             seq, PAIR_SEED + k)


def split_evidence(cfg, ds, base, workers, device):
    """Evidence of every sequence of the split, in sorted order."""
    jobs = [(cfg, s, k, base) for k, s in enumerate(ds.seqs)]
    if workers <= 1:
        _init_worker(cfg, device)
        return [_evidence(j) for j in jobs]
    with mp.get_context('spawn').Pool(workers, initializer=_init_worker, initargs=(cfg, device)) as pool:
        return pool.map(_evidence, jobs, chunksize=1)


def summary(params, fits, videos):
    """One line: calibration, (kappa, pi), exchanges, interactions."""
    ka, pa = np.array([f[0] for f in fits]), np.array([f[1] for f in fits])
    n_ex = sum(len(f[2]) for f in fits)
    n_t = sum(len(v['inter']) for v in videos)
    n_e = sum(x['D'] is not None for v in videos for x in v['inter'])
    if params.type == 'online':
        cal = (f'kappa {ka[0]:.2f} -> {ka[-1]:.2f} (median {np.median(ka):.2f}), pi {100 * pa[0]:.2f}% -> '
               f'{100 * pa[-1]:.2f}% (median {100 * np.median(pa):.2f}%)')
    else:
        cal = f'kappa {ka[0]:.2f}, pi {100 * pa[0]:.2f}%'
    return f'{params.type}: {cal}; {n_ex} exchanges; {n_t} tested interactions, {n_e} with evidence'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', required=True, help='ours.yaml | ours_batch.yaml | ours_cross.yaml')
    ap.add_argument('--data', required=True, help='dataset config (configs/datasets)')
    ap.add_argument('--base', required=True, help='folder with the base tracker outputs <seq>.txt')
    ap.add_argument('--out', required=True, help='output folder')
    ap.add_argument('--warm', default='', help='evidence.json of the warm-start dataset (online, cross)')
    ap.add_argument('--evidence', default='', help='evidence.json of this split (skips the embeddings)')
    ap.add_argument('--workers', type=int, default=8, help='parallel sequences for the evidence')
    ap.add_argument('--device', default=None, help='device of the embedder (default: reid.device)')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='overrides, e.g. identity.prior=0.01')
    args = ap.parse_args()

    cfg = apply_overrides(merge(load_config(args.method), load_config(args.data)), args.set)
    params = IdentityParams.from_config(cfg.get('identity'))
    if params.type == 'none':
        raise SystemExit('identity.type is none')
    if params.type in ('online', 'cross') and not args.warm:
        raise SystemExit(f'{params.type} calibration needs --warm (evidence.json of the warm-start dataset)')
    ds = load_dataset(cfg)
    os.makedirs(args.out, exist_ok=True)

    if args.evidence:
        with open(args.evidence) as f:
            videos = json.load(f)
        if [v['seq'] for v in videos] != ds.seqs:
            raise SystemExit(f'{args.evidence} does not cover the sequences of {ds.name} in sorted order')
    else:
        videos = split_evidence(cfg, ds, args.base, args.workers, args.device)
    with open(os.path.join(args.out, 'evidence.json'), 'w') as f:
        json.dump(videos, f)
    warm = []
    if args.warm:
        with open(args.warm) as f:
            warm = json.load(f)

    fits = decide_split(videos, warm, params)
    log = []
    for v, (kappa, pi, ex) in zip(videos, fits):
        rows = read_rows(os.path.join(args.base, v['seq'] + '.txt'))
        write_mot(os.path.join(args.out, v['seq'] + '.txt'), relabel(rows, [x[:3] for x in ex]))
        log.append(dict(seq=v['seq'], kappa=kappa, pi=pi,
                        exchanges=[dict(f0=f0, a=a, b=b, s=r['s'], e=r['e'], D=r['D']) for f0, a, b, r in ex]))
    with open(os.path.join(args.out, 'identity.json'), 'w') as f:
        json.dump(dict(identity=cfg['identity'], base=args.base, warm=args.warm, videos=log), f, indent=1)
    print(f'{ds.name}: {summary(params, fits, videos)} -> {args.out}')


if __name__ == '__main__':
    main()
