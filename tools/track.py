"""Run a method config on a dataset config; writes <out>/<seq>.txt (MOTChallenge format) and <out>/config.yaml.

  python tools/track.py --method configs/methods/pgn_track.yaml --data configs/datasets/smvu.yaml
  python tools/track.py --method configs/ablations/pgn_one_layer.yaml --data configs/datasets/soccernet_test.yaml \
      --workers 16 --set motion.pgn.device=cpu
  python tools/track.py --method configs/methods/pgn_track_reid.yaml --data configs/datasets/sportsmot_val.yaml \
      --workers 8 --gpus 0,1                      # re-ID weights under $WEIGHTS_ROOT
  python tools/track.py --method configs/methods/mcbytepp.yaml --data configs/datasets/smvu.yaml \
      --workers 4 --gpus 0,1 --set masks.offload=true
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):   # one thread per tracker process
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pgntrack.config import apply_overrides, load_config, merge  # noqa: E402
from pgntrack.data import load_dataset  # noqa: E402


def resolve_config(method, data, overrides=()):
    """Method config, then dataset config (data / eval / tracker / output sections), then overrides."""
    return apply_overrides(merge(load_config(method), load_config(data)), overrides)


def _init_worker(gpus):
    """Pin each worker process to one GPU of the list (round robin)."""
    if gpus:
        idx = mp.current_process()._identity[0] - 1
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus[idx % len(gpus)]


def _track(job):
    cfg, seq, out_dir = job
    from pgntrack.run import run_sequence, single_thread, write_mot
    single_thread()
    t0 = time.perf_counter()
    rows, tracker = run_sequence(cfg, load_dataset(cfg), seq)
    write_mot(os.path.join(out_dir, seq + '.txt'), rows)
    ft = tracker.frame_times
    return seq, len(ft), time.perf_counter() - t0, sum(m for m, _ in ft)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', required=True, help='method or ablation config (configs/methods, configs/ablations)')
    ap.add_argument('--data', required=True, help='dataset config (configs/datasets)')
    ap.add_argument('--out', default='', help='output folder (default: results/<method>/<dataset name>)')
    ap.add_argument('--workers', type=int, default=8, help='parallel sequences (one process each)')
    ap.add_argument('--gpus', default='', help='comma-separated GPU ids for the workers (default: unchanged)')
    ap.add_argument('--seqs', default='', help='comma-separated subset of sequences')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='overrides, e.g. motion.pgn.levels=[4]')
    args = ap.parse_args()

    cfg = resolve_config(args.method, args.data, args.set)
    ds = load_dataset(cfg)
    seqs = [s for s in ds.seqs if not args.seqs or s in args.seqs.split(',')]
    method = os.path.splitext(os.path.basename(args.method))[0]
    out_dir = args.out or os.path.join('results', method, ds.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    jobs = sorted(((cfg, s, out_dir) for s in seqs), key=lambda j: -len(ds.frames(j[1])))   # longest first
    t0 = time.time()
    if args.workers <= 1:
        _init_worker(None)
        stats = [_track(j) for j in jobs]
    else:
        gpus = [g for g in args.gpus.split(',') if g]
        with mp.get_context('spawn').Pool(args.workers, initializer=_init_worker, initargs=(gpus,)) as pool:
            stats = list(pool.imap_unordered(_track, jobs))
    frames = sum(s[1] for s in stats)
    print(f'{method} on {ds.name}: {len(stats)} sequences, {frames} frames, {time.time() - t0:.1f} s wall, '
          f'motion stage {1e3 * sum(s[3] for s in stats) / max(frames, 1):.2f} ms/frame -> {out_dir}')


if __name__ == '__main__':
    main()
