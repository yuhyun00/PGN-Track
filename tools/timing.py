"""Motion-stage latency as in the paper: one process, one CPU thread, no detection or JPEG decoding in the timed
part, the first 400 frames of five evenly spaced sequences, the first 50 frames of each sequence discarded.
FPS counts the motion stage and the association (the tracker without detection and decoding).
Run it on an otherwise idle machine; PGN-Track runs on the GPU given by CUDA_VISIBLE_DEVICES.

  python tools/timing.py --method configs/methods/pgn_track.yaml --data configs/datasets/sportsmot_val.yaml
  python tools/timing.py --method configs/methods/bytetrack_cmc.yaml --data configs/datasets/soccernet_test.yaml
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pgntrack.config import apply_overrides, load_config, merge  # noqa: E402
from pgntrack.data import load_dataset  # noqa: E402
from pgntrack.run import run_sequence, single_thread  # noqa: E402
from pgntrack.timing import MAX_FRAMES, N_SEQS, latency_stats, timing_sequences, warmup_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--seqs', type=int, default=N_SEQS, help='number of timing sequences')
    ap.add_argument('--frames', type=int, default=MAX_FRAMES, help='first frames of each sequence')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
    ap.add_argument('--json', default='', help='append the result to this JSON file')
    args = ap.parse_args()

    single_thread()
    cfg = apply_overrides(merge(load_config(args.method), load_config(args.data)), args.set)
    ds = load_dataset(cfg)
    times = []
    for seq in timing_sequences(ds.seqs, args.seqs):
        _, tracker = run_sequence(cfg, ds, seq, max_frames=args.frames)
        times.append(tracker.frame_times)
    r = latency_stats(times, warmup_frames(args.frames))
    name = os.path.splitext(os.path.basename(args.method))[0]
    print(f"{name} on {ds.name}: motion {r['motion_ms']:.2f} ms (p50 {r['motion_p50']:.2f}, p95 {r['motion_p95']:.2f}), "
          f"association {r['assoc_ms']:.2f} ms, total {r['total_ms']:.2f} ms, {r['fps']:.1f} FPS over {r['frames']} frames")
    if args.json:
        old = json.load(open(args.json)) if os.path.exists(args.json) else {}
        old[f'{name}/{ds.name}'] = r
        with open(args.json, 'w') as f:
            json.dump(old, f, indent=1)


if __name__ == '__main__':
    main()
