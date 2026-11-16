"""HOTA / IDF1 of a result folder (TrackEval, the paper's settings), optionally against a baseline folder with the
paired per-sequence bootstrap (mean difference, 10,000 resamples, seed 0, 95% percentile interval).

  python tools/evaluate.py --data configs/datasets/smvu.yaml --results results/pgn_track/sportsmot_valtr
  python tools/evaluate.py --data configs/datasets/smvu.yaml --results results/pgn_track/sportsmot_valtr \
      --baseline results/bytetrack/sportsmot_valtr
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pgntrack.config import load_config  # noqa: E402
from pgntrack.data import load_dataset  # noqa: E402
from pgntrack.eval import evaluate, paired_bootstrap  # noqa: E402


def summary(m):
    return (f"HOTA {m['HOTA']:.2f}  IDF1 {m['IDF1']:.2f}  AssA {m['AssA']:.2f}  DetA {m['DetA']:.2f}  "
            f"MOTA {m['MOTA']:.2f}  IDSW {m['IDSW']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='dataset config (configs/datasets)')
    ap.add_argument('--results', required=True, help='folder with <seq>.txt')
    ap.add_argument('--baseline', default='', help='second folder: paired bootstrap of results - baseline')
    ap.add_argument('--json', default='', help='write all metrics (per sequence and combined) to this file')
    args = ap.parse_args()

    ds = load_dataset(load_config(args.data))
    res = evaluate(args.results, ds)
    print(f'{args.results}: {summary(res["COMBINED"])}')
    out = {'results': res}
    if args.baseline:
        base = evaluate(args.baseline, ds)
        print(f'{args.baseline}: {summary(base["COMBINED"])}')
        out['baseline'] = base
        for key in ('HOTA', 'IDF1'):
            b = paired_bootstrap([res[s][key] for s in ds.seqs], [base[s][key] for s in ds.seqs])
            out[f'bootstrap_{key}'] = b
            print(f"{key}: combined {res['COMBINED'][key] - base['COMBINED'][key]:+.2f}, per-sequence mean "
                  f"{b['mean']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}] (95%), up {b['up']} / down {b['down']} of {b['n']}")
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
