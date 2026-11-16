"""HOTA / CLEAR / Identity with TrackEval, using the evaluation settings of the paper.

SportsMOT     : MotChallenge2DBox, benchmark 'custom', default preprocessing (as the SportsMOT evaluation script).
SoccerNet     : MotChallenge2DBox, DO_PREPROC False (as sn-tracking's evaluate_soccernet_v3_tracking.py).
MOT17 / DanceTrack : benchmark 'MOT17', default preprocessing (pedestrian class; boxes matched to distractor
                classes and zero-marked ground truth are removed), as in their official commands.
"""
import contextlib
import io
import os
import tempfile

import numpy as np


def evaluate(results_dir: str, dataset, seqs=None, quiet=True) -> dict:
    """Metrics of the MOT files `results_dir/<seq>.txt` -> {seq: {...}, 'COMBINED': {...}} (percentages)."""
    import trackeval
    if not dataset.benchmark:
        raise ValueError(f'{dataset.name} has no ground truth to evaluate against')
    seqs = list(seqs or dataset.seqs)
    results_dir = os.path.abspath(results_dir)
    with tempfile.TemporaryDirectory() as tmp:
        seqmap = os.path.join(tmp, 'seqmap.txt')
        with open(seqmap, 'w') as f:
            f.write('name\n' + '\n'.join(seqs) + '\n')
        eval_cfg = trackeval.Evaluator.get_default_eval_config()
        eval_cfg.update(dict(USE_PARALLEL=True, NUM_PARALLEL_CORES=4, PRINT_RESULTS=False, PRINT_ONLY_COMBINED=True,
                             PRINT_CONFIG=False, TIME_PROGRESS=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False,
                             PLOT_CURVES=False, DISPLAY_LESS_PROGRESS=True, OUTPUT_EMPTY_CLASSES=False))
        ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        ds_cfg.update(dict(GT_FOLDER=dataset.root, TRACKERS_FOLDER=os.path.dirname(results_dir),
                           TRACKERS_TO_EVAL=[os.path.basename(results_dir)], TRACKER_SUB_FOLDER='',
                           SKIP_SPLIT_FOL=True, SEQMAP_FILE=seqmap, BENCHMARK=dataset.benchmark, SPLIT_TO_EVAL='val',
                           DO_PREPROC=dataset.preproc, CLASSES_TO_EVAL=['pedestrian'], PRINT_CONFIG=False,
                           OUTPUT_FOLDER=os.path.join(tmp, 'out')))
        metrics = [trackeval.metrics.HOTA(dict(PRINT_CONFIG=False)), trackeval.metrics.CLEAR(dict(PRINT_CONFIG=False)),
                   trackeval.metrics.Identity(dict(PRINT_CONFIG=False))]
        evaluator = trackeval.Evaluator(eval_cfg)
        with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
            res, _ = evaluator.evaluate([trackeval.datasets.MotChallenge2DBox(ds_cfg)], metrics)
    out = {}
    for seq, v in res['MotChallenge2DBox'][os.path.basename(results_dir)].items():
        h, c, i = v['pedestrian']['HOTA'], v['pedestrian']['CLEAR'], v['pedestrian']['Identity']
        out['COMBINED' if seq == 'COMBINED_SEQ' else seq] = dict(
            HOTA=100 * float(np.mean(h['HOTA'])), DetA=100 * float(np.mean(h['DetA'])),
            AssA=100 * float(np.mean(h['AssA'])), LocA=100 * float(np.mean(h['LocA'])),
            MOTA=100 * float(c['MOTA']), IDF1=100 * float(i['IDF1']), IDP=100 * float(i['IDP']),
            IDR=100 * float(i['IDR']), IDSW=int(c['IDSW']), Frag=int(c['Frag']), FP=int(c['CLR_FP']),
            FN=int(c['CLR_FN']), MT=int(c['MT']), ML=int(c['ML']))
    return out
