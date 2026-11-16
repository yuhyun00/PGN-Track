"""Latency protocol of the paper: single process, one CPU thread, detection and JPEG decoding excluded,
the first 400 frames of five evenly spaced sequences per split, warm-up frames of each sequence discarded."""
import numpy as np

N_SEQS = 5
MAX_FRAMES = 400
WARMUP = 50


def timing_sequences(seqs, n=N_SEQS):
    """n evenly spaced sequences of the sorted split."""
    return [seqs[i] for i in np.linspace(0, len(seqs) - 1, n).astype(int)]


def warmup_frames(max_frames):
    """Frames discarded at the start of each sequence."""
    return min(WARMUP, max_frames // 4)


def latency_stats(frame_times, warmup):
    """frame_times: one (n, 2) array per sequence of (motion stage, rest) seconds -> ms statistics."""
    ft = np.concatenate([np.asarray(f)[warmup:] for f in frame_times]) * 1e3
    motion, rest = ft[:, 0], ft[:, 1]
    total = motion + rest
    return dict(motion_ms=float(motion.mean()), motion_p50=float(np.median(motion)),
                motion_p95=float(np.percentile(motion, 95)), assoc_ms=float(rest.mean()), total_ms=float(total.mean()),
                fps=float(1000 / total.mean()), frames=int(len(total)))
