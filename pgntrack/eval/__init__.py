"""Evaluation: TrackEval with the paper's settings and the paired per-sequence bootstrap."""
from .bootstrap import paired_bootstrap
from .trackeval_wrapper import evaluate

__all__ = ['evaluate', 'paired_bootstrap']
