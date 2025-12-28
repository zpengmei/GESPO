"""Training subpackage: RL loop and agents.

This module intentionally avoids importing heavy simulation dependencies (e.g.
OpenMM) at import time. Training loops are imported lazily inside `train()`.
"""

from .rl import GRPOAgent, ResidueTransformerPolicy
from .scorer import PCAScorer, VendiNovelty, featurize_for_pca, vendi_score


def train(args):
    """
    Dispatch to the appropriate training loop.
    
    By default uses the standard loop; when args.generative_loop is True,
    runs the torsion-autoregressive generative loop.
    """
    if getattr(args, "generative_loop", False):
        from . import loop_generative as _loop_generative

        _loop_generative.train(args)
        return

    from . import loop as _loop_standard

    _loop_standard.train(args)


__all__ = [
    "train",
    "GRPOAgent",
    "ResidueTransformerPolicy",
    "PCAScorer",
    "VendiNovelty",
    "featurize_for_pca",
    "vendi_score",
]
