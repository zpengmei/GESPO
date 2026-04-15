"""Training loop classes for GESPO.

Active loops:
- generative: TorsionAR generative loop with geom2vec+VAMP
- generative_path: Path-sampling extension with committor rewards

Legacy loops (in _legacy/):
- discrete: Original discrete-state loop with ensemble uncertainty
- toy_phi_psi: Toy phi/psi start distribution loops
"""


def train(args):
    """Dispatch to the appropriate training loop.

    This replaces the dispatcher in training/__init__.py.
    """
    if getattr(args, "toy_phi_psi_dga_loop", False):
        from ._legacy.toy_phi_psi_dga import train as _train
        _train(args)
        return

    if getattr(args, "toy_phi_psi_loop", False):
        from ._legacy.toy_phi_psi import train as _train
        _train(args)
        return

    if getattr(args, "path_sampling", False) and not getattr(args, "generative_loop", False):
        raise ValueError("--path-sampling currently requires --generative-loop.")

    if getattr(args, "generative_loop", False):
        if getattr(args, "path_sampling", False):
            from .generative_path import train as _train
            _train(args)
            return
        from .generative import train as _train
        _train(args)
        return

    # Default: legacy discrete loop
    from ._legacy.discrete import train as _train
    _train(args)
