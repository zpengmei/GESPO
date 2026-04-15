"""Training subpackage: RL loop and agents.

This module intentionally avoids importing heavy simulation dependencies (e.g.
OpenMM) at import time. Training loops are imported lazily inside `train()`.
"""


def train(args):
    """
    Dispatch to the appropriate training loop.
    
    By default uses the standard loop; when args.generative_loop is True,
    runs the torsion-autoregressive generative loop.
    """
    if getattr(args, "toy_phi_psi_dga_loop", False):
        from . import loop_toy_phi_psi_dga as _loop_toy_phi_psi_dga

        _loop_toy_phi_psi_dga.train(args)
        return

    if getattr(args, "toy_phi_psi_loop", False):
        from . import loop_toy_phi_psi as _loop_toy_phi_psi

        _loop_toy_phi_psi.train(args)
        return

    if getattr(args, "path_sampling", False) and not getattr(args, "generative_loop", False):
        raise ValueError("--path-sampling currently requires --generative-loop.")

    if getattr(args, "generative_loop", False):
        if getattr(args, "path_sampling", False):
            from . import loop_generative_path as _loop_generative_path

            _loop_generative_path.train(args)
            return
        from . import loop_generative as _loop_generative

        _loop_generative.train(args)
        return

    from . import loop as _loop_standard

    _loop_standard.train(args)


def __getattr__(name: str):
    if name == "GRPOAgent":
        try:
            from .rl import GRPOAgent as _GRPOAgent
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "GRPOAgent requires OpenMM; install `openmm` (or use the OpenMM conda env) to import it."
            ) from e
        return _GRPOAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "train",
    "GRPOAgent",

]
