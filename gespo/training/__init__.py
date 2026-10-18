"""Training subpackage: utilities, sampler, checkpointing, optimization.

Training loops have moved to ``gespo.loops``. This module delegates to them
for backward compatibility.
"""


def train(args):
    """Dispatch to the appropriate training loop (delegates to gespo.loops)."""
    from gespo.config.validation import validate_grpo_sampling
    validate_grpo_sampling(args)
    from gespo.loops import train as _loops_train
    _loops_train(args)


def __getattr__(name: str):
    if name == "GRPOAgent":
        from gespo.policies import GRPOAgent as _GRPOAgent
        return _GRPOAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "train",
    "GRPOAgent",
]
