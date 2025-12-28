"""Training loop entrypoints."""

from .generative import train as train_generative
from .standard import train as train_standard

__all__ = ["train_generative", "train_standard"]
