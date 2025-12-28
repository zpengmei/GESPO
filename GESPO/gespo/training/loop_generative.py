"""Backward-compatible wrapper for the generative training loop."""

import argparse

from .loops.generative import train as _core_train


def train(args: argparse.Namespace):
    """Delegate to the generative training loop implementation."""
    return _core_train(args)
