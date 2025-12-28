"""Backward-compatible wrapper for the standard training loop."""

import argparse

from .loops.standard import train as _core_train


def train(args: argparse.Namespace):
    """Delegate to the standard training loop implementation."""
    return _core_train(args)
