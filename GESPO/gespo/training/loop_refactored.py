"""Backward-compatible wrapper for the legacy refactored loop entrypoint."""

import argparse

from .loops.standard import train as _core_train


def train(args: argparse.Namespace):
    """
    Thin wrapper that delegates to the main training loop.

    This file is kept only for backward compatibility with older scripts that
    imported `gespo.training.loop_refactored.train`; all logic now lives in
    `gespo.training.loops.standard.train`.
    """
    return _core_train(args)
