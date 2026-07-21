"""RMSD-biased MD seeding — adapter over the proven implementation.

The RMSD-bias sampler (ramp a harmonic RMSD restraint from 0 → r_max to sweep a
protein open, optionally emitting a stratified multi-model ``start_pool.pdb``)
already exists, battle-tested, in ``gespo/bootstrap_cli.py``. To unify it under
``gespo seed`` **without physically relocating ~400 lines of OpenMM code that
cannot be re-verified in a non-OpenMM environment**, this module is a thin
adapter: the seed CLI reuses ``bootstrap_cli``'s argument builder, and the run
function delegates to ``bootstrap_cli._run_rmsd_biased_md``. Both entry points —
``gespo seed rmsd-bias`` and ``python -m gespo.bootstrap_cli rmsd-bias`` — share
one implementation.
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence


def add_rmsd_bias_args(parser) -> None:
    """Populate a parser with the RMSD-bias flags (reuses bootstrap_cli)."""
    from ...bootstrap_cli import _add_rmsd_bias_common_args
    parser.add_argument("--pdb", required=True,
                        help="Input sanitized PDB (protein-only recommended).")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory (biased traj + logs + optional start_pool.pdb).")
    _add_rmsd_bias_common_args(parser)


def run_rmsd_bias(args, argv: Optional[Sequence[str]] = None) -> int:
    """Run RMSD-biased MD via the proven bootstrap implementation.

    ``argv`` is the raw command line — passed through to the bootstrap protocol
    resolver so ``--protocol paper_310k_nacl`` only fills in flags the user did
    NOT explicitly set (matching ``bootstrap_cli.main``).
    """
    from ...bootstrap_cli import _apply_protocol_defaults, _run_rmsd_biased_md
    _apply_protocol_defaults(args, list(sys.argv if argv is None else argv))
    return _run_rmsd_biased_md(args)
