"""``gespo seed`` subcommand: generate seed trajectories for pretrain + replay.

Builds the argparse subtree (OpenMM-free, so `--help` is cheap) and dispatches to
the walker methods (``metad`` / ``high-temp``) or the RMSD-bias adapter. Output
walker DCDs feed ``gespo pretrain --source`` and ``gespo train --replay-only-dcd-dir``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import walkers as _walkers
from .cvs import CV_SETS
from .methods import rmsd_bias as _rmsd_bias

_CV_SET_CHOICES = sorted(CV_SETS)


def _add_solvent_flags(p: argparse.ArgumentParser) -> None:
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--explicit-solvent", action="store_true",
                     help="Explicit TIP3P solvent + PME (default).")
    grp.add_argument("--implicit-solvent", action="store_true",
                     help="Implicit (GBn2) solvent — no box/barostat (e.g. small proteins).")
    p.add_argument("--forcefield-protein", default="amber14-all.xml")
    p.add_argument("--forcefield-water", default="amber14/tip3p.xml")
    p.add_argument("--forcefield-implicit", default="implicit/gbn2.xml")


def _explicit(args) -> bool:
    return not bool(getattr(args, "implicit_solvent", False))


def build_seed_subparser(subparsers) -> None:
    """Add the ``seed`` subparser (with metad / high-temp / rmsd-bias) to ``subparsers``."""
    seed = subparsers.add_parser(
        "seed", help="Generate seed trajectories (metadynamics / high-T / RMSD-bias)")
    methods = seed.add_subparsers(dest="seed_method")

    # ── metad ────────────────────────────────────────────────────────────
    m = methods.add_parser("metad", help="Well-tempered metadynamics seeding")
    m.add_argument("--pdb", required=True, help="Input PDB (holo for cv-set holo).")
    m.add_argument("--out-dir", required=True, help="Output dir; walkers -> <out>/walker_i/.")
    m.add_argument("--cv-set", default="holo", choices=_CV_SET_CHOICES,
                   help="Which 2 CVs to bias (holo|apo|folding).")
    m.add_argument("--ligand-smiles", default="", help="Ligand SMILES (registers GAFF-2.11).")
    m.add_argument("--pocket-ref-pdb", default="",
                   help="Holo reference PDB for pocket detection (use with an apo --pdb).")
    m.add_argument("--n-walkers", type=int, default=1)
    m.add_argument("--device", default="cuda:0", help="Device, or comma-list for fan-out.")
    m.add_argument("--platform", default="CUDA")
    m.add_argument("--bias-dir", default="",
                   help="Shared metaD bias dir (default <out>/bias; walkers share it).")
    m.add_argument("--production-ns", type=float, default=25.0)
    m.add_argument("--equilibration-ns", type=float, default=2.0,
                   help="Equilibration before metadynamics (ns); 0 is useful for a smoke test only.")
    m.add_argument("--save-ps", type=float, default=10.0)
    m.add_argument("--dt-fs", type=float, default=2.0)
    m.add_argument("--temperature", type=float, default=298.0)
    m.add_argument("--bias-factor", type=float, default=10.0)
    m.add_argument("--hill-height", type=float, default=1.0)
    m.add_argument("--hill-freq", type=int, default=500)
    m.add_argument("--pocket-cutoff-nm", type=float, default=0.5)
    m.add_argument("--cv1-max-nm", type=float, default=None)
    m.add_argument("--cv2-min-nm", type=float, default=None)
    m.add_argument("--cv2-max-nm", type=float, default=None)
    m.add_argument("--no-lig-restraint", dest="lig_restraint", action="store_false",
                   help="Disable the ligand↔pocket flat-bottom restraint (holo default: on).")
    m.set_defaults(lig_restraint=True)
    m.add_argument("--lig-restraint-r0-nm", type=float, default=1.0)
    m.add_argument("--lig-restraint-k", type=float, default=1000.0)
    _add_solvent_flags(m)

    # ── high-temp ─────────────────────────────────────────────────────────
    h = methods.add_parser("high-temp", help="High-temperature MD seeding")
    h.add_argument("--pdb", required=True)
    h.add_argument("--out-dir", required=True, help="Output dir; walkers -> <out>/walker_i/.")
    h.add_argument("--temperature", type=float, required=True, help="Production temperature (K).")
    h.add_argument("--n-walkers", type=int, default=1)
    h.add_argument("--device", default="cuda:0", help="Device, or comma-list for fan-out.")
    h.add_argument("--platform", default="CUDA")
    h.add_argument("--production-ns", type=float, default=50.0)
    h.add_argument("--save-ps", type=float, default=10.0)
    h.add_argument("--ckpt-ns", type=float, default=1.0)
    h.add_argument("--dt-fs", type=float, default=2.0)
    h.add_argument("--ligand-smiles", default="")
    h.add_argument("--restart", action="store_true", help="Resume production from checkpoint.")
    h.add_argument("--ligand-pocket-restraint", action="store_true")
    h.add_argument("--pocket-radius-nm", type=float, default=1.5)
    h.add_argument("--pocket-k", type=float, default=1000.0)
    h.add_argument("--pocket-seq-pos", type=int, default=170)
    _add_solvent_flags(h)

    # ── rmsd-bias (adapter over bootstrap_cli) ────────────────────────────
    rb = methods.add_parser("rmsd-bias", help="RMSD-biased MD seeding (single trajectory)")
    _rmsd_bias.add_rmsd_bias_args(rb)


def _metad_kwargs(args) -> dict:
    return dict(
        pdb=args.pdb, cv_set=args.cv_set, ligand_smiles=args.ligand_smiles,
        pocket_ref_pdb=args.pocket_ref_pdb, explicit=_explicit(args),
        platform_name=args.platform, production_ns=args.production_ns,
        equilibration_ns=args.equilibration_ns,
        save_ps=args.save_ps, dt_fs=args.dt_fs, temperature=args.temperature,
        bias_factor=args.bias_factor, hill_height=args.hill_height, hill_freq=args.hill_freq,
        pocket_cutoff_nm=args.pocket_cutoff_nm, cv1_max_nm=args.cv1_max_nm,
        cv2_min_nm=args.cv2_min_nm, cv2_max_nm=args.cv2_max_nm,
        lig_restraint=args.lig_restraint, lig_restraint_r0_nm=args.lig_restraint_r0_nm,
        lig_restraint_k=args.lig_restraint_k, forcefield_protein=args.forcefield_protein,
        forcefield_water=args.forcefield_water, forcefield_implicit=args.forcefield_implicit,
    )


def _high_temp_kwargs(args) -> dict:
    return dict(
        pdb=args.pdb, temperature=args.temperature, explicit=_explicit(args),
        platform_name=args.platform, production_ns=args.production_ns, save_ps=args.save_ps,
        ckpt_ns=args.ckpt_ns, dt_fs=args.dt_fs, ligand_smiles=args.ligand_smiles,
        restart=args.restart, ligand_pocket_restraint=args.ligand_pocket_restraint,
        pocket_radius_nm=args.pocket_radius_nm, pocket_k=args.pocket_k,
        pocket_seq_pos=args.pocket_seq_pos, forcefield_protein=args.forcefield_protein,
        forcefield_water=args.forcefield_water, forcefield_implicit=args.forcefield_implicit,
    )


def cmd_seed(args, argv=None) -> None:
    """Dispatch a ``gespo seed`` invocation."""
    method = getattr(args, "seed_method", None)
    if method == "metad":
        bias = args.bias_dir or str(Path(args.out_dir) / "bias")
        dcds = _walkers.run_method(
            "metad", out_dir=args.out_dir, n_walkers=args.n_walkers,
            device=args.device, bias_dir=bias, method_kwargs=_metad_kwargs(args))
        print("[seed] metad done. Wrote:\n  " + "\n  ".join(dcds), flush=True)
    elif method == "high-temp":
        dcds = _walkers.run_method(
            "high-temp", out_dir=args.out_dir, n_walkers=args.n_walkers,
            device=args.device, method_kwargs=_high_temp_kwargs(args))
        print("[seed] high-temp done. Wrote:\n  " + "\n  ".join(dcds), flush=True)
    elif method == "rmsd-bias":
        _rmsd_bias.run_rmsd_bias(args, argv=argv)
    else:
        raise SystemExit("Usage: gespo seed {metad|high-temp|rmsd-bias} --help")
