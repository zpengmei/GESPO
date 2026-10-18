"""CLI entrypoint with subcommands for GESPO.

Usage:
    gespo seed metad --pdb a.pdb --out-dir seeds/  # generate seed trajectories
    gespo pretrain --pdb a.pdb --source seeds/ --out r0.pt  # TorsionAR pretraining
    gespo train --config config.yaml              # training from YAML
    gespo train --pdb a.pdb --generative-loop ...  # training from CLI flags
    gespo config template --mode generative        # generate config template
    gespo config validate config.yaml              # validate a config file
    gespo bootstrap rmsd-bias --help               # bootstrap utilities
"""
import argparse
import atexit
import datetime as _dt
import os
import shutil
import sys
from pathlib import Path

from .config import TrainingConfig
# token_sets is dependency-light (no OpenMM/torch), so importing it to build help
# text stays cheap. The authoritative KNOWN_GLOBAL_TOKEN_SETS (validated at run
# time in pretrain_torsion_ar) lives there too.
from .training.token_sets import COMMON_GLOBAL_TOKEN_SETS as _COMMON_TOKEN_SETS


# ---------------------------------------------------------------------------
# Helpers (preflight, logging, snapshot)
# ---------------------------------------------------------------------------

def _preflight_compatibility_warnings(args: argparse.Namespace) -> None:
    """Print warnings for common configuration incompatibilities."""
    try:
        projection = str(getattr(args, "projection", "") or "")
        if projection != "geom2vec_vamp":
            return
        n_md_steps = int(getattr(args, "n_md_steps", 0) or 0)
        record_stride = int(getattr(args, "record_stride", 0) or 0)
        embed_stride = max(1, int(getattr(args, "geom2vec_embed_stride", 1) or 1))
        lag_raw = max(1, int(getattr(args, "geom2vec_lagtime", 1) or 1))
        saved_frames = (n_md_steps // record_stride) + 1 if n_md_steps > 0 and record_stride > 0 else 0
        embedded_frames = (saved_frames + embed_stride - 1) // embed_stride if saved_frames > 0 else 0
        lag_embed = max(1, int(round(float(lag_raw) / float(embed_stride))))

        if int(getattr(args, "geom2vec_dim", 2) or 2) < 2:
            print("[Preflight] Warning: geom2vec_dim < 2; VAMP rewards will be degenerate.", flush=True)
        if saved_frames > 0 and embedded_frames <= lag_embed:
            print(
                f"[Preflight] Warning: incompatible sampling/lag: saved_frames≈{saved_frames}, "
                f"embedded_frames≈{embedded_frames}, lag_embed≈{lag_embed}. "
                "Need embedded_frames > lag_embed.", flush=True,
            )
    except Exception:
        pass


def _preflight_pdb_present(args: argparse.Namespace) -> None:
    """Require a structure for non-toy runs.

    ``pdb`` no longer silently defaults to a phantom ``chignolin.pdb``; fail
    early with a clear message instead of a downstream file-not-found.
    """
    is_toy = bool(
        getattr(args, "toy_phi_psi_loop", False)
        or getattr(args, "toy_phi_psi_dga_loop", False)
    )
    if is_toy:
        return
    pdb = str(getattr(args, "pdb", "") or "").strip()
    if not pdb:
        raise ValueError(
            "No structure provided: set --pdb (or endpoints.pdb_a in YAML) to a PDB file. "
            "There is no default structure."
        )


def _preflight_geom2vec_checkpoint(args: argparse.Namespace) -> None:
    """Auto-resolve a geom2vec checkpoint for the ``geom2vec_vamp`` projection.

    If ``--geom2vec-ckpt`` is unset (or a bare filename), locate a checkpoint from
    the geom2vec install / ``$GEOM2VEC_CKPT`` and fill it in, so a run does not fail
    with "geom2vec checkpoint required" just because the path was not spelled out.
    """
    if str(getattr(args, "projection", "") or "") != "geom2vec_vamp":
        return
    if bool(getattr(args, "toy_phi_psi_loop", False)
            or getattr(args, "toy_phi_psi_dga_loop", False)):
        return
    from .utils.geom2vec_paths import resolve_geom2vec_checkpoint

    current = str(getattr(args, "geom2vec_checkpoint", "") or "").strip()
    resolved = resolve_geom2vec_checkpoint(current)
    if resolved and resolved != current:
        args.geom2vec_checkpoint = resolved
        if current:
            print(f"[Preflight] Resolved geom2vec checkpoint '{current}' -> {resolved}", flush=True)
        else:
            print(f"[Preflight] Auto-detected geom2vec checkpoint: {resolved}", flush=True)
    elif not resolved and not current:
        print(
            "[Preflight] Warning: projection=geom2vec_vamp but no geom2vec checkpoint "
            "was found. Set --geom2vec-ckpt or $GEOM2VEC_CKPT (see docs/inputs.md).",
            flush=True,
        )


def _preflight_endpoint_validation(args: argparse.Namespace) -> None:
    """Validate endpoint A/B PDBs when provided."""
    pdb_a = str(getattr(args, "pdb", "") or "").strip()
    pdb_b = str(getattr(args, "pdb_b", "") or "").strip()
    if bool(getattr(args, "path_sampling", False)) and not pdb_b:
        raise ValueError("--path-sampling requires --pdb-b.")
    if not pdb_b:
        return
    from .utils.endpoints import validate_endpoint_pdbs
    res = validate_endpoint_pdbs(pdb_a, pdb_b)
    print(f"[Preflight] Endpoints validated: {res.n_solute_residues} residues, {res.n_solute_atoms} atoms", flush=True)


def _snapshot_endpoint_pdbs(args: argparse.Namespace) -> None:
    """Copy endpoint PDBs into snapshot_dir for provenance."""
    snapshot_dir = str(getattr(args, "snapshot_dir", "") or "").strip()
    if not snapshot_dir:
        return
    out_dir = Path(snapshot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        pdb_a = str(getattr(args, "pdb", "") or "").strip()
        pdb_b = str(getattr(args, "pdb_b", "") or "").strip()
        if pdb_a:
            shutil.copyfile(pdb_a, out_dir / "endpoint_A.pdb")
        if pdb_b:
            shutil.copyfile(pdb_b, out_dir / "endpoint_B.pdb")
    except Exception:
        pass


def _setup_logging(log_file: str):
    """Set up console tee logging to a file."""
    try:
        _log_path = log_file or os.path.join(os.getcwd(), "gespo_run.log")
        _log_dir = os.path.dirname(_log_path)
        if _log_dir:
            os.makedirs(_log_dir, exist_ok=True)
        _log_fh = open(_log_path, "a", buffering=1)
        _ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log_fh.write(f"\n===== GESPO run started at {_ts} =====\n")

        class _Tee:
            def __init__(self, stream, logfh):
                self._stream = stream
                self._log = logfh
            def write(self, data):
                try: self._stream.write(data)
                except: pass
                try: self._log.write(data)
                except: pass
            def flush(self):
                try: self._stream.flush()
                except: pass
                try: self._log.flush()
                except: pass

        sys.stdout = _Tee(sys.stdout, _log_fh)
        sys.stderr = _Tee(sys.stderr, _log_fh)
        atexit.register(lambda: (_log_fh.flush(), _log_fh.close()))
        print(f"[Log] Capturing stdout/stderr to {_log_path}")
    except Exception:
        pass


def _resolve_device(args: argparse.Namespace) -> None:
    """Set args.device from args.gen_device."""
    gen_device_str = str(getattr(args, "gen_device", "cuda:0") or "cuda:0")
    if gen_device_str.startswith("cuda:"):
        args.device = int(gen_device_str.split(":")[1])
    elif gen_device_str.isdigit():
        args.device = int(gen_device_str)
    else:
        args.device = 0


def _load_config_from_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve args into a fully-defaulted Namespace.

    If --config is set, load the YAML and merge with CLI overrides.
    Otherwise, resolve straight from the CLI args.

    Every field ends up either explicitly set or filled from its
    TrainingConfig dataclass default.
    """
    config_path = str(getattr(args, "config", "") or "").strip()
    if config_path:
        from .config.loader import load_config
        cli_overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
        config = load_config(config_path, overrides=cli_overrides)
        print(f"[Config] Loaded from {config_path}")
    else:
        config = TrainingConfig.from_args(args)
    return argparse.Namespace(**config.to_dict())


def _save_effective_config(args: argparse.Namespace) -> None:
    """Auto-save effective config to snapshot_dir for reproducibility."""
    snapshot_dir = str(getattr(args, "snapshot_dir", "") or "").strip()
    if not snapshot_dir:
        return
    try:
        from .config.loader import save_config
        config_obj = TrainingConfig.from_args(args)
        effective_path = Path(snapshot_dir) / "effective_config.yaml"
        save_config(config_obj, str(effective_path))
        print(f"[Config] Saved effective config to {effective_path}")
    except Exception as exc:
        print(f"[Config] Warning: could not save effective config: {exc}")


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def _apply_mode_solvent_shortcuts(args: argparse.Namespace) -> None:
    """Expand the CLI ``--mode`` / ``--solvent`` shortcuts into concrete flags.

    Explicitly-passed boolean flags win (with the None-sentinel default, an
    explicit flag is non-None while an unset one is None). The shim attributes
    are removed so they never reach TrainingConfig (which has no such fields).
    """
    from .config.loader import mode_to_flags, solvent_to_flags

    for attr, to_flags in (("mode", mode_to_flags), ("solvent", solvent_to_flags)):
        value = getattr(args, attr, None)
        if value:
            for flag, flag_value in to_flags(value).items():
                if getattr(args, flag, None) is None:  # not explicitly set
                    setattr(args, flag, flag_value)
        if hasattr(args, attr):
            delattr(args, attr)


def _cmd_train(args: argparse.Namespace) -> None:
    """Run GESPO training."""
    from .training import train
    from .utils.config import RunConfigurer

    _apply_mode_solvent_shortcuts(args)
    args = _load_config_from_args(args)
    _resolve_device(args)
    _preflight_geom2vec_checkpoint(args)

    log_file = os.path.join(args.snapshot_dir, "gespo.log") if getattr(args, "snapshot_dir", "") else "gespo.log"
    _setup_logging(log_file)

    config_log_dir = getattr(args, "snapshot_dir", "") or ""
    RunConfigurer(args, log_dir=config_log_dir).print_summary()

    _save_effective_config(args)
    _preflight_pdb_present(args)
    _preflight_endpoint_validation(args)
    _snapshot_endpoint_pdbs(args)
    _preflight_compatibility_warnings(args)
    train(args)


# ---------------------------------------------------------------------------
# Subcommand: pretrain
# ---------------------------------------------------------------------------

def _cmd_pretrain(args: argparse.Namespace) -> None:
    """Pretrain a TorsionAR policy from trajectory data."""
    from .training.pretrain_torsion_ar import pretrain_torsion_ar

    source = str(getattr(args, "source", "") or "").strip()
    if not source:
        raise ValueError("--source is required (path to DCD directory or trajectory file).")
    out_path = str(getattr(args, "out", "") or "").strip()
    if not out_path:
        raise ValueError("--out is required (output checkpoint path).")

    pretrain_torsion_ar(
        source_path=source,
        out_path=out_path,
        batch_size=int(getattr(args, "batch_size", 64) or 64),
        epochs=int(getattr(args, "epochs", 10) or 10),
        lr=float(getattr(args, "lr", 1e-4) or 1e-4),
        device=(str(getattr(args, "device", "") or "").strip() or None),
        pdb_path=(str(getattr(args, "pdb", "") or "").strip() or None),
        pdb_b_path=(str(getattr(args, "pdb_b", "") or "").strip() or None),
        stride=int(getattr(args, "stride", 1) or 1),
        global_token_set=str(getattr(args, "global_token_set", "rmsd_e2e") or "rmsd_e2e"),
        torsion_mode=str(getattr(args, "torsion_mode", "backbone") or "backbone"),
        ligand_smiles=str(getattr(args, "ligand_smiles", "") or ""),
        reweight_zenodo_dir=str(getattr(args, "reweight_zenodo_dir", "") or ""),
        reweight_ref_pdb=str(getattr(args, "reweight_ref_pdb", "") or ""),
        reweight_labels_npz=str(getattr(args, "reweight_labels_npz", "") or ""),
        reweight_alpha=float(getattr(args, "reweight_alpha", 1.0) or 1.0),
        pocket_ref_pdb=str(getattr(args, "pocket_ref_pdb", "") or ""),
    )


# ---------------------------------------------------------------------------
# Subcommand: config
# ---------------------------------------------------------------------------

def _cmd_config(args: argparse.Namespace) -> None:
    """Config utilities (template, validate)."""
    sub = getattr(args, "config_cmd", None)
    if sub == "template":
        from .config import generate_template
        mode = str(getattr(args, "mode", "generative") or "generative")
        print(generate_template(mode))
    elif sub == "validate":
        config_path = str(getattr(args, "config_file", "") or "").strip()
        if not config_path:
            print("Error: provide a config file path to validate.", file=sys.stderr)
            sys.exit(1)
        from .config.loader import load_config
        try:
            config = load_config(config_path, strict=bool(getattr(args, "strict", False)))
            print(f"Config loaded successfully from {config_path}")
            print(f"  mode: {'path-sampling' if config.path_sampling else ('generative' if config.generative_loop else 'standard')}")
            print(f"  pdb: {config.pdb}")
            print(f"  episodes: {config.episodes}")
            print(f"  grp_size: {config.grp_size}")
            # Run preflight checks
            import argparse as _ap
            ns = _ap.Namespace(**config.to_dict())
            _preflight_compatibility_warnings(ns)
            try:
                _preflight_endpoint_validation(ns)
            except Exception as e:
                print(f"  Endpoint validation: FAILED ({e})")
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Usage: gespo config {template,validate}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="gespo",
        description="GESPO: RL-based enhanced sampling with geom2vec VAMP backbone.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- gespo train ---
    # `--help` shows only the curated essential flags; `--help-all` shows every
    # flag. Detect it up front so the parser is built with full help text when
    # requested (SUPPRESS is baked in at add_argument time, so the choice must be
    # made before populating).
    _show_advanced = "--help-all" in sys.argv
    train_parser = subparsers.add_parser(
        "train", help="Run GESPO training",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "" if _show_advanced else
            "Showing essential options only. Run 'gespo train --help-all' to see "
            "all options."
        ),
    )
    train_parser.add_argument("--config", "-c", default="", metavar="YAML",
        help="Path to YAML config file. CLI flags override YAML values.")
    train_parser.add_argument("--help-all", action="help",
        help="Show all options, including advanced ones, and exit.")
    train_parser.add_argument("--mode", choices=["generative", "path-sampling", "standard"],
        default=None,
        help="Training mode shortcut: sets --generative-loop / --path-sampling.")
    train_parser.add_argument("--solvent", choices=["explicit", "implicit", "gbsa", "gbn2"],
        default=None,
        help="Solvent shortcut: sets --implicit-solvent / --explicit-solvent.")
    TrainingConfig.populate_parser(train_parser, show_advanced=_show_advanced)

    # --- gespo pretrain ---
    pretrain_parser = subparsers.add_parser("pretrain", help="Pretrain TorsionAR policy")
    pretrain_parser.add_argument("--source", required=True, help="DCD directory or trajectory file")
    pretrain_parser.add_argument("--out", required=True, help="Output checkpoint path")
    pretrain_parser.add_argument("--pdb", default="", help="PDB file for topology")
    pretrain_parser.add_argument("--pdb-b", default="", help="Endpoint B PDB (optional)")
    pretrain_parser.add_argument("--batch-size", type=int, default=64)
    pretrain_parser.add_argument("--epochs", type=int, default=10)
    pretrain_parser.add_argument("--lr", type=float, default=1e-4)
    pretrain_parser.add_argument("--device", default="", help="Device (e.g., cuda:0)")
    pretrain_parser.add_argument("--stride", type=int, default=1)
    pretrain_parser.add_argument(
        "--global-token-set", default="rmsd_e2e",
        help="Global CV tokens the AR policy conditions on. Common: "
             + ", ".join(_COMMON_TOKEN_SETS)
             + ". See docs/reference.md for the full catalog.")
    pretrain_parser.add_argument(
        "--torsion-mode", default="backbone",
        choices=["backbone", "backbone+sidechain", "backbone_chi"],
        help="Torsion tokens: 'backbone' (phi/psi/omega) or 'backbone+sidechain' "
             "(adds chi1-chi4). 'backbone_chi' is a legacy alias for "
             "'backbone+sidechain'.")
    pretrain_parser.add_argument("--ligand-smiles", default="",
                                  help="SMILES of the ligand (for RDKit-based rotatable bond detection). "
                                       "When provided, uses standard cheminformatics rotatable bond definition "
                                       "(allows H as 4th dihedral atom).")
    pretrain_parser.add_argument("--reweight-zenodo-dir", default="",
                                  help="If set, cluster training frames into 7 macrostates (loop RMSDs vs 3PTB, "
                                       "k-means anchored on Zenodo cnew-2micro trajectories) and use importance "
                                       "weights to balance the per-state contribution (uniform across states).")
    pretrain_parser.add_argument("--pocket-ref-pdb", default="",
                                  help="Reference PDB with bound ligand for pocket auto-detection. "
                                       "Use when --pdb is an apo structure without ligand.")
    pretrain_parser.add_argument("--reweight-ref-pdb", default="",
                                  help="Reference PDB for loop RMSD computation (default: same as --pdb).")
    pretrain_parser.add_argument("--reweight-labels-npz", default="",
                                  help="Path to precomputed labels.npz (from compute_vamp_reweight.py). "
                                       "Provides per-frame cluster labels; weights = 1/cluster_count^alpha. "
                                       "Use this for clean (non-Plattner) reweighting in geom2vec-VAMP space.")
    pretrain_parser.add_argument("--reweight-alpha", type=float, default=1.0,
                                  help="Exponent for reweighting: weight = 1/count^alpha. "
                                       "alpha=1 is standard uniform; alpha=2 aggressively upweights rare clusters.")

    # --- gespo config ---
    config_parser = subparsers.add_parser("config", help="Config utilities")
    config_sub = config_parser.add_subparsers(dest="config_cmd")
    template_parser = config_sub.add_parser("template", help="Generate a YAML config template")
    template_parser.add_argument("--mode", default="generative",
        choices=["generative", "path-sampling", "standard"],
        help="Training mode for the template")
    validate_parser = config_sub.add_parser("validate", help="Validate a config file")
    validate_parser.add_argument("config_file", help="Path to YAML config file")
    validate_parser.add_argument("--strict", action="store_true",
        help="Raise on unknown config keys/sections instead of warning.")

    # --- gespo seed ---
    # Seed generation lives in gespo/seed (dependency-light parser build).
    from .seed.cli import build_seed_subparser
    build_seed_subparser(subparsers)

    # --- gespo bootstrap ---
    subparsers.add_parser("bootstrap", help="Bootstrap utilities (run: python -m gespo.bootstrap_cli)")

    # --- Backward compat: if no subcommand, treat as 'train' ---
    # Check if the first positional arg is a known subcommand.
    # If not (e.g., user passed --pdb directly), insert 'train' as the subcommand.
    _KNOWN_COMMANDS = ("train", "pretrain", "config", "seed", "bootstrap", "-h", "--help")
    if len(sys.argv) > 1 and sys.argv[1] not in _KNOWN_COMMANDS:
        sys.argv.insert(1, "train")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "pretrain":
        _cmd_pretrain(args)
    elif args.command == "config":
        _cmd_config(args)
    elif args.command == "seed":
        from .seed.cli import cmd_seed
        cmd_seed(args, argv=sys.argv)
    elif args.command == "bootstrap":
        print("Run bootstrap utilities with: python -m gespo.bootstrap_cli --help")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
