"""CLI entrypoint with subcommands for GESPO.

Usage:
    gespo train --config config.yaml              # training from YAML
    gespo train --pdb a.pdb --generative-loop ...  # training from CLI flags
    gespo pretrain --pdb a.pdb --source dcd_dir/   # TorsionAR pretraining
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
    """If --config is set, load YAML and merge with CLI overrides."""
    config_path = str(getattr(args, "config", "") or "").strip()
    if not config_path:
        return args
    from .config.loader import load_config
    cli_overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    config = load_config(config_path, overrides=cli_overrides)
    args = argparse.Namespace(**config.to_dict())
    print(f"[Config] Loaded from {config_path}")
    return args


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

def _cmd_train(args: argparse.Namespace) -> None:
    """Run GESPO training."""
    from .training import train
    from .utils.config import RunConfigurer

    args = _load_config_from_args(args)
    _resolve_device(args)

    log_file = os.path.join(args.snapshot_dir, "gespo.log") if getattr(args, "snapshot_dir", "") else "gespo.log"
    _setup_logging(log_file)

    config_log_dir = getattr(args, "snapshot_dir", "") or ""
    RunConfigurer(args, log_dir=config_log_dir).print_summary()

    _save_effective_config(args)
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
            config = load_config(config_path)
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
    train_parser = subparsers.add_parser(
        "train", help="Run GESPO training",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    train_parser.add_argument("--config", "-c", default="", metavar="YAML",
        help="Path to YAML config file. CLI flags override YAML values.")
    TrainingConfig.populate_parser(train_parser)

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
    pretrain_parser.add_argument("--global-token-set", default="rmsd_e2e")
    pretrain_parser.add_argument("--torsion-mode", default="backbone", choices=["backbone", "backbone_chi"])

    # --- gespo config ---
    config_parser = subparsers.add_parser("config", help="Config utilities")
    config_sub = config_parser.add_subparsers(dest="config_cmd")
    template_parser = config_sub.add_parser("template", help="Generate a YAML config template")
    template_parser.add_argument("--mode", default="generative",
        choices=["generative", "path-sampling", "standard"],
        help="Training mode for the template")
    validate_parser = config_sub.add_parser("validate", help="Validate a config file")
    validate_parser.add_argument("config_file", help="Path to YAML config file")

    # --- gespo bootstrap ---
    subparsers.add_parser("bootstrap", help="Bootstrap utilities (run: python -m gespo.bootstrap_cli)")

    # --- Backward compat: if no subcommand, treat as 'train' ---
    # Check if the first positional arg is a known subcommand.
    # If not (e.g., user passed --pdb directly), insert 'train' as the subcommand.
    if len(sys.argv) > 1 and sys.argv[1] not in ("train", "pretrain", "config", "bootstrap", "-h", "--help"):
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
    elif args.command == "bootstrap":
        print("Run bootstrap utilities with: python -m gespo.bootstrap_cli --help")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
