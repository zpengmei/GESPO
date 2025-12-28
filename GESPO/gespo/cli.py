"""CLI entrypoint (argument parsing and dispatch to training or utilities)."""
from .training import train
from .utils.config import RunConfigurer
from .utils.logging import configure_logging
from .config import TrainingConfig
import argparse
import copy
import logging
import os
import warnings

logger = logging.getLogger(__name__)

def _suffix_dir(path: str, suffix: str) -> str:
    """Append a subdirectory suffix if path is non-empty."""
    if not path:
        return path
    return os.path.join(path, suffix)

def _suffix_file(path: str, suffix: str) -> str:
    """Append a suffix before file extension if path is non-empty."""
    if not path:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_{suffix}{ext}"

def _make_experiment_args(base_args: argparse.Namespace, label: str) -> argparse.Namespace:
    """
    Create a deep-copied args namespace for an auxiliary experiment
    with output paths redirected into separate subfolders / files.
    """
    exp_args = copy.deepcopy(base_args)

    # Redirect checkpoint/snapshot/stats outputs if the user provided paths.
    ckpt_dir = getattr(base_args, "checkpoint_dir", "")
    snap_dir = getattr(base_args, "snapshot_dir", "")
    stats_log = getattr(base_args, "stats_log", "")
    config_log = getattr(base_args, "config_log", "")

    if ckpt_dir:
        setattr(exp_args, "checkpoint_dir", _suffix_dir(ckpt_dir, label))
    if snap_dir:
        setattr(exp_args, "snapshot_dir", _suffix_dir(snap_dir, label))
    if stats_log:
        setattr(exp_args, "stats_log", _suffix_file(stats_log, label))
    if config_log:
        setattr(exp_args, "config_log", _suffix_dir(config_log, label))

    return exp_args


def _extract_openmm_device(gen_device: str | None) -> int:
    if not gen_device:
        return 0
    if gen_device.startswith("cuda:"):
        return int(gen_device.split(":")[1])
    if gen_device.isdigit():
        return int(gen_device)
    return 0


def _normalize_repeat(value: object, default: int = 3) -> int:
    try:
        repeat = int(value)
    except Exception:
        repeat = default
    return max(1, repeat)


def _run_experiment(args: argparse.Namespace, label: str, rep_note: str) -> None:
    configurer = RunConfigurer(args, log_dir=getattr(args, "config_log", ""))
    configurer.print_summary()
    configurer.dump()
    logger.info("Starting training (%s run%s)...", label, rep_note)
    train(args)
    logger.info("%s run complete%s", label.capitalize(), rep_note)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "GESPO: RL-based enhanced sampling with geom2vec VAMP discrete-state starts.\n"
            "Projection backend: geom2vec_vamp."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    # Populate arguments from centralized config
    TrainingConfig.populate_parser(parser)
    
    # Add CLI-specific arguments not in TrainingConfig (utilities)
    parser.add_argument(
        "--log-file",
        default="",
        help="Path to log file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., DEBUG, INFO, WARNING).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    parser.add_argument(
        "--config-log",
        default="",
        help="Directory to store JSON dumps of configuration.",
    )
    parser.add_argument(
        "--repeat-exp",
        dest="repeat_exp",
        type=int,
        default=3,
        help="Repeat each configured experiment N times (independent runs).",
    )
    parser.add_argument("--suppress-warnings", action="store_true", help="Suppress warnings.")
    parser.add_argument(
        "--pretrain-torsion-ar",
        default="",
        metavar="PATH",
        help=(
            "Path to .npz file OR directory of .dcd files (or a comma-separated list of directories) "
            "containing torsion data for TorsionARPolicy pretraining. If set, runs pretraining and exits."
        ),
    )
    parser.add_argument(
        "--torsion-ar-out",
        default="torsion_ar_pretrained.pt",
        help="Output path for pretrained TorsionARPolicy weights.",
    )
    parser.add_argument(
        "--torsion-ar-epochs",
        type=int,
        default=10,
        help="Number of epochs for TorsionARPolicy pretraining.",
    )
    parser.add_argument(
        "--torsion-ar-batch-size",
        type=int,
        default=64,
        help="Batch size for TorsionARPolicy pretraining.",
    )
    parser.add_argument(
        "--torsion-ar-stride",
        type=int,
        default=1,
        help="Stride for sampling frames from DCDs during TorsionARPolicy pretraining.",
    )
    parser.add_argument(
        "--torsion-ar-global-token-set",
        dest="torsion_ar_global_token_set",
        default="rmsd_e2e",
        help=(
            "Which global tokens to compute/store when pretraining from DCDs. "
            "Supported: 'rmsd_e2e' (default) or 'adk' (RMSD + hinge angles + two distances)."
        ),
    )

    args = parser.parse_args()

    # Extract device index from gen_device for OpenMM platform
    args.device = _extract_openmm_device(getattr(args, "gen_device", None))

    log_level = args.log_level
    if args.verbose:
        log_level = "DEBUG"
    configure_logging(log_level=log_level, log_file=args.log_file)
    logger.debug("Logging configured with level=%s file=%s", log_level, args.log_file)

    if args.suppress_warnings:
        warnings.filterwarnings("ignore")

    # Offline TorsionARPolicy pretraining
    if getattr(args, "pretrain_torsion_ar", ""):
        from .training.scripts.pretrain_torsion_ar import pretrain_torsion_ar
        source_path = args.pretrain_torsion_ar
        out_path = args.torsion_ar_out
        epochs = int(getattr(args, "torsion_ar_epochs", 10) or 10)
        batch_size = int(getattr(args, "torsion_ar_batch_size", 64) or 64)
        logger.info("Running TorsionAR pretraining on %s", source_path)
        pretrain_torsion_ar(
            source_path=source_path,
            out_path=out_path,
            epochs=epochs,
            batch_size=batch_size,
            pdb_path=getattr(args, "pdb", ""),
            stride=getattr(args, "torsion_ar_stride", 1),
            global_token_set=getattr(args, "torsion_ar_global_token_set", "rmsd_e2e"),
        )
        return

    # Number of independent repetitions per experiment
    repeat = _normalize_repeat(getattr(args, "repeat_exp", 3))

    for rep_idx in range(repeat):
        rep_label = f"rep{rep_idx+1}" if repeat > 1 else ""
        rep_note = f" (repeat {rep_idx+1}/{repeat})" if repeat > 1 else ""

        # Primary run for this repetition
        primary_args = _make_experiment_args(args, label=rep_label) if rep_label else args
        _run_experiment(primary_args, "primary", rep_note)

if __name__ == "__main__":
    main()
