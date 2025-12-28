"""CLI entrypoint (argument parsing and dispatch to training or utilities)."""
from ._imports import *
from .training import train
from .utils.config import RunConfigurer
from .utils.logging import configure_logging
from .config import TrainingConfig
import os
import copy
import argparse
import warnings
import logging

logger = logging.getLogger(__name__)

def main():
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

    def _make_experiment_args(base_args: argparse.Namespace, mode: str, label: str) -> argparse.Namespace:
        """
        Create a deep-copied args namespace for an auxiliary experiment
        (e.g., least-count or uniform starts) with output paths redirected
        into separate subfolders / files.
        """
        exp_args = copy.deepcopy(base_args)

        # Toggle start-mode flags for the experiment.
        if mode == "least":
            setattr(exp_args, "least_count_starts", True)
            setattr(exp_args, "uniform_starts", False)
        elif mode == "uniform":
            setattr(exp_args, "uniform_starts", True)
            setattr(exp_args, "least_count_starts", False)
        # mode == "primary": keep user-provided flags as-is

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
        "--run-extra-exp",
        dest="run_extra_exp",
        action="store_true",
        help=(
            "Run additional ablation experiments with least-counting and uniform "
            "discrete-state starts in separate output folders (checkpoint/snapshot/stats)."
        ),
    )
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
    if hasattr(args, 'gen_device'):
        gen_device_str = args.gen_device
        if gen_device_str.startswith("cuda:"):
            args.device = int(gen_device_str.split(":")[1])
        elif gen_device_str.isdigit():
            args.device = int(gen_device_str)
        else:
            args.device = 0  # default to 0 for cpu or other
    else:
        args.device = 0

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

    configurer = RunConfigurer(args, log_dir=args.config_log)
    configurer.print_summary()

    # Number of independent repetitions per experiment
    repeat = getattr(args, "repeat_exp", 3)
    try:
        repeat = int(repeat)
    except Exception:
        repeat = 3
    if repeat < 1:
        repeat = 1

    for rep_idx in range(repeat):
        rep_label = f"rep{rep_idx+1}"
        rep_note = f" (repeat {rep_idx+1}/{repeat})" if repeat > 1 else ""

        # Primary run for this repetition
        if repeat > 1:
            primary_args = _make_experiment_args(args, mode="primary", label=rep_label)
        else:
            primary_args = args
        primary_configurer = RunConfigurer(primary_args, log_dir=getattr(primary_args, "config_log", ""))
        primary_configurer.print_summary()
        logger.info("Starting training (primary run%s)...", rep_note)
        train(primary_args)
        logger.info("Primary training complete%s", rep_note)

        # Optional: run extra ablation experiments with modified start policies
        if getattr(args, "run_extra_exp", False):
            # Least-counting starts experiment
            least_label = f"least_count_{rep_label}" if repeat > 1 else "least_count"
            least_args = _make_experiment_args(args, mode="least", label=least_label)
            least_configurer = RunConfigurer(least_args, log_dir=getattr(least_args, "config_log", ""))
            least_configurer.print_summary()
            logger.info("Starting training (least-count experiment%s)...", rep_note)
            train(least_args)
            logger.info("Least-count experiment complete%s", rep_note)

            # Uniform random starts experiment
            uniform_label = f"uniform_{rep_label}" if repeat > 1 else "uniform"
            uniform_args = _make_experiment_args(args, mode="uniform", label=uniform_label)
            uniform_configurer = RunConfigurer(uniform_args, log_dir=getattr(uniform_args, "config_log", ""))
            uniform_configurer.print_summary()
            logger.info("Starting training (uniform-start experiment%s)...", rep_note)
            train(uniform_args)
            logger.info("Uniform-start experiment complete%s", rep_note)

if __name__ == "__main__":
    main()
