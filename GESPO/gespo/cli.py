"""CLI entrypoint (argument parsing and dispatch to training or utilities)."""
from ._imports import *
from .training import train
from .utils.config import RunConfigurer
from .config import TrainingConfig
import os
import copy
import pickle
import argparse
import sys
import atexit
import datetime as _dt
import warnings

def _fit_tica_from_dcd(base_dirs,
                       pdb_path: str,
                       out_path: str,
                       lagtime: int = 20,
                       dim: int = 2,
                       stride: int = 2) -> None:
    """
    Fit a TICA model (deeptime) from existing DCD trajectories and save it.
    """
    try:
        from deeptime.decomposition import TICA
    except ImportError as exc:
        raise RuntimeError(
            "deeptime is required to fit TICA models. "
            "Install it (e.g. `pip install deeptime`) or run the notebook-based workflow."
        ) from exc

    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]
    base_dirs = [os.path.abspath(d) for d in base_dirs]
    pdb_path = os.path.abspath(pdb_path)
    out_path = os.path.abspath(out_path)

    print(f"[TICA] Fitting TICA from DCDs under:")
    for d in base_dirs:
        print(f"  - {d}")
    print(f"[TICA] Using reference PDB: {pdb_path}")
    print(f"[TICA] Output model path: {out_path}")
    print(f"[TICA] Settings: lagtime={lagtime} frames, dim={dim}, stride={stride}")

    ref = md.load(pdb_path)
    dcd_files = []
    for base_dir in base_dirs:
        for root, _, names in os.walk(base_dir):
            for name in names:
                if name.endswith(".dcd"):
                    dcd_files.append(os.path.join(root, name))
    dcd_files = sorted(set(dcd_files))
    if not dcd_files:
        raise RuntimeError(f"[TICA] No .dcd files found under: {', '.join(base_dirs)}")

    trajs = []
    for fpath in dcd_files:
        try:
            tr = md.load(fpath, top=pdb_path, stride=stride)
            if tr.n_frames == 0:
                continue
            tr = tr.superpose(ref)
            trajs.append(tr)
        except Exception as exc:
            print(f"[TICA] Skipping {fpath}: {exc}")
            continue

    if not trajs:
        raise RuntimeError(f"[TICA] No usable trajectories loaded from: {', '.join(base_dirs)}")

    ca_list = []
    for tr in trajs:
        ca = tr.atom_slice(tr.topology.select("name CA"))
        X = ca.xyz.reshape(ca.n_frames, -1)  # nm, flattened
        ca_list.append(X.astype(np.float32))

    print(f"[TICA] Prepared {len(ca_list)} CA feature arrays.")
    tica = TICA(lagtime=lagtime, dim=dim)
    tica.fit(ca_list)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(tica, f)
    print(f"[TICA] Saved TICA model to {out_path}")

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
        
        def _close_log():
            try:
                _log_fh.flush()
                _log_fh.close()
            except: pass
            
        atexit.register(_close_log)
        print(f"[Log] Capturing stdout/stderr to {_log_path}")
    except Exception:
        pass

def main():
    print("[DEBUG cli.main] Entered main()")

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
    
    print("[DEBUG cli.main] Parser created, populating arguments...")
    # Populate arguments from centralized config
    TrainingConfig.populate_parser(parser)
    print("[DEBUG cli.main] Arguments populated")
    
    # Add CLI-specific arguments not in TrainingConfig (utilities)
    parser.add_argument(
        "--fit-tica-from-dcd",
        nargs="+",
        default=[],
        metavar="DIR",
        help="Run offline TICA fit from DCDs and exit.",
    )
    parser.add_argument(
        "--tica-out",
        default="",
        help="Output path for fitted TICA model (default: tica_model.pkl).",
    )
    parser.add_argument(
        "--tica-stride",
        type=int,
        default=2,
        help="Stride for loading frames from DCDs during offline TICA fitting.",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Path to log file.",
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

    print("[DEBUG cli.main] Parsing arguments...")
    args = parser.parse_args()
    print("[DEBUG cli.main] Arguments parsed successfully")
    
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

    _setup_logging(args.log_file)

    if args.suppress_warnings:
        warnings.filterwarnings("ignore")

    print("[DEBUG cli.main] Checking for offline TICA fitting...")
    # Offline TICA fitting
    if args.fit_tica_from_dcd:
        print("[DEBUG cli.main] Running offline TICA fitting")
        tica_out = args.tica_out or "tica_model.pkl"
        _fit_tica_from_dcd(
            base_dirs=args.fit_tica_from_dcd,
            pdb_path=args.pdb,
            out_path=tica_out,
            lagtime=args.tica_lagtime,
            dim=args.tica_dim,
            stride=getattr(args, "tica_stride", 2),
        )
        return

    # Offline TorsionARPolicy pretraining
    if getattr(args, "pretrain_torsion_ar", ""):
        from .training.pretrain_torsion_ar import pretrain_torsion_ar
        source_path = args.pretrain_torsion_ar
        out_path = args.torsion_ar_out
        epochs = int(getattr(args, "torsion_ar_epochs", 10) or 10)
        batch_size = int(getattr(args, "torsion_ar_batch_size", 64) or 64)
        print(f"[DEBUG cli.main] Running TorsionAR pretraining on {source_path}")
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

    print("[DEBUG cli.main] Creating RunConfigurer...")
    configurer = RunConfigurer(args, log_dir=args.config_log)
    print("[DEBUG cli.main] Printing config summary for base configuration...")
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
        print(f"[DEBUG cli.main] Creating RunConfigurer for primary run{rep_note}...")
        primary_configurer = RunConfigurer(primary_args, log_dir=getattr(primary_args, "config_log", ""))
        print(f"[DEBUG cli.main] Printing config summary for primary run{rep_note}...")
        primary_configurer.print_summary()
        print(f"[DEBUG cli.main] Starting training (primary run{rep_note})...")
        train(primary_args)
        print(f"[DEBUG cli.main] Primary training complete{rep_note}")

        # Optional: run extra ablation experiments with modified start policies
        if getattr(args, "run_extra_exp", False):
            # Least-counting starts experiment
            least_label = f"least_count_{rep_label}" if repeat > 1 else "least_count"
            least_args = _make_experiment_args(args, mode="least", label=least_label)
            print(f"[DEBUG cli.main] Creating RunConfigurer for least-count experiment{rep_note}...")
            least_configurer = RunConfigurer(least_args, log_dir=getattr(least_args, "config_log", ""))
            print(f"[DEBUG cli.main] Printing config summary for least-count experiment{rep_note}...")
            least_configurer.print_summary()
            print(f"[DEBUG cli.main] Starting training (least-count experiment{rep_note})...")
            train(least_args)
            print(f"[DEBUG cli.main] Least-count experiment complete{rep_note}")

            # Uniform random starts experiment
            uniform_label = f"uniform_{rep_label}" if repeat > 1 else "uniform"
            uniform_args = _make_experiment_args(args, mode="uniform", label=uniform_label)
            print(f"[DEBUG cli.main] Creating RunConfigurer for uniform-start experiment{rep_note}...")
            uniform_configurer = RunConfigurer(uniform_args, log_dir=getattr(uniform_args, "config_log", ""))
            print(f"[DEBUG cli.main] Printing config summary for uniform-start experiment{rep_note}...")
            uniform_configurer.print_summary()
            print(f"[DEBUG cli.main] Starting training (uniform-start experiment{rep_note})...")
            train(uniform_args)
            print(f"[DEBUG cli.main] Uniform-start experiment complete{rep_note}")

if __name__ == "__main__":
    print("[DEBUG cli.py] Executing as __main__")
    main()
