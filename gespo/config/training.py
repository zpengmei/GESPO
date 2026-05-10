"""Centralized GESPO training configuration.

After decomposition, TrainingConfig holds ~50 core/RL/policy/projection fields
and delegates to 8 sub-config dataclasses via ``__getattr__``/``__setattr__``.
"""
from dataclasses import dataclass, field, fields
import argparse
from typing import ClassVar, Optional
import numpy as np

from .committor import CommittorConfig
from .geom2vec import Geom2VecConfig
from .io import IOConfig
from .md import MDConfig
from .msm_ig import MSMIGConfig
from .path_sampling import PathSamplingConfig
from .steering import SteeringConfig
from .vamp_bin import VampBinConfig

# Constants for fixed RL parameters.
ENT_COEF_DEFAULT = 1e-3
CLIP_EPS_DEFAULT = 0.1
NORM_ADV_DEFAULT = True
KL_COEF_DEFAULT = 0.04
GRAD_CLIP_DEFAULT = 1.0


@dataclass
class TrainingConfig:
    """Centralized configuration for GESPO training.

    Fields are organized into sub-config dataclasses for cohesive groups
    (MD, Geom2Vec, I/O, committor, steering, path-sampling, VAMP-bin,
    MSM-IG).  Flat attribute access (``config.temperature``) is delegated
    transparently via ``__getattr__`` / ``__setattr__``.
    """

    # ── Core ────────────────────────────────────────────────────────────
    grp_size: int
    episodes: int
    platform: str
    gen_device: str
    pdb: str
    pdb_a: str = ""
    pdb_b: str = ""

    # ── Sub-configs ─────────────────────────────────────────────────────
    path: PathSamplingConfig = field(default_factory=PathSamplingConfig)
    committor: CommittorConfig = field(default_factory=CommittorConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    vamp_bin: VampBinConfig = field(default_factory=VampBinConfig)
    msm_ig: MSMIGConfig = field(default_factory=MSMIGConfig)
    geom2vec: Geom2VecConfig = field(default_factory=Geom2VecConfig)
    md: MDConfig = field(default_factory=MDConfig)
    io: IOConfig = field(default_factory=IOConfig)

    # ── Loop selection ──────────────────────────────────────────────────
    generative_loop: bool = False
    toy_phi_psi_loop: bool = False
    toy_phi_psi_dga_loop: bool = False

    # ── RL parameters (PPO / GRPO) ──────────────────────────────────────
    lr: float = 1e-4
    grpo_iters: int = 1
    sync_interval: int = 5
    ent_coef: float = ENT_COEF_DEFAULT
    clip_eps: float = CLIP_EPS_DEFAULT
    kl_coef: float = KL_COEF_DEFAULT
    norm_adv: bool = NORM_ADV_DEFAULT
    start_cluster_entropy_coef: float = -1.0
    off_action_penalty_coef: float = 0.0
    loo_vamp_reward_weight: float = 1.0
    grad_clip: float = GRAD_CLIP_DEFAULT

    # ── Policy ──────────────────────────────────────────────────────────
    policy_type: str = "mlp"
    policy_conditioning_mode: str = "hybrid"
    use_bias: bool = False
    num_mixtures: int = 3
    discrete_state_starts: bool = False
    use_boundary_replay: bool = True
    reset_policy_on_projector_refresh: bool = True

    # ── Retrieval ───────────────────────────────────────────────────────
    retrieval_candidates: int = 2048
    retrieval_topk: int = 32
    retrieval_max_resamples: int = 3
    retrieval_cache_max_frames: int = 0
    retrieval_use_global_metrics: bool = True
    retrieval_global_weight: float = 1.0
    global_rmsd_range: tuple = None
    global_e2e_range: tuple = None
    global_rg_range: tuple = None
    global_token_ranges: dict = None

    # ── Projection & scoring ────────────────────────────────────────────
    projection: str = "geom2vec_vamp"
    pca_bins: int = 50
    pca_completion_threshold: float = 0.8
    pca_refresh_interval: int = 1
    max_pca_train: int = 0

    # ── Anchoring ───────────────────────────────────────────────────────
    anchoring_round: bool = True
    anchoring_steps: int = 0  # 0 → use n_md_steps
    anchoring_num_trajs: int = 0
    anchoring_pdb_only: bool = False
    anchoring_write_full_dcd: Optional[bool] = None

    # ── Toy-loop controls ───────────────────────────────────────────────
    toy_lag: int = 1
    toy_start_kappa_floor: float = 1e-3
    toy_start_log_kappa_init: float = 0.0
    toy_start_train_kappa: bool = True
    toy_start_kappa_max: float = 0.0

    # ── Multi-GPU ───────────────────────────────────────────────────────
    enable_multi_gpu: bool = False

    # ── Torsion AR ─────────────────────────────────────────────────────
    torsion_ar_checkpoint: str = ""
    torsion_ar_mode: str = "backbone"
    torsion_ar_global_token_set: str = ""
    pretrain_torsion_ar: str = ""
    torsion_ar_epochs: int = 10
    torsion_ar_batch_size: int = 64
    torsion_ar_stride: int = 1

    # ── LOO-VAMP / ensemble ────────────────────────────────────────────
    loo_vamp_exploration_coef: float = 1.0
    loo_vamp_use_ucb: bool = True
    disable_vamp_ensemble: bool = False
    geom2vec_vamp_n_models: int = 3
    tica_start_transition_weight: float = 1.0

    # ── Experimental ───────────────────────────────────────────────────
    use_muon: bool = False
    old_init: str = "copy"
    old_perturb_scale: float = 1e-3
    least_count_starts: bool = False
    uniform_starts: bool = False

    # ── Sub-config attribute names for __getattr__ delegation ───────────
    _SUB_CONFIGS = (
        "committor", "path", "steering", "vamp_bin", "msm_ig",
        "geom2vec", "md", "io",
    )

    # ── CLI metadata for auto-generated argparse ────────────────────────
    CLI_META: ClassVar[dict] = {
        # Core
        "grp_size": {"help": "Number of parallel walkers (batch size).", "default_override": 16},
        "episodes": {"help": "Total RL training episodes.", "default_override": 20},
        "platform": {"help": "Simulation platform (CUDA/CPU).", "default_override": "CUDA"},
        "gen_device": {"help": "Device for generator/policy.", "default_override": "cuda:0"},
        "pdb": {"help": "Endpoint A PDB file (topology + initial coords).", "default_override": "chignolin.pdb", "aliases": ["--pdb-a"]},
        "pdb_a": {"skip": True},
        "pdb_b": "Endpoint B PDB file (path sampling target; requires --path-sampling).",
        # Loop selection
        "generative_loop": {
            "help": "Use torsion autoregressive generative loop.",
            "flag_only": True,
        },
        "toy_phi_psi_loop": {
            "help": "Run lightweight toy phi/psi start distribution loop (e.g. Ala2).",
            "flag_only": True,
        },
        "toy_phi_psi_dga_loop": {
            "help": "Run toy phi/psi loop with DGA committor rewards.",
            "flag_only": True,
        },
        # RL
        "lr": "Learning rate.",
        "grpo_iters": "GRPO inner updates per epoch.",
        "sync_interval": "Epochs between old-policy sync.",
        "ent_coef": "Entropy coefficient.",
        "clip_eps": {"skip": True},  # internal
        "kl_coef": "KL divergence coefficient.",
        "norm_adv": {"skip": True},  # internal
        "start_cluster_entropy_coef": (
            "Start-cluster entropy bonus coefficient (-1=tie to ent-coef, 0=disabled)."
        ),
        "off_action_penalty_coef": (
            "Discount for post-reset CV mismatch: reward *= exp(-coef*mae) (0=disabled)."
        ),
        "loo_vamp_reward_weight": "Weight for ΔLOO-VAMP2 reward (0=disabled).",
        "grad_clip": "Gradient clipping threshold.",
        # Policy
        "policy_type": {
            "help": "Policy architecture.",
            "choices": ["mlp", "residual"],
        },
        "policy_conditioning_mode": {
            "help": "Feature mode for policy conditioning.",
            "choices": ["torsion", "pairwise", "hybrid", "ca", "ca_torsion"],
        },
        "use_bias": {"help": "Enable learnable bias (hill) output.", "flag_only": True},
        "num_mixtures": "Number of Gaussian mixtures.",
        "discrete_state_starts": {
            "help": "Use categorical start policy over VAMP states.",
            "no_help": "Use continuous CV start policy.",
        },
        "use_boundary_replay": {
            "help": "Use entropy-based boundary frames for replay starts (default).",
            "flag": "--boundary-replay",
            "no_flag": "--random-replay",
            "no_help": "Use uniform random frame within each discrete state.",
        },
        "reset_policy_on_projector_refresh": {
            "help": "Reinitialize start policy after projector refresh (default).",
            "no_help": "Keep start policy across projector refreshes.",
        },
        # Retrieval
        "retrieval_candidates": {
            "help": "Candidate pool size for retrieval-based starts (0=all cached frames).",
            "default_override": 4096,
        },
        "retrieval_topk": "Top-K nearest candidates for retrieval sampling.",
        "retrieval_max_resamples": "Max torsion resamples per walker for out-of-support.",
        "retrieval_cache_max_frames": "Max frames in retrieval cache (0=unlimited).",
        "retrieval_use_global_metrics": {
            "help": "Use RMSD/E2E global metrics in retrieval distance.",
            "flag": "--retrieval-global-metrics",
            "no_flag": "--no-retrieval-global-metrics",
            "no_help": "Use φ/ψ only for retrieval distance.",
        },
        "retrieval_global_weight": "Weight for RMSD/E2E metrics in retrieval embedding (0=disabled).",
        "global_rmsd_range": {
            "help": "Conditional sampling range for RMSD global token (nm): MIN MAX.",
            "nargs": 2, "metavar": ("MIN", "MAX"), "type_override": float,
        },
        "global_e2e_range": {
            "help": "Conditional sampling range for E2E distance global token (nm): MIN MAX.",
            "nargs": 2, "metavar": ("MIN", "MAX"), "type_override": float,
        },
        "global_rg_range": {
            "help": "Conditional sampling range for Rg global token (nm): MIN MAX.",
            "nargs": 2, "metavar": ("MIN", "MAX"), "type_override": float,
        },
        "global_token_ranges": {
            "help": "Conditional range for named global token: NAME MIN MAX (repeatable).",
            "nargs": 3, "action": "append", "metavar": ("NAME", "MIN", "MAX"),
            "flag": "--global-token-range",
        },
        # Projection
        "projection": {"help": "Projection method.", "choices": ["geom2vec_vamp"]},
        "pca_bins": "Grid bins for novelty.",
        "pca_completion_threshold": "Threshold for PCA readiness.",
        "pca_refresh_interval": "Epochs between PCA/TICA updates.",
        "max_pca_train": "Max frames for PCA training (0=all).",
        # Anchoring
        "anchoring_round": {
            "help": "Run initial anchoring round before training.",
            "no_help": "Disable anchoring round.",
        },
        "anchoring_steps": {
            "help": "MD steps per anchoring trajectory (0=use n_md_steps).",
            "default_override": 100000,
        },
        "anchoring_num_trajs": "Max anchoring trajectories (0=one per walker).",
        "anchoring_pdb_only": {
            "help": "Anchor from PDB(s) only, no generator-based starts.",
            "flag_only": True,
        },
        "anchoring_write_full_dcd": {
            "help": "Write full-system DCDs during anchoring.",
            "no_help": "Write protein-only DCDs during anchoring.",
        },
        # Toy
        "toy_lag": "Toy loop lag (recorded frames) for VAMP/MSM-IG pairs.",
        "toy_start_kappa_floor": "Min VonMises concentration added to softplus.",
        "toy_start_log_kappa_init": "Initial log_kappa for VonMisesStartPolicy.",
        "toy_start_train_kappa": {
            "help": "Allow VonMises kappa to be trained (default).",
            "no_flag": "--toy-start-no-train-kappa",
            "no_help": "Freeze VonMises kappa at init.",
        },
        "toy_start_kappa_max": "Clamp VonMises kappa to at most this (0=disabled).",
        # Multi-GPU
        "enable_multi_gpu": {
            "help": "Enable multi-GPU parallelization.",
            "flag_only": True,
        },
        # Torsion AR
        "torsion_ar_checkpoint": {
            "help": "Pretrained TorsionAR policy checkpoint (.pt).",
            "aliases": ["--torsion-ar-out"],
        },
        "torsion_ar_mode": {
            "help": "Torsion channels for TorsionAR start policy.",
            "choices": ["backbone", "backbone+sidechain"],
        },
        "torsion_ar_global_token_set": (
            "Legacy compat; global token layout is taken from the checkpoint."
        ),
        "pretrain_torsion_ar": (
            "Pretrain TorsionAR from trajectory directories (comma-separated) or .npz."
        ),
        "torsion_ar_epochs": "(Pretrain only) epochs.",
        "torsion_ar_batch_size": "(Pretrain only) batch size.",
        "torsion_ar_stride": "(Pretrain only) frame stride.",
        # LOO-VAMP / ensemble
        "loo_vamp_exploration_coef": "UCB exploration coefficient λ: score = μ + λ·σ.",
        "loo_vamp_use_ucb": {
            "help": "Use UCB uncertainty term (default).",
            "no_flag": "--no-ucb",
            "no_help": "Disable UCB uncertainty term (use μ only).",
        },
        "disable_vamp_ensemble": {
            "help": "Disable VAMP ensemble (forces geom2vec_vamp_n_models=1).",
            "flag": "--no-vamp-ensemble",
            "flag_only": True,
        },
        "geom2vec_vamp_n_models": (
            "Number of independent geom2vec VAMP lobe fits (ensemble size)."
        ),
        "tica_start_transition_weight": (
            "Weight for start-to-end displacement bonus in projection space."
        ),
        # Experimental
        "use_muon": {
            "help": "Use Muon optimizer for 2D parameters.",
            "flag_only": True,
        },
        "old_init": {
            "help": "Initialization for old policy.",
            "choices": ["copy", "perturb", "normal", "random"],
        },
        "old_perturb_scale": "Perturbation scale if old-init is 'perturb'.",
        "least_count_starts": {
            "help": "Always launch from least-populated state (RL start policy disabled).",
            "no_help": "Use normal start policy (default).",
        },
        "uniform_starts": {
            "help": "Force uniform probability distribution over all hard states.",
            "flag_only": True,
        },
    }

    # ── Attribute delegation ────────────────────────────────────────────

    def __getattr__(self, name: str):
        """Delegate attribute lookups to sub-config dataclasses for flat access.

        This allows ``config.temperature`` to work even though the field
        lives on ``config.md``.
        """
        for attr in self._SUB_CONFIGS:
            try:
                sub = object.__getattribute__(self, attr)
            except AttributeError:
                continue
            if hasattr(type(sub), name):
                return getattr(sub, name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value):
        """Route attribute sets to sub-configs when the field belongs there."""
        for attr in self._SUB_CONFIGS:
            try:
                sub = object.__getattribute__(self, attr)
            except AttributeError:
                continue
            if hasattr(type(sub), name):
                setattr(sub, name, value)
                return
        object.__setattr__(self, name, value)

    # ── CLI population ──────────────────────────────────────────────────

    @classmethod
    def populate_parser(cls, parser: argparse.ArgumentParser):
        """Populate the argparse parser with arguments mapping to this config."""
        from .cli_gen import populate_from_dataclass

        # Auto-generate from all config dataclasses.
        populate_from_dataclass(parser, cls, group_name="Core & RL")
        populate_from_dataclass(parser, PathSamplingConfig, group_name="Path Sampling")
        populate_from_dataclass(parser, MDConfig, group_name="Sampling & MD")
        populate_from_dataclass(parser, SteeringConfig, group_name="Steering")
        populate_from_dataclass(parser, CommittorConfig, group_name="Committor Rewards")
        populate_from_dataclass(parser, VampBinConfig, group_name="VAMP-Bin Bonus")
        populate_from_dataclass(parser, MSMIGConfig, group_name="MSM Information Gain")
        populate_from_dataclass(parser, Geom2VecConfig, group_name="Geom2Vec")
        populate_from_dataclass(parser, IOConfig, group_name="I/O & Checkpointing")

        # All args are now auto-generated from dataclass fields + CLI_META.

    # ── Factory methods ─────────────────────────────────────────────────

    @classmethod
    def from_args(cls, args) -> "TrainingConfig":
        """Create config from argparse namespace."""
        def get(name, default):
            return getattr(args, name, default)

        # Derive OpenMM device index from the generator device string.
        gen_device_str = str(get("gen_device", "cuda:0"))
        if gen_device_str.startswith("cuda:"):
            device_idx = int(gen_device_str.split(":")[1])
        elif gen_device_str.isdigit():
            device_idx = int(gen_device_str)
        else:
            device_idx = 0

        # Solvent mode normalization.
        # --implicit-solvent takes precedence: MDConfig defaults explicit_solvent=True,
        # so passing --implicit-solvent alone would trigger both flags without this guard.
        implicit_flag = bool(get("implicit_solvent", False))
        explicit_flag = bool(get("explicit_solvent", False))
        if implicit_flag:
            explicit_solvent = False
        else:
            explicit_solvent = bool(explicit_flag or True)

        # Build sub-configs from args.
        sub_configs = {}
        consumed_names: set = set()
        for attr_name, sub_cls in [
            ("committor", CommittorConfig),
            ("path", PathSamplingConfig),
            ("steering", SteeringConfig),
            ("vamp_bin", VampBinConfig),
            ("msm_ig", MSMIGConfig),
            ("geom2vec", Geom2VecConfig),
            ("md", MDConfig),
            ("io", IOConfig),
        ]:
            sub_field_names = {f.name for f in fields(sub_cls) if not f.name.startswith("_")}
            # Exclude ClassVar-like entries (CLI_META etc.)
            sub_field_names.discard("CLI_META")
            sub_cfg = {name: getattr(args, name) for name in sub_field_names if hasattr(args, name)}
            sub_configs[attr_name] = sub_cls(**sub_cfg)
            consumed_names.update(sub_field_names)

        # Apply post-build normalization to MD sub-config.
        md_cfg: MDConfig = sub_configs["md"]
        md_cfg.implicit_solvent = implicit_flag
        md_cfg.explicit_solvent = explicit_solvent
        md_cfg.device = int(device_idx)
        try:
            md_cfg.record_stride = max(1, int(md_cfg.record_stride or 10))
        except Exception:
            md_cfg.record_stride = 10

        # Build TrainingConfig's own fields.
        field_names = {f.name for f in fields(cls)}
        cfg = {name: getattr(args, name) for name in field_names if hasattr(args, name)}
        # Remove sub-config field names that were consumed above.
        for name in consumed_names:
            cfg.pop(name, None)
        cfg.update(sub_configs)

        # Normalize endpoint aliases.
        try:
            pdb_a = str(cfg.get("pdb_a", "") or cfg.get("pdb", "") or "")
        except Exception:
            pdb_a = str(cfg.get("pdb", "") or "")
        cfg["pdb_a"] = pdb_a
        cfg["pdb"] = pdb_a

        # Normalize global-range args to tuples (argparse gives lists).
        for k in ("global_rmsd_range", "global_e2e_range", "global_rg_range"):
            v = cfg.get(k, None)
            if v is not None and not isinstance(v, tuple):
                try:
                    cfg[k] = tuple(v)
                except Exception:
                    cfg[k] = None

        # Normalize per-token ranges to a dict[str, tuple[float,float]].
        gtr = cfg.get("global_token_ranges", None)
        if gtr:
            out = {}
            for item in gtr:
                try:
                    name, vmin, vmax = item
                except Exception:
                    continue
                try:
                    key = str(name).strip().lower()
                    if not key:
                        continue
                    mn = float(vmin)
                    mx = float(vmax)
                    if not (np.isfinite(mn) and np.isfinite(mx)):
                        continue
                    if mx < mn:
                        mn, mx = mx, mn
                    out[key] = (mn, mx)
                except Exception:
                    continue
            cfg["global_token_ranges"] = out if out else None
        else:
            cfg["global_token_ranges"] = None

        return cls(**cfg)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        """Create config from a flat dict (e.g., from YAML or programmatic use).

        Works like ``from_args`` but accepts a plain dict instead of an
        ``argparse.Namespace``.
        """
        import argparse as _ap
        ns = _ap.Namespace(**d)
        return cls.from_args(ns)

    def to_dict(self) -> dict:
        """Serialize to a flat dict for YAML/JSON output.

        Sub-configs are flattened so that ``from_dict(to_dict())`` roundtrips.
        """
        from dataclasses import fields as _fields, asdict as _asdict
        flat: dict = {}
        for f in _fields(self):
            if f.name.startswith("_"):
                continue
            val = getattr(self, f.name)
            if f.name in self._SUB_CONFIGS:
                for sf in _fields(val):
                    if sf.name.startswith("_") or sf.name == "CLI_META":
                        continue
                    flat[sf.name] = getattr(val, sf.name)
            else:
                flat[f.name] = val
        return flat
