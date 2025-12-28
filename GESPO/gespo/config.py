from __future__ import annotations

from dataclasses import dataclass, asdict
import argparse
from typing import ClassVar, Dict, Optional, Tuple

# Constants for fixed parameters (formerly CLI args)
ENT_COEF_DEFAULT = 1e-3
CLIP_EPS_DEFAULT = 0.1
NORM_ADV_DEFAULT = True
KL_COEF_DEFAULT = 0.04
GRAD_CLIP_DEFAULT = 1.0
LOSS_PREC_DEFAULT = 3


def _validate_solvent_flags(implicit_flag: bool, explicit_flag: bool) -> Tuple[bool, bool]:
    if implicit_flag and explicit_flag:
        raise ValueError("Cannot set both --implicit-solvent and --explicit-solvent.")
    explicit_solvent = explicit_flag or not implicit_flag
    return implicit_flag, explicit_solvent


def _resolve_explicit_replay(explicit_solvent: bool, explicit_replay_full_coords: bool) -> bool:
    if explicit_solvent:
        return explicit_replay_full_coords
    return False


def _resolve_vamp_ensemble(disable_vamp_ensemble: bool, n_models: int) -> int:
    n_models = int(n_models or 1)
    if n_models < 1:
        raise ValueError("--geom2vec-vamp-n-models must be >= 1")
    if disable_vamp_ensemble:
        return 1
    return n_models


def _validate_range(name: str, value: Optional[Tuple[float, float]]) -> None:
    if value is None:
        return
    if len(value) != 2:
        raise ValueError(f"{name} expects (min, max) values")
    if value[0] >= value[1]:
        raise ValueError(f"{name} expects min < max; got {value}")


@dataclass
class CoreConfig:
    grp_size: int
    episodes: int
    platform: str
    gen_device: str
    pdb: str
    generative_loop: bool = False
    cond_checkpoint: str = ""
    cond_data: str = ""
    enable_multi_gpu: bool = False

    @classmethod
    def populate_parser(cls, parser: argparse.ArgumentParser) -> None:
        g_core = parser.add_argument_group("Core Configuration")
        g_core.add_argument("--pdb", "--pdb-path", dest="pdb", default="chignolin.pdb", help="PDB file (topology + initial coords).")
        g_core.add_argument("--episodes", "--epochs", dest="episodes", type=int, default=20, help="Total RL training episodes.")
        g_core.add_argument("--grp_size", "--grp-size", dest="grp_size", type=int, default=16, help="Number of parallel walkers (batch size).")
        g_core.add_argument("--platform", default="CUDA", help="Simulation platform (CUDA/CPU).")
        g_core.add_argument("--gen_device", "--device", dest="gen_device", default="cuda:0", help="Device for the conditional generator.")
        g_core.add_argument("--cond-checkpoint", required=True, help="Path to diffusion model checkpoint.")
        g_core.add_argument("--cond-data", default="", help="Path to conditioning data (npz).")
        g_core.add_argument(
            "--generative-loop",
            dest="generative_loop",
            action="store_true",
            help="Use torsion autoregressive generative loop instead of the standard discrete-start loop.",
        )
        g_core.add_argument("--enable-multi-gpu", action="store_true", help="Enable multi-GPU parallelization (one walker per GPU or round-robin assignment).")


@dataclass
class RLConfig:
    lr: float = 1e-4
    grpo_iters: int = 1
    sync_interval: int = 5
    ent_coef: float = ENT_COEF_DEFAULT
    clip_eps: float = CLIP_EPS_DEFAULT
    kl_coef: float = KL_COEF_DEFAULT
    norm_adv: bool = NORM_ADV_DEFAULT
    start_cluster_entropy_coef: float = -1.0
    vamp_bin_entropy_coef: float = -1.0
    vamp_bin_norm_ema_beta: float = 0.9
    grad_clip: float = GRAD_CLIP_DEFAULT
    loss_prec: int = LOSS_PREC_DEFAULT

    @classmethod
    def populate_parser(cls, parser: argparse.ArgumentParser) -> None:
        g_rl = parser.add_argument_group("RL Parameters")
        g_rl.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
        g_rl.add_argument("--ent-coef", "--ent_coef", dest="ent_coef", type=float, default=ENT_COEF_DEFAULT, help="Entropy coefficient.")
        g_rl.add_argument("--kl-coef", "--kl_coef", dest="kl_coef", type=float, default=KL_COEF_DEFAULT, help="KL divergence coefficient.")
        g_rl.add_argument(
            "--start-cluster-entropy-coef",
            "--start_cluster_entropy_coef",
            dest="start_cluster_entropy_coef",
            type=float,
            default=-1.0,
            help="Coefficient for start-cluster entropy bonus based on retrieved cluster IDs (-1=tie to --ent-coef, 0=disabled).",
        )
        g_rl.add_argument(
            "--vamp-bin-entropy-coef",
            "--vamp_bin_entropy_coef",
            dest="vamp_bin_entropy_coef",
            type=float,
            default=-1.0,
            help=(
                "Coefficient for 2D VAMP-bin count exploration bonus in the first two VAMP coordinates "
                "(history-aware mean 1/sqrt(count+1), normalized to the LOO-VAMP scale when available) "
                "(-1=tie to --ent-coef, 0=disabled)."
            ),
        )
        g_rl.add_argument(
            "--vamp-bin-norm-ema-beta",
            "--vamp_bin_norm_ema_beta",
            dest="vamp_bin_norm_ema_beta",
            type=float,
            default=0.9,
            help="EMA beta for VAMP-bin bonus normalization (0=off, closer to 1.0=smoother).",
        )
        g_rl.add_argument("--grad-clip", "--grad_clip", dest="grad_clip", type=float, default=GRAD_CLIP_DEFAULT, help="Gradient clipping threshold.")
        g_rl.add_argument("--grpo-iters", type=int, default=1, help="GRPO inner updates per epoch.")
        g_rl.add_argument("--sync-interval", type=int, default=5, help="Epochs between old-policy sync.")


@dataclass
class PolicyConfig:
    policy_type: str = "mlp"
    use_muon: bool = False
    old_init: str = "copy"
    old_perturb_scale: float = 1e-3
    num_mixtures: int = 3
    use_bias: bool = False
    policy_conditioning_mode: str = "hybrid"
    discrete_state_starts: bool = False
    use_boundary_replay: bool = True
    reset_policy_on_projector_refresh: bool = True
    least_count_starts: bool = False
    uniform_starts: bool = False

    @classmethod
    def populate_parser(cls, parser: argparse.ArgumentParser) -> None:
        g_policy = parser.add_argument_group("Policy Architecture")
        g_policy.add_argument("--policy-type", default="mlp", choices=["mlp", "residual"], help="Policy architecture.")
        g_policy.add_argument("--use-muon", action="store_true", help="Use Muon optimizer for 2D parameters.")
        g_policy.add_argument("--old-init", default="copy", choices=["copy", "perturb", "normal", "random"], help="Initialization for old policy.")
        g_policy.add_argument("--old-perturb-scale", type=float, default=1e-3, help="Perturbation scale if old-init is 'perturb'.")
        g_policy.add_argument("--num-mixtures", type=int, default=3, help="Number of Gaussian mixtures.")
        g_policy.add_argument("--use-bias", action="store_true", help="Enable learnable bias (hill) output.")
        g_policy.add_argument(
            "--policy-conditioning-mode",
            default="hybrid",
            choices=["torsion", "pairwise", "hybrid", "ca", "ca_torsion"],
            help="Feature mode for policy conditioning.",
        )
        g_policy.add_argument(
            "--discrete-state-starts",
            dest="discrete_state_starts",
            action="store_true",
            help="Use a categorical start policy over VAMP states instead of continuous CV starts.",
        )
        g_policy.add_argument(
            "--no-discrete-state-starts",
            dest="discrete_state_starts",
            action="store_false",
            help="Disable categorical starts; use continuous CV start policy.",
        )
        g_policy.add_argument(
            "--least-count-starts",
            dest="least_count_starts",
            action="store_true",
            help="Experimental: always launch all walkers from the current least-populated hard state (RL start policy disabled).",
        )
        g_policy.add_argument(
            "--no-least-count-starts",
            dest="least_count_starts",
            action="store_false",
            help="Use the configured start policy (RL or continuous) instead of forcing least-count launches.",
        )
        g_policy.add_argument(
            "--uniform-starts",
            dest="uniform_starts",
            action="store_true",
            help="Experimental: force uniform probability distribution over all hard states for starts.",
        )
        g_policy.add_argument(
            "--reset-policy-on-projector-refresh",
            dest="reset_policy_on_projector_refresh",
            action="store_true",
            help="Reinitialize the start policy after each projector refresh to avoid learning on drifting state labels (default).",
        )
        g_policy.add_argument(
            "--no-reset-policy-on-projector-refresh",
            dest="reset_policy_on_projector_refresh",
            action="store_false",
            help="Keep the start policy parameters across projector refreshes.",
        )
        g_policy.add_argument(
            "--random-replay",
            dest="use_boundary_replay",
            action="store_false",
            help="Use uniform random frame within each discrete state for replay starts instead of entropy-based boundary frames.",
        )
        g_policy.add_argument(
            "--boundary-replay",
            dest="use_boundary_replay",
            action="store_true",
            help="Use entropy-based boundary frames within each discrete state for replay starts (default).",
        )
        parser.set_defaults(
            discrete_state_starts=False,
            least_count_starts=False,
            uniform_starts=False,
            use_boundary_replay=False,
            reset_policy_on_projector_refresh=True,
        )


@dataclass
class SamplingConfig:
    record_stride: int = 10
    sigma_min: float = 0.1
    retrieval_candidates: int = 2048
    retrieval_topk: int = 32
    retrieval_max_resamples: int = 3
    retrieval_cache_max_frames: int = 0
    retrieval_use_global_metrics: bool = True
    retrieval_global_weight: float = 1.0
    global_rmsd_range: Optional[Tuple[float, float]] = None
    global_e2e_range: Optional[Tuple[float, float]] = None

    @classmethod
    def populate_parser(cls, parser: argparse.ArgumentParser) -> None:
        g_sampling = parser.add_argument_group("Sampling Parameters")
        g_sampling.add_argument("--record-stride", type=int, default=10, help="MD steps between recorded frames.")
        g_sampling.add_argument("--sigma-min", type=float, default=0.1, help="Minimum sigma for policy.")
        g_sampling.add_argument("--retrieval-candidates", type=int, default=4096, help="Candidate pool size M for retrieval-based starts.")
        g_sampling.add_argument("--retrieval-topk", type=int, default=32, help="Top-K nearest candidates for retrieval sampling.")
        g_sampling.add_argument("--retrieval-max-resamples", type=int, default=3, help="Max torsion resamples per walker when retrieval query is out-of-support.")
        g_sampling.add_argument("--retrieval-cache-max-frames", type=int, default=0, help="Max frames to keep in the in-memory retrieval cache (0=use max_pca_train).")
        g_sampling.add_argument(
            "--no-retrieval-global-metrics",
            dest="retrieval_use_global_metrics",
            action="store_false",
            help="Disable using RMSD/E2E global metrics in retrieval distance (use φ/ψ only).",
        )
        g_sampling.add_argument(
            "--retrieval-global-weight",
            dest="retrieval_global_weight",
            type=float,
            default=1.0,
            help="Weight for normalized RMSD/E2E metrics appended to the retrieval embedding (0 disables).",
        )
        g_sampling.add_argument(
            "--global-rmsd-range",
            dest="global_rmsd_range",
            type=float,
            nargs=2,
            metavar=("MIN", "MAX"),
            default=None,
            help="Conditional sampling range for RMSD global token (nm): MIN MAX. Uses rejection sampling.",
        )
        g_sampling.add_argument(
            "--global-e2e-range",
            dest="global_e2e_range",
            type=float,
            nargs=2,
            metavar=("MIN", "MAX"),
            default=None,
            help="Conditional sampling range for end-to-end distance global token (nm): MIN MAX. Uses rejection sampling.",
        )


@dataclass
class MDConfig:
    forcefield_protein: str = "amber14-all.xml"
    forcefield_water: str = "amber14/tip3pfb.xml"
    forcefield_implicit: str = "implicit/gbn2.xml"
    temperature: float = 300.0
    friction: float = 1.0
    minimize_iters: int = 200
    heat_stages: int = 4
    heat_steps: int = 100
    equilibration_steps: int = 5000
    implicit_solvent: bool = False
    explicit_solvent: bool = True
    explicit_solvent_replay_full_coords: bool = True
    n_md_steps: int = 10000
    energy_threshold: float = 1e10
    max_resample_attempts: int = 5
    torsion_decimals: int = 3
    device: int = 0
    use_replay_resets: bool = True
    steer_replay_starts: bool = True
    steer_replay_verbose: bool = False
    steer_global_steps: int = 2000
    steer_global_updates: int = 25
    steer_torsion_steps: int = 1000
    steer_torsion_updates: int = 25
    steer_release_steps: int = 200
    steer_refine_minimize_iters: int = 0
    steer_k_rmsd: float = 2000.0
    steer_k_e2e: float = 2000.0
    steer_k_torsion: float = 200.0
    steer_global_lambda_during_torsion: float = 0.5
    anchoring_round: bool = True
    anchoring_steps: int = 0
    anchoring_num_trajs: int = 0
    anchoring_pdb_only: bool = False
    anchoring_source_dir: str = ""

    @classmethod
    def populate_parser(cls, parser: argparse.ArgumentParser) -> None:
        g_md = parser.add_argument_group("MD Configuration")
        g_md.add_argument("--forcefield-protein", default="amber14-all.xml", help="Protein force field XML file.")
        g_md.add_argument("--forcefield-water", default="amber14/tip3pfb.xml", help="Water model XML file for explicit solvent.")
        g_md.add_argument("--forcefield-implicit", default="implicit/gbn2.xml", help="Implicit solvent model XML file.")
        g_md.add_argument("--temperature", type=float, default=300.0, help="Simulation temperature (Kelvin).")
        g_md.add_argument("--friction", type=float, default=1.0, help="Langevin friction (1/ps).")
        g_md.add_argument("--minimize-iters", type=int, default=200, help="Energy minimization iterations.")
        g_md.add_argument("--heat-stages", type=int, default=4, help="Number of heating stages.")
        g_md.add_argument("--heat-steps", type=int, default=100, help="Steps per heating stage.")
        g_md.add_argument("--equilibration-steps", type=int, default=5000, help="Solvent equilibration steps before recording (explicit solvent).")
        g_md.add_argument("--torsion-decimals", type=int, default=3, help="Decimal places for torsion rounding.")
        g_md.add_argument("--implicit-solvent", action="store_true", help="Use implicit solvent.")
        g_md.add_argument(
            "--explicit-solvent",
            action="store_true",
            help=(
                "Enable explicit solvent (default unless --implicit-solvent). "
                "By default, uses full-coordinate replay (protein+solvent) for replay resets."
            ),
        )
        g_md.add_argument(
            "--explicit-solvent-replay-full-coords",
            dest="explicit_solvent_replay_full_coords",
            action="store_true",
            help=(
                "Explicit-solvent replay: keep a fixed solvated topology and restore full-system "
                "(protein+solvent) coordinates from saved DCD frames for replay resets. "
                "When enabled, resets will not rebuild the solvent box."
            ),
        )
        g_md.add_argument(
            "--no-explicit-solvent-replay-full-coords",
            dest="explicit_solvent_replay_full_coords",
            action="store_false",
            help="Disable explicit-solvent full-coordinate replay; rebuild the solvent box each reset.",
        )
        parser.set_defaults(explicit_solvent_replay_full_coords=True)
        g_md.add_argument("--n-md-steps", type=int, default=10000, help="MD steps per trajectory.")
        g_md.add_argument("--energy-threshold", type=float, default=1e10, help="Energy threshold for resampling.")
        g_md.add_argument("--max-resample-attempts", type=int, default=5, help="Max attempts to resample structures.")
        g_md.add_argument("--anchoring-round", action="store_true", help="Run an initial anchoring round before training.")
        g_md.add_argument("--no-anchoring-round", dest="anchoring_round", action="store_false", help="Disable anchoring round.")
        g_md.add_argument("--anchoring-steps", type=int, default=100000, help="MD steps per anchoring trajectory (walkers 1+). Should be >= lagtime * record_stride for MSM.")
        g_md.add_argument("--anchoring-num-trajs", type=int, default=0, help="Max number of trajectories to collect during anchoring (0=at most one per walker).")
        g_md.add_argument("--anchoring-pdb-only", action="store_true", help="Anchoring: launch all anchoring trajectories directly from PDB(s) without generator-based starts.")
        g_md.add_argument("--no-replay-resets", dest="use_replay_resets", action="store_false", help="Disable history-based replay resets; always use generator for training resets.")
        g_md.add_argument(
            "--steer-replay-starts",
            dest="steer_replay_starts",
            action="store_true",
            help=(
                "Enable targeted MD steering for replay resets (implicit solvent and explicit-solvent full-coordinate replay)."
            ),
        )
        g_md.add_argument(
            "--no-steer-replay-starts",
            dest="steer_replay_starts",
            action="store_false",
            help="Disable targeted MD steering for replay resets.",
        )
        g_md.add_argument(
            "--steer-replay-verbose",
            dest="steer_replay_verbose",
            action="store_true",
            help="Print per-reset steering diagnostics when replay steering is enabled.",
        )
        g_md.add_argument("--steer-global-steps", type=int, default=2000, help="Replay steering: steps for global (RMSD/E2E) stage.")
        g_md.add_argument("--steer-global-updates", type=int, default=25, help="Replay steering: number of schedule updates for global stage.")
        g_md.add_argument("--steer-torsion-steps", type=int, default=1000, help="Replay steering: steps for torsion stage.")
        g_md.add_argument("--steer-torsion-updates", type=int, default=25, help="Replay steering: number of schedule updates for torsion stage.")
        g_md.add_argument("--steer-release-steps", type=int, default=200, help="Replay steering: unbiased relaxation steps after steering.")
        g_md.add_argument(
            "--steer-refine-minimize-iters",
            type=int,
            default=0,
            help="Replay steering: optional energy minimization iterations under active steering (0 disables).",
        )
        g_md.add_argument("--steer-k-rmsd", type=float, default=5000.0, help="Replay steering: force constant for CA RMSD restraint (kJ/mol/nm^2).")
        g_md.add_argument("--steer-k-e2e", type=float, default=5000.0, help="Replay steering: force constant for end-to-end distance restraint (kJ/mol/nm^2).")
        g_md.add_argument("--steer-k-torsion", type=float, default=200.0, help="Replay steering: force constant for torsion restraints (kJ/mol/rad^2).")
        g_md.add_argument(
            "--steer-global-lambda-during-torsion",
            type=float,
            default=0.2,
            help="Replay steering: global lambda during torsion stage (keeps RMSD/E2E approximately fixed).",
        )
        parser.set_defaults(steer_replay_starts=True, steer_replay_verbose=False)
        g_md.add_argument(
            "--anchoring-source-dir",
            default="",
            help="Directory containing existing anchoring DCDs to bootstrap history (overrides snapshot_dir/anchoring lookup).",
        )


@dataclass
class ProjectionConfig:
    projection: str = "geom2vec_vamp"
    pca_bins: int = 50
    pca_completion_threshold: float = 0.8
    pca_refresh_interval: int = 1
    pca_reward_weight: float = 1.0
    novelty_launch_weight: float = 1.0
    novelty_traj_weight: float = 1.0
    max_pca_train: int = 0
    anchor_balance_weight: float = 1.0
    geom2vec_lagtime: int = 20
    geom2vec_dim: int = 2
    geom2vec_start_transition_weight: float = 0.0
    loo_vamp_exploration_coef: float = 0.0
    loo_vamp_use_ucb: bool = False
    disable_vamp_ensemble: bool = False
    geom2vec_checkpoint: str = ""
    geom2vec_cache_root: str = ""
    geom2vec_manifest_name: str = "geom2vec_manifest.jsonl"
    geom2vec_device: str = ""
    geom2vec_embed_stride: int = 2
    geom2vec_embed_batch_size: int = 32
    geom2vec_hidden_channels: int = 0
    geom2vec_reduction: str = "sum"
    geom2vec_center_ca: bool = True
    geom2vec_exclude_hydrogens: bool = True
    geom2vec_vamp_pair_stride: int = 1
    geom2vec_vamp_train_fraction: float = 0.8
    geom2vec_vamp_n_models: int = 1
    geom2vec_vamp_seed_base: int = 42
    geom2vec_vamp_epsilon: float = 1e-6
    geom2vec_vamp_inverse_mode: str = "regularize"
    geom2vec_vamp_epochs: int = 10
    geom2vec_vamp_batch_size: int = 256
    geom2vec_vamp_learning_rate: float = 5e-4
    geom2vec_vamp_weight_decay: float = 0.0
    geom2vec_vamp_optimizer: str = "Adam"
    geom2vec_vamp_softmax_output: bool = True
    vamp_bin_bins: int = 50
    vamp_bin_clip: float = 3.0
    vamp_bin_max_samples: int = 20000

    @classmethod
    def populate_parser(cls, parser: argparse.ArgumentParser) -> None:
        g_proj = parser.add_argument_group("Projection & Scoring")
        g_proj.add_argument(
            "--projection",
            default="geom2vec_vamp",
            choices=["geom2vec_vamp"],
            help="Projection method (geom2vec_vamp).",
        )
        g_proj.add_argument("--pca-bins", type=int, default=50, help="Grid bins for novelty.")
        g_proj.add_argument("--pca-completion-threshold", type=float, default=0.8, help="Threshold for PCA readiness.")
        g_proj.add_argument("--pca-refresh-interval", type=int, default=1, help="Epochs between PCA/projection updates.")
        g_proj.add_argument("--pca-reward-weight", type=float, default=1.0, help="Weight for coverage reward.")
        g_proj.add_argument("--novelty-launch-weight", type=float, default=1.0, help="Weight for launch novelty.")
        g_proj.add_argument("--novelty-traj-weight", type=float, default=1.0, help="Weight for trajectory novelty.")
        g_proj.add_argument("--max-pca-train", type=int, default=0, help="Max frames for PCA training (0=all).")
        g_proj.add_argument("--anchor-balance-weight", type=float, default=1.0, help="Weight for dual-anchor balance reward (encourages equal visitation of both PDB basins).")

        g_loo = parser.add_argument_group("LOO-VAMP Controls")
        g_loo.add_argument(
            "--loo-vamp-exploration-coef",
            dest="loo_vamp_exploration_coef",
            type=float,
            default=1.0,
            help="UCB exploration coefficient λ for LOO-VAMP: score = μ + λ·σ.",
        )
        g_loo.add_argument(
            "--no-ucb",
            dest="loo_vamp_use_ucb",
            action="store_false",
            help="Disable UCB uncertainty term (use μ only).",
        )
        g_loo.add_argument(
            "--no-vamp-ensemble",
            dest="disable_vamp_ensemble",
            action="store_true",
            help="Disable VAMP ensemble (forces geom2vec_vamp_n_models=1).",
        )
        g_loo.add_argument(
            "--vamp-bin-bins",
            dest="vamp_bin_bins",
            type=int,
            default=50,
            help="Number of bins per axis for the 2D VAMP-bin exploration bonus (first two VAMP coords).",
        )
        g_loo.add_argument(
            "--vamp-bin-clip",
            dest="vamp_bin_clip",
            type=float,
            default=3.0,
            help="Z-score clip range for 2D VAMP-bin exploration bonus (applies after mean/std standardization).",
        )
        g_loo.add_argument(
            "--vamp-bin-max-samples",
            dest="vamp_bin_max_samples",
            type=int,
            default=20000,
            help="Max samples used to estimate mean/std for VAMP-bin standardization at refresh/init.",
        )

        g_vamp = parser.add_argument_group("Geom2Vec VAMP Lobe")
        g_vamp.add_argument(
            "--geom2vec-dim",
            "--geom2vec-n-states",
            dest="geom2vec_dim",
            type=int,
            default=2,
            help="Number of discrete states (K) for the geom2vec VAMP lobe.",
        )
        g_vamp.add_argument(
            "--geom2vec-lagtime",
            dest="geom2vec_lagtime",
            type=int,
            default=20,
            help="Lagtime (in MD steps) used to form time-lagged pairs for the geom2vec VAMP lobe.",
        )
        g_vamp.add_argument(
            "--geom2vec-start-transition-weight",
            dest="geom2vec_start_transition_weight",
            type=float,
            default=1.0,
            help="Weight for a start-to-end displacement bonus in projection space (optional).",
        )

        g_geom = parser.add_argument_group("Geom2Vec VAMP Backbone")
        g_geom.add_argument(
            "--geom2vec-ckpt",
            dest="geom2vec_checkpoint",
            default="",
            help="Path to a geom2vec representation checkpoint (required for projection=geom2vec_vamp).",
        )
        g_geom.add_argument(
            "--geom2vec-cache-root",
            dest="geom2vec_cache_root",
            default="",
            help="Root directory for cached geom2vec embeddings (default: --snapshot-dir; caches next to .dcd).",
        )
        g_geom.add_argument(
            "--geom2vec-manifest-name",
            dest="geom2vec_manifest_name",
            default="geom2vec_manifest.jsonl",
            help="Manifest filename written under --geom2vec-cache-root.",
        )
        g_geom.add_argument(
            "--geom2vec-device",
            dest="geom2vec_device",
            default="",
            help="Device for geom2vec inference and VAMP lobe training (default: cuda if available).",
        )
        g_geom.add_argument(
            "--geom2vec-embed-stride",
            dest="geom2vec_embed_stride",
            type=int,
            default=2,
            help="Stride for subsampling DCD frames before embedding (1 = every frame).",
        )
        g_geom.add_argument(
            "--geom2vec-embed-batch-size",
            dest="geom2vec_embed_batch_size",
            type=int,
            default=32,
            help="Batch size for geom2vec embedding inference.",
        )
        g_geom.add_argument(
            "--geom2vec-hidden-channels",
            dest="geom2vec_hidden_channels",
            type=int,
            default=0,
            help="Override geom2vec hidden channels H (0 = infer from model).",
        )
        g_geom.add_argument(
            "--geom2vec-reduction",
            dest="geom2vec_reduction",
            choices=["sum", "mean"],
            default="sum",
            help="Coarse-graining reduction when pooling atom reps to residue tokens.",
        )
        g_geom.add_argument(
            "--geom2vec-center-ca",
            dest="geom2vec_center_ca",
            action="store_true",
            help="Center per-frame CA coords across tokens (default).",
        )
        g_geom.add_argument(
            "--no-geom2vec-center-ca",
            dest="geom2vec_center_ca",
            action="store_false",
            help="Disable CA centering for the geom2vec lobe input.",
        )
        g_geom.add_argument(
            "--geom2vec-exclude-hydrogens",
            dest="geom2vec_exclude_hydrogens",
            action="store_true",
            help="Exclude hydrogens when building geom2vec inputs (default).",
        )
        g_geom.add_argument(
            "--no-geom2vec-exclude-hydrogens",
            dest="geom2vec_exclude_hydrogens",
            action="store_false",
            help="Include hydrogens when building geom2vec inputs.",
        )
        g_geom.add_argument(
            "--geom2vec-vamp-pair-stride",
            dest="geom2vec_vamp_pair_stride",
            type=int,
            default=1,
            help="Stride for time-lagged pair sampling in the VAMP dataset.",
        )
        g_geom.add_argument(
            "--geom2vec-vamp-train-fraction",
            dest="geom2vec_vamp_train_fraction",
            type=float,
            default=0.8,
            help="Trajectory-level train/val split fraction for VAMP lobe training.",
        )
        g_geom.add_argument(
            "--geom2vec-vamp-n-models",
            dest="geom2vec_vamp_n_models",
            type=int,
            default=3,
            help="Number of independent geom2vec VAMP lobe fits (ensemble size).",
        )
        g_geom.add_argument(
            "--geom2vec-vamp-seed-base",
            dest="geom2vec_vamp_seed_base",
            type=int,
            default=42,
            help="Base RNG seed for geom2vec VAMP lobe ensemble members (seed_base + 1000*i).",
        )
        g_geom.add_argument(
            "--geom2vec-vamp-epsilon",
            dest="geom2vec_vamp_epsilon",
            type=float,
            default=1e-6,
            help="Numerical stabilizer epsilon for VAMP covariance inverses (higher => more stable).",
        )
        g_geom.add_argument(
            "--geom2vec-vamp-inverse-mode",
            dest="geom2vec_vamp_inverse_mode",
            choices=["regularize", "trunc"],
            default="regularize",
            help="Mode for VAMP covariance inversion: regularize or trunc.",
        )
        g_geom.add_argument(
            "--geom2vec-vamp-epochs",
            dest="geom2vec_vamp_epochs",
            type=int,
            default=10,
            help="Epochs for geom2vec VAMP lobe training.",
        )
        g_geom.add_argument(
            "--geom2vec-vamp-batch-size",
            dest="geom2vec_vamp_batch_size",
            type=int,
            default=256,
            help="Batch size for geom2vec VAMP lobe training.",
        )
        g_geom.add_argument(
            "--geom2vec-vamp-learning-rate",
            dest="geom2vec_vamp_learning_rate",
            type=float,
            default=5e-4,
            help="Learning rate for geom2vec VAMP lobe training.",
        )
        g_geom.add_argument(
            "--geom2vec-vamp-weight-decay",
            dest="geom2vec_vamp_weight_decay",
            type=float,
            default=0.0,
            help="Weight decay for geom2vec VAMP lobe training.",
        )
        g_geom.add_argument(
            "--geom2vec-vamp-optimizer",
            dest="geom2vec_vamp_optimizer",
            default="Adam",
            help="Optimizer name for geom2vec VAMP lobe training (passed to geom2vec VAMPNetConfig).",
        )
        g_geom.add_argument(
            "--geom2vec-vamp-softmax-output",
            dest="geom2vec_vamp_softmax_output",
            action="store_true",
            help="Use a Softmax output activation inside the geom2vec lobe (default).",
        )
        g_geom.add_argument(
            "--no-geom2vec-vamp-softmax-output",
            dest="geom2vec_vamp_softmax_output",
            action="store_false",
            help="Disable Softmax output activation (treat lobe outputs as logits).",
        )
        parser.set_defaults(
            geom2vec_center_ca=True,
            geom2vec_exclude_hydrogens=True,
            geom2vec_vamp_softmax_output=True,
        )


@dataclass
class IOConfig:
    checkpoint_dir: str = ""
    checkpoint_interval: int = 0
    resume: bool = False
    snapshot_dir: str = ""
    stats_log: str = ""
    plot_freq: int = 1
    history_dcd_dir: str = ""
    history_dcd_stride: int = 2
    history_dcd_max_frames: int = 0
    rmsd_filter: float = 0.0
    save_epoch_coords: bool = True

    @classmethod
    def populate_parser(cls, parser: argparse.ArgumentParser) -> None:
        g_io = parser.add_argument_group("I/O & Checkpointing")
        g_io.add_argument("--checkpoint-dir", default="", help="Directory to save checkpoints.")
        g_io.add_argument("--checkpoint-interval", type=int, default=0, help="Epochs between checkpoints.")
        g_io.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
        g_io.add_argument("--snapshot-dir", default="", help="Directory for epoch snapshots.")
        g_io.add_argument("--stats-log", default="", help="Path to CSV stats log.")
        g_io.add_argument("--plot-freq", type=int, default=1, help="Epochs between plots.")
        g_io.add_argument("--history-dcd-dir", default="", help="Directory of DCDs to bootstrap history.")
        g_io.add_argument("--history-dcd-stride", type=int, default=1, help="Stride for history DCDs.")
        g_io.add_argument("--history-dcd-max-frames", type=int, default=0, help="Max frames to load from history.")
        g_io.add_argument("--rmsd-filter", type=float, default=0.0, help="RMSD threshold (nm) for filtering.")
        g_io.add_argument(
            "--save-epoch-coords",
            dest="save_epoch_coords",
            action="store_true",
            help="Save coordinates in epoch NPZs (default).",
        )
        g_io.add_argument(
            "--no-save-epoch-coords",
            dest="save_epoch_coords",
            action="store_false",
            help="Disable saving coordinates in epoch NPZs.",
        )
        parser.set_defaults(save_epoch_coords=True)


@dataclass
class TrainingConfig:
    """Centralized configuration for GESPO training."""

    core: CoreConfig
    rl: RLConfig
    policy: PolicyConfig
    sampling: SamplingConfig
    md: MDConfig
    projection: ProjectionConfig
    io: IOConfig

    _flat_field_map: ClassVar[Dict[str, Tuple[str, str]]] = {
        "grp_size": ("core", "grp_size"),
        "episodes": ("core", "episodes"),
        "platform": ("core", "platform"),
        "gen_device": ("core", "gen_device"),
        "pdb": ("core", "pdb"),
        "generative_loop": ("core", "generative_loop"),
        "cond_checkpoint": ("core", "cond_checkpoint"),
        "cond_data": ("core", "cond_data"),
        "enable_multi_gpu": ("core", "enable_multi_gpu"),
        "lr": ("rl", "lr"),
        "grpo_iters": ("rl", "grpo_iters"),
        "sync_interval": ("rl", "sync_interval"),
        "ent_coef": ("rl", "ent_coef"),
        "clip_eps": ("rl", "clip_eps"),
        "kl_coef": ("rl", "kl_coef"),
        "norm_adv": ("rl", "norm_adv"),
        "start_cluster_entropy_coef": ("rl", "start_cluster_entropy_coef"),
        "vamp_bin_entropy_coef": ("rl", "vamp_bin_entropy_coef"),
        "vamp_bin_norm_ema_beta": ("rl", "vamp_bin_norm_ema_beta"),
        "grad_clip": ("rl", "grad_clip"),
        "loss_prec": ("rl", "loss_prec"),
        "policy_type": ("policy", "policy_type"),
        "use_muon": ("policy", "use_muon"),
        "old_init": ("policy", "old_init"),
        "old_perturb_scale": ("policy", "old_perturb_scale"),
        "num_mixtures": ("policy", "num_mixtures"),
        "use_bias": ("policy", "use_bias"),
        "policy_conditioning_mode": ("policy", "policy_conditioning_mode"),
        "discrete_state_starts": ("policy", "discrete_state_starts"),
        "use_boundary_replay": ("policy", "use_boundary_replay"),
        "reset_policy_on_projector_refresh": ("policy", "reset_policy_on_projector_refresh"),
        "least_count_starts": ("policy", "least_count_starts"),
        "uniform_starts": ("policy", "uniform_starts"),
        "record_stride": ("sampling", "record_stride"),
        "sigma_min": ("sampling", "sigma_min"),
        "retrieval_candidates": ("sampling", "retrieval_candidates"),
        "retrieval_topk": ("sampling", "retrieval_topk"),
        "retrieval_max_resamples": ("sampling", "retrieval_max_resamples"),
        "retrieval_cache_max_frames": ("sampling", "retrieval_cache_max_frames"),
        "retrieval_use_global_metrics": ("sampling", "retrieval_use_global_metrics"),
        "retrieval_global_weight": ("sampling", "retrieval_global_weight"),
        "global_rmsd_range": ("sampling", "global_rmsd_range"),
        "global_e2e_range": ("sampling", "global_e2e_range"),
        "forcefield_protein": ("md", "forcefield_protein"),
        "forcefield_water": ("md", "forcefield_water"),
        "forcefield_implicit": ("md", "forcefield_implicit"),
        "temperature": ("md", "temperature"),
        "friction": ("md", "friction"),
        "minimize_iters": ("md", "minimize_iters"),
        "heat_stages": ("md", "heat_stages"),
        "heat_steps": ("md", "heat_steps"),
        "equilibration_steps": ("md", "equilibration_steps"),
        "implicit_solvent": ("md", "implicit_solvent"),
        "explicit_solvent": ("md", "explicit_solvent"),
        "explicit_solvent_replay_full_coords": ("md", "explicit_solvent_replay_full_coords"),
        "n_md_steps": ("md", "n_md_steps"),
        "energy_threshold": ("md", "energy_threshold"),
        "max_resample_attempts": ("md", "max_resample_attempts"),
        "torsion_decimals": ("md", "torsion_decimals"),
        "device": ("md", "device"),
        "use_replay_resets": ("md", "use_replay_resets"),
        "steer_replay_starts": ("md", "steer_replay_starts"),
        "steer_replay_verbose": ("md", "steer_replay_verbose"),
        "steer_global_steps": ("md", "steer_global_steps"),
        "steer_global_updates": ("md", "steer_global_updates"),
        "steer_torsion_steps": ("md", "steer_torsion_steps"),
        "steer_torsion_updates": ("md", "steer_torsion_updates"),
        "steer_release_steps": ("md", "steer_release_steps"),
        "steer_refine_minimize_iters": ("md", "steer_refine_minimize_iters"),
        "steer_k_rmsd": ("md", "steer_k_rmsd"),
        "steer_k_e2e": ("md", "steer_k_e2e"),
        "steer_k_torsion": ("md", "steer_k_torsion"),
        "steer_global_lambda_during_torsion": ("md", "steer_global_lambda_during_torsion"),
        "anchoring_round": ("md", "anchoring_round"),
        "anchoring_steps": ("md", "anchoring_steps"),
        "anchoring_num_trajs": ("md", "anchoring_num_trajs"),
        "anchoring_pdb_only": ("md", "anchoring_pdb_only"),
        "anchoring_source_dir": ("md", "anchoring_source_dir"),
        "projection": ("projection", "projection"),
        "pca_bins": ("projection", "pca_bins"),
        "pca_completion_threshold": ("projection", "pca_completion_threshold"),
        "pca_refresh_interval": ("projection", "pca_refresh_interval"),
        "pca_reward_weight": ("projection", "pca_reward_weight"),
        "novelty_launch_weight": ("projection", "novelty_launch_weight"),
        "novelty_traj_weight": ("projection", "novelty_traj_weight"),
        "max_pca_train": ("projection", "max_pca_train"),
        "anchor_balance_weight": ("projection", "anchor_balance_weight"),
        "geom2vec_lagtime": ("projection", "geom2vec_lagtime"),
        "geom2vec_dim": ("projection", "geom2vec_dim"),
        "geom2vec_start_transition_weight": ("projection", "geom2vec_start_transition_weight"),
        "loo_vamp_exploration_coef": ("projection", "loo_vamp_exploration_coef"),
        "loo_vamp_use_ucb": ("projection", "loo_vamp_use_ucb"),
        "disable_vamp_ensemble": ("projection", "disable_vamp_ensemble"),
        "geom2vec_checkpoint": ("projection", "geom2vec_checkpoint"),
        "geom2vec_cache_root": ("projection", "geom2vec_cache_root"),
        "geom2vec_manifest_name": ("projection", "geom2vec_manifest_name"),
        "geom2vec_device": ("projection", "geom2vec_device"),
        "geom2vec_embed_stride": ("projection", "geom2vec_embed_stride"),
        "geom2vec_embed_batch_size": ("projection", "geom2vec_embed_batch_size"),
        "geom2vec_hidden_channels": ("projection", "geom2vec_hidden_channels"),
        "geom2vec_reduction": ("projection", "geom2vec_reduction"),
        "geom2vec_center_ca": ("projection", "geom2vec_center_ca"),
        "geom2vec_exclude_hydrogens": ("projection", "geom2vec_exclude_hydrogens"),
        "geom2vec_vamp_pair_stride": ("projection", "geom2vec_vamp_pair_stride"),
        "geom2vec_vamp_train_fraction": ("projection", "geom2vec_vamp_train_fraction"),
        "geom2vec_vamp_n_models": ("projection", "geom2vec_vamp_n_models"),
        "geom2vec_vamp_seed_base": ("projection", "geom2vec_vamp_seed_base"),
        "geom2vec_vamp_epsilon": ("projection", "geom2vec_vamp_epsilon"),
        "geom2vec_vamp_inverse_mode": ("projection", "geom2vec_vamp_inverse_mode"),
        "geom2vec_vamp_epochs": ("projection", "geom2vec_vamp_epochs"),
        "geom2vec_vamp_batch_size": ("projection", "geom2vec_vamp_batch_size"),
        "geom2vec_vamp_learning_rate": ("projection", "geom2vec_vamp_learning_rate"),
        "geom2vec_vamp_weight_decay": ("projection", "geom2vec_vamp_weight_decay"),
        "geom2vec_vamp_optimizer": ("projection", "geom2vec_vamp_optimizer"),
        "geom2vec_vamp_softmax_output": ("projection", "geom2vec_vamp_softmax_output"),
        "vamp_bin_bins": ("projection", "vamp_bin_bins"),
        "vamp_bin_clip": ("projection", "vamp_bin_clip"),
        "vamp_bin_max_samples": ("projection", "vamp_bin_max_samples"),
        "checkpoint_dir": ("io", "checkpoint_dir"),
        "checkpoint_interval": ("io", "checkpoint_interval"),
        "resume": ("io", "resume"),
        "snapshot_dir": ("io", "snapshot_dir"),
        "stats_log": ("io", "stats_log"),
        "plot_freq": ("io", "plot_freq"),
        "history_dcd_dir": ("io", "history_dcd_dir"),
        "history_dcd_stride": ("io", "history_dcd_stride"),
        "history_dcd_max_frames": ("io", "history_dcd_max_frames"),
        "rmsd_filter": ("io", "rmsd_filter"),
        "save_epoch_coords": ("io", "save_epoch_coords"),
    }

    def __getattr__(self, name: str):
        mapping = self._flat_field_map.get(name)
        if mapping:
            section, field_name = mapping
            return getattr(getattr(self, section), field_name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        if name in {"core", "rl", "policy", "sampling", "md", "projection", "io"}:
            object.__setattr__(self, name, value)
            return
        mapping = self._flat_field_map.get(name)
        if mapping and mapping[0] in self.__dict__:
            setattr(getattr(self, mapping[0]), mapping[1], value)
            return
        object.__setattr__(self, name, value)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def populate_parser(cls, parser: argparse.ArgumentParser) -> None:
        CoreConfig.populate_parser(parser)
        RLConfig.populate_parser(parser)
        PolicyConfig.populate_parser(parser)
        SamplingConfig.populate_parser(parser)
        MDConfig.populate_parser(parser)
        ProjectionConfig.populate_parser(parser)
        IOConfig.populate_parser(parser)

    @classmethod
    def from_args(cls, args) -> "TrainingConfig":
        """Create config from argparse namespace."""
        def get(name, default):
            return getattr(args, name, default)

        gen_device_str = args.gen_device
        if gen_device_str.startswith("cuda:"):
            device_idx = int(gen_device_str.split(":")[1])
        elif gen_device_str.isdigit():
            device_idx = int(gen_device_str)
        else:
            device_idx = 0

        implicit_flag, explicit_solvent = _validate_solvent_flags(
            bool(get("implicit_solvent", False)),
            bool(get("explicit_solvent", False)),
        )

        explicit_solvent_replay_full_coords = _resolve_explicit_replay(
            explicit_solvent,
            bool(get("explicit_solvent_replay_full_coords", False)),
        )

        geom2vec_vamp_n_models = _resolve_vamp_ensemble(
            bool(get("disable_vamp_ensemble", False)),
            int(get("geom2vec_vamp_n_models", 1) or 1),
        )

        retrieval_global_weight = get("retrieval_global_weight", 1.0)
        if retrieval_global_weight is None:
            retrieval_global_weight = 1.0

        global_rmsd_range = tuple(get("global_rmsd_range", None)) if get("global_rmsd_range", None) else None
        global_e2e_range = tuple(get("global_e2e_range", None)) if get("global_e2e_range", None) else None
        _validate_range("global_rmsd_range", global_rmsd_range)
        _validate_range("global_e2e_range", global_e2e_range)

        return cls(
            core=CoreConfig(
                grp_size=args.grp_size,
                episodes=args.episodes,
                platform=args.platform,
                gen_device=args.gen_device,
                pdb=args.pdb,
                generative_loop=get("generative_loop", False),
                cond_checkpoint=args.cond_checkpoint,
                cond_data=args.cond_data,
                enable_multi_gpu=get("enable_multi_gpu", False),
            ),
            rl=RLConfig(
                lr=get("lr", 1e-4),
                grpo_iters=get("grpo_iters", 1),
                sync_interval=get("sync_interval", 5),
                ent_coef=float(get("ent_coef", ENT_COEF_DEFAULT) or ENT_COEF_DEFAULT),
                kl_coef=float(get("kl_coef", KL_COEF_DEFAULT) or KL_COEF_DEFAULT),
                grad_clip=float(get("grad_clip", GRAD_CLIP_DEFAULT) or GRAD_CLIP_DEFAULT),
                start_cluster_entropy_coef=float(get("start_cluster_entropy_coef", -1.0)),
                vamp_bin_entropy_coef=float(get("vamp_bin_entropy_coef", -1.0)),
                vamp_bin_norm_ema_beta=float(get("vamp_bin_norm_ema_beta", 0.9) or 0.9),
            ),
            policy=PolicyConfig(
                policy_type=get("policy_type", "mlp"),
                use_muon=get("use_muon", False),
                old_init=get("old_init", "copy"),
                old_perturb_scale=get("old_perturb_scale", 1e-3),
                num_mixtures=get("num_mixtures", 3),
                use_bias=get("use_bias", False),
                policy_conditioning_mode=get("policy_conditioning_mode", "hybrid"),
                discrete_state_starts=get("discrete_state_starts", False),
                use_boundary_replay=get("use_boundary_replay", True),
                reset_policy_on_projector_refresh=get("reset_policy_on_projector_refresh", True),
                least_count_starts=get("least_count_starts", False),
                uniform_starts=get("uniform_starts", False),
            ),
            sampling=SamplingConfig(
                record_stride=max(1, get("record_stride", 10)),
                sigma_min=get("sigma_min", 0.1),
                retrieval_candidates=int(get("retrieval_candidates", 2048) or 2048),
                retrieval_topk=int(get("retrieval_topk", 32) or 32),
                retrieval_max_resamples=int(get("retrieval_max_resamples", 3) or 3),
                retrieval_cache_max_frames=int(get("retrieval_cache_max_frames", 0) or 0),
                retrieval_use_global_metrics=bool(get("retrieval_use_global_metrics", True)),
                retrieval_global_weight=float(retrieval_global_weight),
                global_rmsd_range=global_rmsd_range,
                global_e2e_range=global_e2e_range,
            ),
            md=MDConfig(
                forcefield_protein=get("forcefield_protein", "amber14-all.xml"),
                forcefield_water=get("forcefield_water", "amber14/tip3pfb.xml"),
                forcefield_implicit=get("forcefield_implicit", "implicit/gbn2.xml"),
                temperature=get("temperature", 300.0),
                friction=get("friction", 1.0),
                minimize_iters=get("minimize_iters", 200),
                heat_stages=get("heat_stages", 4),
                heat_steps=get("heat_steps", 100),
                equilibration_steps=get("equilibration_steps", 5000),
                implicit_solvent=implicit_flag,
                explicit_solvent=explicit_solvent,
                explicit_solvent_replay_full_coords=explicit_solvent_replay_full_coords,
                n_md_steps=get("n_md_steps", 10000),
                energy_threshold=get("energy_threshold", 1e10),
                max_resample_attempts=get("max_resample_attempts", 5),
                torsion_decimals=get("torsion_decimals", 3),
                device=device_idx,
                use_replay_resets=get("use_replay_resets", True),
                steer_replay_starts=bool(get("steer_replay_starts", True)),
                steer_replay_verbose=bool(get("steer_replay_verbose", False)),
                steer_global_steps=int(get("steer_global_steps", 5000) or 5000),
                steer_global_updates=int(get("steer_global_updates", 25) or 25),
                steer_torsion_steps=int(get("steer_torsion_steps", 1000) or 1000),
                steer_torsion_updates=int(get("steer_torsion_updates", 25) or 25),
                steer_release_steps=int(get("steer_release_steps", 200) or 200),
                steer_refine_minimize_iters=int(get("steer_refine_minimize_iters", 0) or 0),
                steer_k_rmsd=float(get("steer_k_rmsd", 5000.0) or 5000.0),
                steer_k_e2e=float(get("steer_k_e2e", 5000.0) or 5000.0),
                steer_k_torsion=float(get("steer_k_torsion", 200.0) or 200.0),
                steer_global_lambda_during_torsion=float(get("steer_global_lambda_during_torsion", 0.2) or 0.2),
                anchoring_round=get("anchoring_round", True),
                anchoring_steps=get("anchoring_steps", 0),
                anchoring_num_trajs=get("anchoring_num_trajs", 0),
                anchoring_pdb_only=get("anchoring_pdb_only", False),
                anchoring_source_dir=get("anchoring_source_dir", ""),
            ),
            projection=ProjectionConfig(
                projection=get("projection", "geom2vec_vamp").lower(),
                pca_bins=get("pca_bins", 30),
                pca_completion_threshold=get("pca_completion_threshold", 0.8),
                pca_refresh_interval=get("pca_refresh_interval", 1),
                pca_reward_weight=get("pca_reward_weight", 1.0),
                novelty_launch_weight=get("novelty_launch_weight", 1.0),
                novelty_traj_weight=get("novelty_traj_weight", 1.0),
                max_pca_train=get("max_pca_train", 0),
                anchor_balance_weight=get("anchor_balance_weight", 0.0),
                geom2vec_lagtime=get("geom2vec_lagtime", 20),
                geom2vec_dim=get("geom2vec_dim", 2),
                geom2vec_start_transition_weight=get("geom2vec_start_transition_weight", 0.0),
                loo_vamp_exploration_coef=float(get("loo_vamp_exploration_coef", 0.0) or 0.0),
                loo_vamp_use_ucb=bool(get("loo_vamp_use_ucb", False)),
                disable_vamp_ensemble=bool(get("disable_vamp_ensemble", False)),
                geom2vec_checkpoint=get("geom2vec_checkpoint", ""),
                geom2vec_cache_root=get("geom2vec_cache_root", ""),
                geom2vec_manifest_name=get("geom2vec_manifest_name", "geom2vec_manifest.jsonl"),
                geom2vec_device=get("geom2vec_device", ""),
                geom2vec_embed_stride=int(get("geom2vec_embed_stride", 2) or 2),
                geom2vec_embed_batch_size=int(get("geom2vec_embed_batch_size", 32) or 32),
                geom2vec_hidden_channels=int(get("geom2vec_hidden_channels", 0) or 0),
                geom2vec_reduction=str(get("geom2vec_reduction", "sum") or "sum"),
                geom2vec_center_ca=bool(get("geom2vec_center_ca", True)),
                geom2vec_exclude_hydrogens=bool(get("geom2vec_exclude_hydrogens", True)),
                geom2vec_vamp_pair_stride=int(get("geom2vec_vamp_pair_stride", 1) or 1),
                geom2vec_vamp_train_fraction=float(get("geom2vec_vamp_train_fraction", 0.8) or 0.8),
                geom2vec_vamp_n_models=geom2vec_vamp_n_models,
                geom2vec_vamp_seed_base=int(get("geom2vec_vamp_seed_base", 42) or 42),
                geom2vec_vamp_epsilon=float(get("geom2vec_vamp_epsilon", 1e-6) or 1e-6),
                geom2vec_vamp_inverse_mode=str(get("geom2vec_vamp_inverse_mode", "regularize") or "regularize"),
                geom2vec_vamp_epochs=int(get("geom2vec_vamp_epochs", 10) or 10),
                geom2vec_vamp_batch_size=int(get("geom2vec_vamp_batch_size", 256) or 256),
                geom2vec_vamp_learning_rate=float(get("geom2vec_vamp_learning_rate", 5e-4) or 5e-4),
                geom2vec_vamp_weight_decay=float(get("geom2vec_vamp_weight_decay", 0.0) or 0.0),
                geom2vec_vamp_optimizer=str(get("geom2vec_vamp_optimizer", "Adam") or "Adam"),
                geom2vec_vamp_softmax_output=bool(get("geom2vec_vamp_softmax_output", True)),
                vamp_bin_bins=int(get("vamp_bin_bins", 50) or 50),
                vamp_bin_clip=float(get("vamp_bin_clip", 3.0) or 3.0),
                vamp_bin_max_samples=int(get("vamp_bin_max_samples", 20000) or 20000),
            ),
            io=IOConfig(
                checkpoint_dir=get("checkpoint_dir", ""),
                checkpoint_interval=get("checkpoint_interval", 0),
                resume=get("resume", False),
                snapshot_dir=get("snapshot_dir", ""),
                stats_log=get("stats_log", ""),
                plot_freq=get("plot_freq", 1),
                history_dcd_dir=get("history_dcd_dir", ""),
                history_dcd_stride=get("history_dcd_stride", 1),
                history_dcd_max_frames=get("history_dcd_max_frames", 0),
                rmsd_filter=get("rmsd_filter", 0.0),
                save_epoch_coords=get("save_epoch_coords", True),
            ),
        )
