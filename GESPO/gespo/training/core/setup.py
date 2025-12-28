import torch
import numpy as np
import math
from typing import Tuple, List, Any
from ...config import TrainingConfig
from ...sampling import PeptideSampler
from ..policies.rl import ResidueTransformerPolicy
from .scorer import VendiNovelty, PCAScorer
from gespo.utils.logging import log_print

def _validate_model_consistency(config: TrainingConfig, env: PeptideSampler) -> None:
    """
    Validate consistency between diffusion model training and policy architecture.
    
    Checks:
    1. Conditioning mode compatibility (torsion/pairwise/hybrid)
    2. Dimensional consistency for hybrid mode
    3. Per-residue vs global conditioning injection alignment
    4. Interleaved distance pattern metadata
    """
    gen = env.generator
    
    # Check 1: Conditioning mode match
    policy_mode = config.policy_conditioning_mode.lower()
    gen_mode = getattr(gen, 'conditioning_mode', 'torsion').lower()
    
    if policy_mode != gen_mode:
        raise ValueError(
            f"[Consistency Error] Policy conditioning mode '{policy_mode}' does not match "
            f"generator conditioning mode '{gen_mode}'. The diffusion model was trained with "
            f"'{gen_mode}' conditioning but the policy is configured for '{policy_mode}'."
        )
    
    # Check 2: For hybrid/pairwise modes, verify interleaved distance metadata exists
    if policy_mode in ("hybrid", "pairwise"):
        if not hasattr(gen, 'partner_residues') or gen.partner_residues is None:
            raise ValueError(
                f"[Consistency Error] Generator lacks 'partner_residues' metadata required "
                f"for {policy_mode} mode. The diffusion model may have been trained without "
                f"pairwise distances, or the checkpoint is incomplete."
            )
        if not hasattr(gen, 'pair_mask') or gen.pair_mask is None:
            raise ValueError(
                f"[Consistency Error] Generator lacks 'pair_mask' metadata required "
                f"for {policy_mode} mode."
            )
        
        # Verify dimensions
        partner_shape = np.asarray(gen.partner_residues).shape
        expected_residues = env.n_residues
        if partner_shape[0] != expected_residues:
            raise ValueError(
                f"[Consistency Error] Generator partner_residues has {partner_shape[0]} residues "
                f"but environment has {expected_residues} residues."
            )
        
        max_partners = partner_shape[1] if len(partner_shape) > 1 else 0
        env_max_partners = getattr(env, 'max_partners', 0)
        if max_partners != env_max_partners:
            raise ValueError(
                f"[Consistency Error] Generator has max_partners={max_partners} but "
                f"environment extracted max_partners={env_max_partners}."
            )
    
    # Check 3: For hybrid mode with Transformer policy, verify per-residue conditioning
    if policy_mode == "hybrid" and not getattr(config, "discrete_state_starts", False):
        # Check if generator was trained with per-residue conditioning
        per_res_torsion = getattr(gen, 'per_res_torsion', False)
        cond_injection = getattr(gen, 'cond_injection_mode', 'global')
        
        # The Transformer policy always outputs per-residue format
        # So the generator MUST use per-residue conditioning injection
        if cond_injection != 'per_residue':
            raise ValueError(
                f"[Consistency Error] Transformer policy outputs per-residue format but "
                f"the diffusion model was trained with cond_injection_mode='{cond_injection}'. "
                f"For hybrid mode with Transformer policy, the diffusion model MUST be trained "
                f"with --cond-injection-mode per_residue to ensure dimensional compatibility.\n"
                f"\nExpected conditioning dimension (per-residue):\n"
                f"  Torsion: {env.n_residues} × 4 = {env.n_residues * 4}\n"
                f"  Pairwise: {env.n_residues} × {env.max_partners} = {env.n_residues * env.max_partners}\n"
                f"  Total: {env.n_residues * 4 + env.n_residues * env.max_partners}\n"
                f"\nCurrent generator conditioning_dim: {gen.conditioning_dim}\n"
                f"\nTo fix: Retrain diffusion model with --cond-injection-mode per_residue"
            )
        
        # Verify actual dimensions match
        expected_torsion_dim = env.n_residues * 4
        expected_pair_dim = env.n_residues * env.max_partners
        expected_total = expected_torsion_dim + expected_pair_dim
        
        actual_total = gen.conditioning_dim
        if actual_total != expected_total:
            hybrid_torsion_dim = getattr(gen, 'hybrid_torsion_dim', 0)
            hybrid_pair_dim = getattr(gen, 'hybrid_pair_dim', 0)
            raise ValueError(
                f"[Consistency Error] Dimensional mismatch in hybrid conditioning.\n"
                f"Expected: torsion={expected_torsion_dim} + pairwise={expected_pair_dim} = {expected_total}\n"
                f"Generator: torsion={hybrid_torsion_dim} + pairwise={hybrid_pair_dim} = {actual_total}\n"
                f"This indicates the diffusion model was trained with a different residue count or "
                f"max_partners setting."
            )
    
    log_print(f"[Consistency Check] ✓ Policy and generator are compatible ({policy_mode} mode)")
    if policy_mode in ("hybrid", "pairwise"):
        log_print(f"[Consistency Check] ✓ Using interleaved distances: "
              f"{env.n_residues} residues × {env.max_partners} partners per residue")


def initialize_policy(config: TrainingConfig, envs: List[PeptideSampler], device: str) -> Tuple[Any, Any, List, List]:
    """Initialize agent, old_agent, optimizers, and parameters."""
    env = envs[0]
    gen = env.generator
    
    # Validate model consistency before initialization
    _validate_model_consistency(config, env)
    
    if getattr(config, "discrete_state_starts", False):
        from ..policies.rl import StateLogitPolicy
        proj_mode = str(getattr(config, "projection", "pca")).lower()
        if proj_mode not in {"geom2vec_vamp"}:
            raise ValueError("Discrete-state starts require --projection=geom2vec_vamp.")
        num_states = int(getattr(config, "geom2vec_dim", 8) or 8)
        agent = StateLogitPolicy(num_states=num_states, feature_dim=None).to(device)
        agent_old = StateLogitPolicy(num_states=num_states, feature_dim=None).to(device)
    else:
        # Transformer policy works on per-residue features
        num_residues = env.n_residues
        feature_dim = env.per_residue_feature_dim
        
        # Determine pair_dim for hybrid mode output (per-residue, not global)
        # IMPORTANT: pair_dim is the per-residue pairwise dimension (max_partners),
        # NOT the global flattened dimension (hybrid_pair_dim = n_residues * max_partners).
        # 
        # The policy outputs per-residue predictions [B, N, K] which are then flattened
        # to [B, N*K] before passing to the generator. This maintains architectural
        # consistency: the policy sees and predicts per-residue structure.
        pair_dim = 0
        if config.policy_conditioning_mode == "hybrid":
            # pair_dim = max_partners (per-residue pairwise dimension)
            pair_dim = env.max_partners if hasattr(env, 'max_partners') else 0
        
        agent = ResidueTransformerPolicy(
            num_residues=num_residues,
            feature_dim=feature_dim,
            dropout=0.0,
            pair_dim=pair_dim
        ).to(device)
        
        agent_old = ResidueTransformerPolicy(
            num_residues=num_residues,
            feature_dim=feature_dim,
            dropout=0.0,
            pair_dim=pair_dim
        ).to(device)
        
        if getattr(config, "use_bias", False):
            log_print("[WARN] Transformer policy ignores metadynamics hill control; --use-bias is currently inactive.")

        # Initialize transformer output uncertainties to be reasonably tight so
        # the initial policy is not effectively uniform over torsions/pairs.
        with torch.no_grad():
            # Torsion head log_std: start with a modest std (e.g. 0.3 rad)
            init_torsion_log_std = math.log(0.3)
            agent.log_std.fill_(init_torsion_log_std)
            # Pairwise head (if present): distances are in normalized space,
            # so a std of ~1.0 is a reasonable starting point.
            if agent.pair_dim > 0 and hasattr(agent, "pair_log_std"):
                init_pair_log_std = math.log(1.0)
                agent.pair_log_std.fill_(init_pair_log_std)
        
    # Initialize Old Policy
    # CRITICAL: agent_old must be synced with agent to ensure PPO ratios start at 1.0.
    agent_old.load_state_dict(agent.state_dict())

    agent_old.eval()

    # Optimizers
    policy_params = list(agent.parameters())
    if config.use_muon:
        try:
            from ...utils.muon import Muon
            # Split params
            muon_params = [p for p in agent.parameters() if p.ndim >= 2]
            adam_params = [p for p in agent.parameters() if p.ndim < 2]
            policy_opts = [
                Muon(muon_params, lr=0.02, momentum=0.95),
                torch.optim.AdamW(adam_params, lr=config.lr, weight_decay=0.01)
            ]
        except ImportError:
            log_print("Muon optimizer not found, falling back to Adam.")
            policy_opts = [torch.optim.Adam(policy_params, lr=config.lr)]
    else:
        policy_opts = [torch.optim.Adam(policy_params, lr=config.lr)]

    return agent, agent_old, policy_opts, policy_params

def initialize_projection_scorer(config: TrainingConfig, action_dim: int, args_ns) -> PCAScorer:
    """Initialize the (auxiliary) PCA scorer used for coverage diagnostics/rewards."""

    proj_dim = int(getattr(config, "geom2vec_dim", 2) or 2)
    return PCAScorer(
        n_cvs=action_dim,
        n_components=proj_dim,
        bins=config.pca_bins,
        completion_threshold=config.pca_completion_threshold,
        projector=None,
        projection_name="PCA",
        freeze_projector=False,
    )
