import torch
import numpy as np
import os
import math
from typing import Tuple, List, Any
from ..config import TrainingConfig
from ..sampling import PeptideSampler
from .rl import GRPOAgent

def _validate_model_consistency(config: TrainingConfig, env: PeptideSampler) -> None:
    """
    Validate consistency between diffusion model training and policy architecture.

    Checks:
    1. Conditioning mode compatibility (torsion/pairwise/hybrid)
    2. Dimensional consistency for hybrid mode
    3. Per-residue vs global conditioning injection alignment
    4. Interleaved distance pattern metadata

    Note: Validation is skipped when generator is None (multi-conformer anchoring mode).
    """
    gen = env.generator

    # Skip validation if generator is not loaded (multi-conformer mode)
    if gen is None:
        print("[INFO] Skipping model consistency validation: Generator not loaded (multi-conformer mode)")
        return

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
    if policy_mode == "hybrid" and config.policy_type != "mlp":
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
    
    print(f"[Consistency Check] ✓ Policy and generator are compatible ({policy_mode} mode)")
    if policy_mode in ("hybrid", "pairwise"):
        print(f"[Consistency Check] ✓ Using interleaved distances: "
              f"{env.n_residues} residues × {env.max_partners} partners per residue")


def initialize_policy(config: TrainingConfig, envs: List[PeptideSampler], device: str) -> Tuple[Any, Any, List, List]:
    """Initialize agent, old_agent, optimizers, and parameters."""
    env = envs[0]
    gen = env.generator
    
    # Validate model consistency before initialization
    _validate_model_consistency(config, env)
    proj_mode = str(getattr(config, "projection", "pca")).lower()
    if getattr(config, "discrete_state_starts", False) and proj_mode in {"geom2vec_vamp"}:
        from .rl import StateLogitPolicy
        num_states = int(getattr(config, "geom2vec_dim", 8) or 8)
        # Use a global categorical distribution over states (no feature conditioning).
        agent = StateLogitPolicy(num_states=num_states, feature_dim=None).to(device)
        agent_old = StateLogitPolicy(num_states=num_states, feature_dim=None).to(device)
    else:
        if config.policy_conditioning_mode == "torsion":
            cond_dim = env.n_cvs * 3  # [angles, cos, sin]
        elif config.policy_conditioning_mode == "pairwise":
            # FIX: Fallback dimension should match env.state() fallback (torsion vec is 3x)
            cond_dim = getattr(gen, 'pair_feature_dim', env.n_cvs * 3)
        elif config.policy_conditioning_mode == "hybrid":
            hybrid_pair_dim = getattr(gen, 'hybrid_pair_dim', 0)
            cond_dim = (env.n_cvs * 3) + hybrid_pair_dim  # torsion_vec + pair features
        elif config.policy_conditioning_mode == "ca":
            # For CA mode, use number of CA atoms * 3 (x,y,z)
            cond_dim = len(getattr(env, 'ca_atom_indices', [])) * 3
        elif config.policy_conditioning_mode == "ca_torsion":
            # CA block (aligned coordinates) plus torsion sin/cos block.
            # Matches env.state() which returns: n_ca*3 + 2*n_cvs.
            n_ca = len(getattr(env, 'ca_atom_indices', []))
            cond_dim = (n_ca * 3) + (env.n_cvs * 2)
        else:
            raise ValueError(f"Unknown conditioning mode: {config.policy_conditioning_mode}")
        
        agent = GRPOAgent(
            state_dim=cond_dim,
            action_dim=env.n_cvs,
            num_mixtures=config.num_mixtures,
            hidden_dim=256
        ).to(device)
        
        agent_old = GRPOAgent(
            state_dim=cond_dim,
            action_dim=env.n_cvs,
            num_mixtures=config.num_mixtures,
            hidden_dim=256
        ).to(device)

        # Initialize start CV distribution around the current environment
        # torsions (typically the reference PDB) with a reasonably small
        # standard deviation so early starts are not purely random.
        with torch.no_grad():
            try:
                baseline_cv = torch.as_tensor(env.cv_values, dtype=torch.float32, device=device)
                if baseline_cv.numel() == env.n_cvs and baseline_cv.numel() == agent.start_mu.numel():
                    agent.start_mu.copy_(baseline_cv)
                    agent_old.start_mu.copy_(baseline_cv)
                    print(f"[Init] Initialized start_mu from environment CVs")
                else:
                    print(f"[Init] CV shape mismatch: env={baseline_cv.numel()}, policy={agent.start_mu.numel()}")
            except Exception as e:
                # If cv_values are unavailable, keep the default zero mean.
                print(f"[Init] Could not initialize start_mu from env: {e}")
            # Use a moderate angular spread (e.g., ~30 degrees) for starts.
            init_start_std = math.pi / 6.0  # ~30°
            init_start_log_std = math.log(init_start_std)
            agent.log_std_s.fill_(init_start_log_std)

    # Initialize Old Policy (always copy for PPO stability)
    agent_old.load_state_dict(agent.state_dict())
    agent_old.eval()

    # Optimizers
    policy_params = list(agent.parameters())
    # Always use Adam optimizer (TorsionAR policy)
    policy_opts = [torch.optim.Adam(policy_params, lr=config.lr)]

    return agent, agent_old, policy_opts, policy_params


