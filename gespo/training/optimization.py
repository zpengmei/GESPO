import math
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
from ..config import TrainingConfig


def _k3_kl(log_pi_new: torch.Tensor, log_pi_old: torch.Tensor,
           max_log_ratio: float = 10.0) -> torch.Tensor:
    """Schulman k3 estimator of KL(pi_old || pi_new) from samples drawn under pi_old.

    Returns mean_i (rho_i - log rho_i - 1) with rho_i = pi_new(y_i) / pi_old(y_i).
    Since rho - log rho - 1 >= 0 for every rho > 0, the estimate is nonnegative
    sample by sample, so no clamping of the mean is needed. It is unbiased for the
    same divergence as the plain mean of (log_pi_old - log_pi_new) it replaces,
    because E_{pi_old}[rho] = 1, and it has lower variance.

    The clamp on the log ratio is numerical protection for the exponential only;
    with clip_eps ~ 0.1 the ratios sit near 1 and it never binds.
    """
    log_ratio = (log_pi_new - log_pi_old).clamp(-max_log_ratio, max_log_ratio)
    return (log_ratio.exp() - log_ratio - 1.0).mean()


class GRPOOptimizer:
    """
    Handles the GRPO/PPO update loop.
    """
    def __init__(self, config: TrainingConfig, agent: nn.Module, agent_old: nn.Module, optimizers: List):
        self.config = config
        self.agent = agent
        self.agent_old = agent_old
        self.optimizers = optimizers
        
    def update(self, 
               start_data: Dict[str, Any],
               hill_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]: # TODO: deprecate hill_data
        """
        Run GRPO updates for both start and hill policies.
        
        Args:
            start_data: Dict containing:
                - features: Input features [B, ...]
                - actions: Actions taken [B, ...]
                - rewards: Rewards [B]
                - valid_mask: Boolean mask [B]
        """
        stats = {
            'loss': [], 'kl': [], 'entropy': [],
            'pg_s': [], 'pg_h': [], 'kl_s': [], 'kl_h': []
        }
        
        mode = start_data.get('mode', 'default')
        
        # Unpack Start Data
        s_rewards = start_data['rewards']
        s_mask = start_data['valid_mask']
        # Keep all start-policy update tensors on the policy device. In multi-GPU
        # workflows the rollout/mask tensors may live on a different CUDA device
        # than the policy network, which breaks boolean indexing.
        try:
            policy_device = next(self.agent.parameters()).device
        except Exception:
            policy_device = torch.device("cpu")
        s_rewards = torch.as_tensor(s_rewards, device=policy_device, dtype=torch.float32)
        s_mask = torch.as_tensor(s_mask, device=policy_device, dtype=torch.bool)

        if mode == 'torsion_ar':
            s_feats = None
            s_actions = None
        else:
            s_feats = start_data['features']
            s_actions = start_data['actions']
            if torch.is_tensor(s_feats) and s_feats.device != policy_device:
                s_feats = s_feats.to(device=policy_device)
            if torch.is_tensor(s_actions) and s_actions.device != policy_device:
                s_actions = s_actions.to(device=policy_device)
        
        # Normalize Start Advantages
        s_adv = s_rewards
        if self.config.norm_adv and s_adv[s_mask].std() > 1e-6:
            mean = s_adv[s_mask].mean()
            std = s_adv[s_mask].std()
            s_adv = (s_adv - mean) / (std + 1e-8)

        extra_stats = {}
        if s_mask.any().item():
            extra_stats['adv_s_mean'] = float(s_adv[s_mask].mean().item())
            extra_stats['adv_s_std'] = float(s_adv[s_mask].std().item())
        else:
            extra_stats['adv_s_mean'] = 0.0
            extra_stats['adv_s_std'] = 0.0
        

        if mode == 'torsion_ar':
            disable_start = False
        else:
            disable_start = (s_actions is None) or bool(
                getattr(self.config, "discrete_state_starts", False)
                and (getattr(self.config, "least_count_starts", False) or getattr(self.config, "uniform_starts", False))
            )

        with torch.no_grad():
            if mode == 'torsion_ar':
                torsions = start_data['torsions']
                residue_types = start_data.get('residue_types', None)
                angle_mask = start_data.get('angle_mask', None)
                global_values = start_data.get('global_values', None)
                old_log_pi_s_sum = self.agent_old.log_prob(
                    torsions,
                    residue_types=residue_types,
                    angle_mask=angle_mask,
                    global_values=global_values,
                )
                old_dist_s = None

            else:

                if getattr(self.config, "discrete_state_starts", False):
                    old_dist_s = self.agent_old(s_feats)
                    old_log_pi_s_sum = old_dist_s.log_prob(s_actions)
                else:
                    old_dist_s = self.agent_old.start_distribution(self.config.grp_size)
                    old_log_pi_s = old_dist_s.log_prob(s_actions)
                    old_log_pi_s_sum = old_log_pi_s.sum(dim=-1)
            

        for _ in range(self.config.grpo_iters):
            loss: Any = 0.0
            
            # --- Start Policy Update ---
            if mode == 'torsion_ar':
                torsions = start_data['torsions']
                residue_types = start_data.get('residue_types', None)
                angle_mask = start_data.get('angle_mask', None)
                global_values = start_data.get('global_values', None)

                log_pi_s_sum = self.agent.log_prob(
                    torsions,
                    residue_types=residue_types,
                    angle_mask=angle_mask,
                    global_values=global_values,
                )
                ratio_s = torch.exp(log_pi_s_sum - old_log_pi_s_sum)
                kl_s = _k3_kl(log_pi_s_sum, old_log_pi_s_sum)
                # Approximate entropy (nats/token) for torsion AR: use a differentiable
                # upper bound for the mixture entropy.
                try:
                    ent_seq = self.agent.mean_token_entropy(
                        torsions,
                        residue_types=residue_types,
                        angle_mask=angle_mask,
                        include_omega=False,
                        global_values=global_values,
                    )
                    ent_seq = torch.where(torch.isfinite(ent_seq), ent_seq, torch.zeros_like(ent_seq))
                    ent_s_mean = ent_seq[s_mask].mean() if s_mask.any().item() else torch.zeros((), device=log_pi_s_sum.device)
                except Exception:
                    ent_s_mean = torch.zeros((), device=log_pi_s_sum.device)
            else:
                if getattr(self.config, "discrete_state_starts", False):
                    # Discrete categorical start policy over states
                    dist_s = self.agent(s_feats)
                    log_pi_s = dist_s.log_prob(s_actions)  # [B]
                    try:
                        entropy_s = dist_s.entropy()
                    except NotImplementedError:
                        entropy_s = torch.zeros_like(log_pi_s)
                    log_pi_s_sum = log_pi_s
                    # Ratio and KL in log-prob space
                    ratio_s = torch.exp(log_pi_s_sum - old_log_pi_s_sum)
                    try:
                        from torch.distributions.kl import kl_divergence
                        kl_vals = kl_divergence(old_dist_s, dist_s)
                        kl_s = kl_vals.mean()
                    except Exception:
                        kl_s = _k3_kl(log_pi_s_sum, old_log_pi_s_sum)
                    ent_s_mean = entropy_s.mean()
                else:
                    dist_s = self.agent.start_distribution(self.config.grp_size)
                    log_pi_s = dist_s.log_prob(s_actions) # [B, D]
                    try:
                        entropy_s = dist_s.entropy()
                    except NotImplementedError:
                        # Some torch distributions (e.g., VonMises) don't implement entropy().
                        # For VonMises we can use a differentiable closed-form expression:
                        #   H = log(2π I0(κ)) - κ * I1(κ)/I0(κ)
                        try:
                            from torch.distributions import VonMises
                            if isinstance(dist_s, VonMises):
                                kappa = dist_s.concentration
                                kappa = torch.clamp(kappa, min=0.0)
                                i0e = torch.special.i0e(kappa)
                                i1e = torch.special.i1e(kappa)
                                denom = torch.clamp(i0e, min=1e-12)
                                ratio = i1e / denom
                                log_i0 = torch.log(denom) + kappa
                                entropy_s = (math.log(2.0 * math.pi) + log_i0 - kappa * ratio)
                            else:
                                entropy_s = torch.zeros_like(log_pi_s)
                        except Exception:
                            entropy_s = torch.zeros_like(log_pi_s)
                    
                    log_pi_s_sum = log_pi_s.sum(dim=-1)
                    ratio_s = torch.exp(log_pi_s_sum - old_log_pi_s_sum)
                    
                    try:
                        kl_s = torch.distributions.kl.kl_divergence(old_dist_s, dist_s).sum(dim=-1).mean()
                    except NotImplementedError:
                        kl_s = _k3_kl(log_pi_s_sum, old_log_pi_s_sum)
                        
                    ent_s_mean = entropy_s.sum(dim=-1).mean()

            if not disable_start:
                ratio_s_valid = ratio_s[s_mask]
                adv_s_valid = s_adv[s_mask]
                
                surr1_s = ratio_s_valid * adv_s_valid
                surr2_s = torch.clamp(ratio_s_valid, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * adv_s_valid
                pg_s = torch.min(surr1_s, surr2_s).mean()
                
                loss += -pg_s + self.config.kl_coef * kl_s - self.config.ent_coef * ent_s_mean
                
                stats['pg_s'].append(pg_s.item())
                stats['kl_s'].append(kl_s.item())
                stats['entropy'].append(ent_s_mean.item())

            for opt in self.optimizers:
                opt.zero_grad()

            if not isinstance(loss, float):
                loss.backward()
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.grad_clip)
                for opt in self.optimizers:
                    opt.step()
            
            if isinstance(loss, float):
                stats['loss'].append(float(loss))
            else:
                stats['loss'].append(loss.item())

        averaged = {k: sum(v)/len(v) for k, v in stats.items() if v}
        averaged.update(extra_stats)
        return averaged

    def get_state_dict(self) -> dict:
        """Get optimizer state for checkpointing."""
        return {
            'optimizers': [opt.state_dict() for opt in self.optimizers]
        }

    def load_state_dict(self, state: dict):
        """Restore optimizer state from checkpoint."""
        if 'optimizers' in state:
            for opt, opt_state in zip(self.optimizers, state['optimizers']):
                opt.load_state_dict(opt_state)
