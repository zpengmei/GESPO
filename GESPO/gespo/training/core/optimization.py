import math
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
from ...config import TrainingConfig

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
               hill_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Run GRPO updates for both start and hill policies.
        
        Args:
            start_data: Dict containing:
                - features: Input features [B, ...]
                - actions: Actions taken [B, ...]
                - rewards: Rewards [B]
                - valid_mask: Boolean mask [B]
                - use_transformer: bool
                - chan_mask: Optional mask for transformer [B, ...]
            hill_data: Optional dict containing:
                - features: Input features [B, D]
                - actions: Actions taken [B, A]
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
        # Only MLP / transformer start policies use s_feats / s_actions.
        if mode == 'torsion_ar':
            s_feats = None
            s_actions = None
            use_transformer = False
            chan_mask = None
        else:
            s_feats = start_data['features']
            s_actions = start_data['actions']
            use_transformer = start_data.get('use_transformer', False)
            chan_mask = start_data.get('chan_mask', None)
        
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
        
        # Unpack Hill Data
        h_feats, h_actions, h_adv, h_mask = None, None, None, None
        if hill_data:
            h_feats = hill_data['features']
            h_actions = hill_data['actions']
            h_rewards = hill_data['rewards']
            h_mask = hill_data['valid_mask']
            
            h_adv = h_rewards
            if self.config.norm_adv and h_adv[h_mask].std() > 1e-6:
                mean = h_adv[h_mask].mean()
                std = h_adv[h_mask].std()
                h_adv = (h_adv - mean) / (std + 1e-8)
            if h_mask.any().item():
                extra_stats['adv_h_mean'] = float(h_adv[h_mask].mean().item())
                extra_stats['adv_h_std'] = float(h_adv[h_mask].std().item())
            else:
                extra_stats['adv_h_mean'] = 0.0
                extra_stats['adv_h_std'] = 0.0

        # Experimental: in discrete-state least-count mode (or uniform), we
        # bypass the start policy entirely and should not attempt to
        # compute or update categorical start log-probs.
        if mode == 'torsion_ar':
            # Always enable start updates for torsion AR mode; masking is
            # handled via s_mask rather than disabling the head entirely.
            disable_start = False
        else:
            disable_start = (s_actions is None) or bool(
                getattr(self.config, "discrete_state_starts", False)
                and (getattr(self.config, "least_count_starts", False) or getattr(self.config, "uniform_starts", False))
            )

        # Pre-compute Old Log Probs (no grad)
        with torch.no_grad():
            # Start Head
            if disable_start:
                old_dist_s = None
                old_log_pi_s_sum = None
            elif mode == 'torsion_ar':
                # Sequence policy over torsions Θ; use direct log_prob.
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
            elif use_transformer:
                old_dist_s, old_pair_dist_s = self.agent_old(s_feats)
                # Torsion part
                old_log_pi_s = old_dist_s.log_prob(s_actions)
                if chan_mask is not None:
                    old_log_pi_s = old_log_pi_s * chan_mask
                
                # Sum torsion log probs [B]
                old_log_pi_s_sum = old_log_pi_s.sum(dim=(1, 2))
                
                # Pairwise part
                pair_actions = start_data.get('pair_actions', None)
                if old_pair_dist_s is not None and pair_actions is not None:
                    old_log_pi_p = old_pair_dist_s.log_prob(pair_actions)  # [B, num_residues, pair_dim]
                    old_log_pi_s_sum = old_log_pi_s_sum + old_log_pi_p.sum(dim=(1, 2))  # Sum over residues and pair_dim
                    
            else:
                # MLP / non-transformer start policy. Support both the
                # original continuous Normal start_distribution and an
                # optional discrete categorical start over projection states
                # states when discrete_state_starts is enabled.
                if getattr(self.config, "discrete_state_starts", False):
                    old_dist_s = self.agent_old(s_feats)
                    old_log_pi_s_sum = old_dist_s.log_prob(s_actions)
                else:
                    old_dist_s = self.agent_old.start_distribution(self.config.grp_size)
                    old_log_pi_s = old_dist_s.log_prob(s_actions)
                    old_log_pi_s_sum = old_log_pi_s.sum(dim=-1)
            
            # Hill Head
            if hill_data:
                old_dist_h = self.agent_old(h_feats)
                old_log_pi_h = old_dist_h.log_prob(h_actions)

        for _ in range(self.config.grpo_iters):
            # Start with a scalar loss and let the first policy term
            # promote it to a tensor with gradient history. If no term
            # is ever added (e.g., least-count starts with no hill
            # policy active), we skip backward entirely.
            loss: Any = 0.0
            
            # --- Start Policy Update ---
            if disable_start:
                # No start update (e.g., least-count or uniform discrete starts)
                device = None
                if isinstance(s_rewards, torch.Tensor):
                    device = s_rewards.device
                elif hasattr(self.agent, "parameters"):
                    try:
                        device = next(self.agent.parameters()).device
                    except StopIteration:
                        device = torch.device("cpu")
                if device is None:
                    device = torch.device("cpu")
                pg_s = torch.tensor(0.0, device=device)
                kl_s = torch.tensor(0.0, device=pg_s.device)
                ent_s_mean = torch.tensor(0.0, device=pg_s.device)
            elif mode == 'torsion_ar':
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
                # KL(old || new) = E_old[log_old - log_new]. In expectation this is
                # non-negative, but with finite samples it can dip slightly below
                # zero; clamp to avoid "negative KL" accidentally rewarding
                # divergence.
                kl_s = (old_log_pi_s_sum - log_pi_s_sum).mean().clamp_min(0.0)
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
            elif use_transformer:
                # Forward returns (torsion_dist, pair_dist)
                dist_s, pair_dist_s = self.agent(s_feats)
                
                # Torsion Log Probs
                # s_actions is [B, N_res, 4] (sin/cos)
                # If hybrid, s_actions might contain pair data in extra dims?
                # No, sampler loop should separate them.
                # We expect s_actions to be a tuple or dict if we have pairs?
                # Or we split s_actions?
                # Let's assume s_actions is just the torsion part for now, 
                # and we need to pass pair targets separately in start_data.
                
                torsion_actions = s_actions
                pair_actions = start_data.get('pair_actions', None)
                
                log_pi_s = dist_s.log_prob(torsion_actions)
                # Some distributions (e.g., VonMises in older PyTorch) do not
                # implement analytic entropy. Fall back to a zero contribution
                # in that case so training can proceed.
                try:
                    entropy_s = dist_s.entropy()
                except NotImplementedError:
                    entropy_s = torch.zeros_like(log_pi_s)
                if chan_mask is not None:
                    log_pi_s = log_pi_s * chan_mask
                    entropy_s = entropy_s * chan_mask
                    
                # Sum over dimensions (residues * 4)
                log_pi_s_sum = log_pi_s.sum(dim=(1, 2)) # [B]
                
                # Add Pairwise Log Probs if available
                if pair_dist_s is not None and pair_actions is not None:
                    log_pi_p = pair_dist_s.log_prob(pair_actions) # [B, n_residues, pair_dim]
                    entropy_p = pair_dist_s.entropy() # [B, n_residues, pair_dim]
                    
                    # Sum over residues and pair_dim
                    log_pi_s_sum = log_pi_s_sum + log_pi_p.sum(dim=(1, 2))
                    entropy_s_sum = entropy_s.sum(dim=(1, 2)) + entropy_p.sum(dim=(1, 2))
                    ent_s_mean = entropy_s_sum.mean()
                else:
                    ent_s_mean = entropy_s.sum(dim=(1, 2)).mean()

                # Ratio
                ratio_s = torch.exp(log_pi_s_sum - old_log_pi_s_sum)
                
                # KL Divergence (Approximate via log-prob difference)
                # KL(old || new) = E_old [ log_old - log_new ]
                kl_s = (old_log_pi_s_sum - log_pi_s_sum).mean().clamp_min(0.0)
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
                        kl_s = (old_log_pi_s_sum - log_pi_s_sum).mean()
                    ent_s_mean = entropy_s.mean()
                else:
                    dist_s = self.agent.start_distribution(self.config.grp_size)
                    log_pi_s = dist_s.log_prob(s_actions) # [B, D]
                    entropy_s = dist_s.entropy()
                    
                    log_pi_s_sum = log_pi_s.sum(dim=-1)
                    # Ratio
                    ratio_s = torch.exp(log_pi_s_sum - old_log_pi_s_sum)
                    
                    try:
                        kl_s = torch.distributions.kl.kl_divergence(old_dist_s, dist_s).sum(dim=-1).mean()
                    except NotImplementedError:
                        kl_s = (old_log_pi_s_sum - log_pi_s_sum).mean().clamp_min(0.0)
                        
                    ent_s_mean = entropy_s.sum(dim=-1).mean()

            # PPO Loss (Start)
            # Filter by mask
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

            # --- Hill Policy Update ---
            if hill_data:
                dist_h = self.agent(h_feats)
                log_pi_h = dist_h.log_prob(h_actions) # [B, A]
                entropy_h = dist_h.entropy()
                
                log_pi_h_sum = log_pi_h.sum(dim=-1)
                old_log_pi_h_sum = old_log_pi_h.sum(dim=-1)
                
                ratio_h = torch.exp(log_pi_h_sum - old_log_pi_h_sum)
                
                try:
                    kl_h = torch.distributions.kl.kl_divergence(old_dist_h, dist_h).sum(dim=-1).mean()
                except NotImplementedError:
                    kl_h = (old_log_pi_h_sum - log_pi_h_sum).mean().clamp_min(0.0)
                    
                ent_h_mean = entropy_h.sum(dim=-1).mean()
                
                # PPO Loss (Hill)
                ratio_h_valid = ratio_h[h_mask]
                adv_h_valid = h_adv[h_mask]
                
                surr1_h = ratio_h_valid * adv_h_valid
                surr2_h = torch.clamp(ratio_h_valid, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * adv_h_valid
                pg_h = torch.min(surr1_h, surr2_h).mean()
                
                loss += -pg_h + self.config.kl_coef * kl_h - self.config.ent_coef * ent_h_mean
                
                stats['pg_h'].append(pg_h.item())
                stats['kl_h'].append(kl_h.item())
                # Add to total entropy: if we already logged start-policy
                # entropy this iteration, accumulate into it; otherwise,
                # create a new entry for hill-only entropy (e.g. when
                # least_count_starts disables the start head).
                if stats['entropy']:
                    stats['entropy'][-1] += ent_h_mean.item()
                else:
                    stats['entropy'].append(ent_h_mean.item())

            # Backward
            for opt in self.optimizers:
                opt.zero_grad()
            # If loss is still a Python float, no head contributed a
            # differentiable term this iteration (e.g., least-count
            # starts with no hill policy); in that case, skip the
            # backward/step to avoid spurious autograd errors.
            if not isinstance(loss, float):
                loss.backward()
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.grad_clip)
                for opt in self.optimizers:
                    opt.step()

            # Enforce a minimum exploration floor on the start policy by
            # clamping standard deviations so the policy cannot fully collapse
            # to deterministic starts. Apply to both MLP and Transformer
            # variants when the corresponding parameters are present.
            with torch.no_grad():
                min_std = 0.2  # radians / unit in output space
                min_log_std = math.log(min_std)
                # MLP start head (GRPOAgent)
                if hasattr(self.agent, "log_std_s"):
                    self.agent.log_std_s.data.clamp_(min=min_log_std)
                    if hasattr(self.agent_old, "log_std_s"):
                        self.agent_old.log_std_s.data.copy_(self.agent.log_std_s.data)
                # Transformer torsion head
                if hasattr(self.agent, "log_std"):
                    self.agent.log_std.data.clamp_(min=min_log_std)
                    if hasattr(self.agent_old, "log_std"):
                        self.agent_old.log_std.data.copy_(self.agent.log_std.data)
                # Transformer pairwise head (if present)
                if hasattr(self.agent, "pair_log_std"):
                    self.agent.pair_log_std.data.clamp_(min=min_log_std)
                    if hasattr(self.agent_old, "pair_log_std"):
                        self.agent_old.pair_log_std.data.copy_(self.agent.pair_log_std.data)
            
            # Record scalar loss for logging; handle both tensor and float.
            if isinstance(loss, float):
                stats['loss'].append(float(loss))
            else:
                stats['loss'].append(loss.item())

        # Average stats and merge with advantage summaries
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
