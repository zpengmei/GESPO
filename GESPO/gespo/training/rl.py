"""Reinforcement learning policies for proposing metadynamics hills or CV targets."""
from .._imports import *
from ..models.transformer import PositionalEncoding, TransformerEncoder
from torch.distributions import VonMises, LogNormal, Categorical, Normal
import torch.nn.functional as F

class GRPOAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, num_mixtures: int, hidden_dim: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.0),  # Dropout disabled for stability
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        action_size_hill = num_mixtures * (1 + action_dim) 
        self.hill_params = nn.Linear(hidden_dim, action_size_hill)
        self.log_stds = nn.Parameter(torch.ones(action_size_hill) * -0.7) 

        self.start_mu = nn.Parameter(torch.zeros(action_dim))
        self.log_std_s = nn.Parameter(torch.full((action_dim,), 3.0))
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.body.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None: nn.init.constant_(module.bias, 0)
        nn.init.uniform_(self.hill_params.weight, -3e-3, 3e-3)
        if self.hill_params.bias is not None: nn.init.constant_(self.hill_params.bias, 0)

    def forward(self, s: torch.Tensor) -> Normal:
        h = self.body(s)
        mus = self.hill_params(h)
        # Clamp log_stds before exp to prevent overflow
        log_stds_clamped = torch.clamp(self.log_stds, min=-10.0, max=5.0)
        stds = torch.exp(log_stds_clamped)
        # Guard against numerical issues: ensure finite, non-negative std
        stds = torch.where(torch.isfinite(stds), stds, torch.full_like(stds, 1.0))
        stds = torch.clamp(stds, min=1e-6, max=1e3)
        return Normal(mus, stds)

    @torch.no_grad()
    def sample_start(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = torch.exp(self.log_std_s)
        scale = torch.where(torch.isfinite(scale), scale, torch.full_like(scale, 1.0))
        scale = torch.clamp(scale, min=1e-6)
        dist = Normal(self.start_mu.expand(batch_size, -1), scale.expand(batch_size, -1))
        raw = dist.sample()
        wrapped = torch.remainder(raw + math.pi, 2 * math.pi) - math.pi
        logp = dist.log_prob(raw).sum(-1)
        return wrapped, logp

    def start_distribution(self, batch_size: int) -> Normal:
        """Return the Normal distribution used for start CVs for a given batch size.

        Exposes the per-batch start distribution so callers can sample raw values
        (with gradient) and compute cross-log-probs under a reference policy.
        """
        scale = torch.exp(self.log_std_s)
        scale = torch.where(torch.isfinite(scale), scale, torch.full_like(scale, 1.0))
        scale = torch.clamp(scale, min=1e-6)
        return Normal(self.start_mu.expand(batch_size, -1), scale.expand(batch_size, -1))


class StateLogitPolicy(nn.Module):
    """
    State-conditioned categorical policy over a fixed number of discrete
    states (e.g., geom2vec soft states). Used for runs where we only
    want to decide which basin/state to start from, with the actual
    coordinates drawn from history.
    """
    def __init__(self, num_states: int, feature_dim: Optional[int] = None, hidden_dim: int = 128):
        super().__init__()
        self.num_states = num_states
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        if feature_dim is None:
            # Global logits over states; ignores features entirely.
            self.logits = nn.Parameter(torch.zeros(num_states, dtype=torch.float32))
            self.net = None
        else:
            # Simple MLP that maps flat features -> logits over states.
            self.logits = None
            self.net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, num_states),
            )
            # Initialize to an exactly uniform policy: all logits = 0
            # so that softmax(logits) = 1 / num_states for every state.
            for module in self.net:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            last_linear = self.net[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)

    def forward(self, features: torch.Tensor) -> Categorical:
        """
        Args:
            features: Start features [B, F]. In the backward-compatible
                      path (feature_dim is None), only the batch dimension
                      is used to expand a global logits vector.
        Returns:
            Categorical distribution over states for each batch element.
        """
        if features.dim() == 0:
            batch_size = 1
        else:
            batch_size = int(features.shape[0])

        if self.net is None:
            # Global, state-independent logits.
            logits = self.logits.unsqueeze(0).expand(batch_size, -1)
        else:
            # Use the provided features to compute per-state logits.
            if features.dim() > 2:
                # Flatten any trailing dimensions after batch.
                features = features.view(batch_size, -1)
            logits = self.net(features)

        return Categorical(logits=logits)


def mixture_vonmises_log_prob(x: torch.Tensor,
                              logits: torch.Tensor,
                              mu: torch.Tensor,
                              kappa: torch.Tensor) -> torch.Tensor:
    """
    Mixture of von Mises log-probability for angular data.

    Args:
        x:      [...], angles in radians.
        logits: [..., K] unnormalized mixture logits.
        mu:     [..., K] mixture means in radians.
        kappa:  [..., K] positive concentrations.

    Returns:
        log_prob: [...] total log probability under the mixture.
    """
    logw = F.log_softmax(logits, dim=-1)  # [..., K]
    vm = VonMises(mu, kappa)              # broadcastable over leading dims
    lp = vm.log_prob(x.unsqueeze(-1))     # [..., K]
    return torch.logsumexp(logw + lp, dim=-1)  # [...]


def mixture_normal_log_prob(
    x: torch.Tensor,
    logits: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Mixture of Normal log-probability for scalar data.

    Args:
        x:      [...], real-valued targets.
        logits: [..., K] unnormalized mixture logits.
        mu:     [..., K] mixture means.
        sigma:  [..., K] positive stddevs.

    Returns:
        log_prob: [...] total log probability under the mixture.
    """
    logw = F.log_softmax(logits, dim=-1)
    sigma = torch.clamp(sigma, min=1e-6)
    norm = Normal(mu, sigma)
    lp = norm.log_prob(x.unsqueeze(-1))  # [..., K]
    return torch.logsumexp(logw + lp, dim=-1)  # [...]


class TorsionARPolicy(nn.Module):
    """
    Autoregressive transformer policy over backbone torsions Θ = {(φ_i, ψ_i, ω_i)}_{i=1..L}.

    - Factorization: π(Θ) = Π_i π(φ_i, ψ_i, ω_i | Θ_<i, seq)
    - Angles are modeled as independent mixtures of K von Mises per residue.
    - Supports teacher-forced log_prob() for pretraining and sequential sampling().
    """

    def __init__(
        self,
        num_residues: int,
        residue_vocab_size: int = 21,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        kmix: int = 8,
        n_global_tokens: int = 0,
        global_kmix: Optional[int] = None,
    ):
        super().__init__()
        self.num_residues = num_residues
        self.residue_vocab_size = residue_vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.kmix = kmix
        self.n_global_tokens = int(n_global_tokens) if n_global_tokens is not None else 0
        if self.n_global_tokens < 0:
            raise ValueError(f"n_global_tokens must be >= 0, got {self.n_global_tokens}")
        self.global_kmix = int(global_kmix) if global_kmix is not None else int(kmix)

        # Residue type and positional embeddings
        self.residue_emb = nn.Embedding(residue_vocab_size, d_model)
        prefix_len = 0
        if self.n_global_tokens > 0:
            # Prefix tokens: [BOS] + global scalar tokens
            prefix_len = 1 + self.n_global_tokens
            self.special_emb = nn.Embedding(prefix_len, d_model)
            self.global_value_proj = nn.Sequential(
                nn.Linear(1, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            # Mixture Normal head for global tokens (in transformed space).
            self.global_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 3 * self.global_kmix),
            )
        self.pos_enc = PositionalEncoding(d_model, max_len=num_residues + prefix_len)

        # Previous-torsion embedding: [sinφ, cosφ, sinψ, cosψ, sinω, cosω, mφ, mψ, mω] → d_model
        # where m* are binary mask bits indicating whether each angle is defined.
        self.torsion_proj = nn.Linear(9, d_model)

        # Causal transformer encoder (GPT-style via attention mask)
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=4 * d_model,
            dropout=dropout,
        )

        # Shared head that outputs mixture params for all three angles per residue:
        # 3 angles × 3*K parameters (logits_K, mu_K, kappa_raw_K) = 9*K total.
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 9 * kmix),
        )

        self._reset_parameters()
        self.register_buffer("_causal_mask", None, persistent=False)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.torsion_proj.weight)
        nn.init.zeros_(self.torsion_proj.bias)
        if self.n_global_tokens > 0:
            for module in self.global_head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            for module in self.global_value_proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Build (or reuse) an upper-triangular causal mask of shape [L, L]
        where True entries are masked (no attention to future positions).
        """
        mask = self._causal_mask
        if mask is None or mask.size(0) != seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            self._causal_mask = mask
        return mask

    def _globals_to_y(self, globals_raw: torch.Tensor) -> torch.Tensor:
        """
        Map raw nonnegative global scalars -> unconstrained space via log1p.
        """
        globals_raw = torch.as_tensor(globals_raw)
        globals_raw = torch.clamp(globals_raw, min=0.0)
        return torch.log1p(globals_raw)
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    def _encode_full_context(
        self,
        torsions_shifted: torch.Tensor,
        residue_types: Optional[torch.Tensor] = None,
        angle_mask_shifted: Optional[torch.Tensor] = None,
        global_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode teacher-forced context sequence, optionally with global prefix tokens.

        When `n_global_tokens>0`, the transformer input becomes:
          [BOS, global_1, ..., global_G, residue_1, ..., residue_L]
        Global tokens embed *their own* values (transformed) so they can condition
        torsion prediction. Global-token log-probs are computed from earlier hidden
        states to avoid leakage.
        """
        if residue_types is None:
            B, L, _ = torsions_shifted.shape
            residue_types = torsions_shifted.new_zeros((B, L), dtype=torch.long)
        if angle_mask_shifted is None:
            angle_mask_shifted = torch.ones_like(torsions_shifted, dtype=torch.bool)

        # Encode torsions as sin/cos per angle and concatenate mask bits
        sin_cos = torch.stack([torch.sin(torsions_shifted), torch.cos(torsions_shifted)], dim=-1)  # [B, L, 3, 2]
        sin_cos = sin_cos.view(sin_cos.shape[0], sin_cos.shape[1], 6)  # [B, L, 6]
        mask_feat = angle_mask_shifted.to(sin_cos.dtype)  # [B, L, 3]
        torsion_input = torch.cat([sin_cos, mask_feat], dim=-1)  # [B, L, 9]
        torsion_emb = self.torsion_proj(torsion_input)  # [B, L, d_model]

        res_emb = self.residue_emb(residue_types)  # [B, L, d_model]
        x_res = res_emb + torsion_emb

        if self.n_global_tokens > 0:
            B = int(x_res.shape[0])
            device = x_res.device
            if global_values is None:
                global_values = torch.zeros((B, self.n_global_tokens), dtype=torch.float32, device=device)
            else:
                global_values = torch.as_tensor(global_values, dtype=torch.float32, device=device)
                if global_values.ndim != 2 or int(global_values.shape[1]) != int(self.n_global_tokens):
                    raise ValueError(
                        f"global_values must be [B,{self.n_global_tokens}], got {tuple(global_values.shape)}"
                    )

            y = self._globals_to_y(global_values)  # [B, G]
            y_in = y.unsqueeze(-1)  # [B, G, 1]
            glob_val_emb = self.global_value_proj(y_in)  # [B, G, d_model]

            prefix_len = 1 + self.n_global_tokens
            # [BOS] type embedding (index 0)
            bos_idx = torch.zeros((B, 1), dtype=torch.long, device=device)
            bos_emb = self.special_emb(bos_idx)  # [B,1,d_model]

            # Global token type embeddings (indices 1..G)
            glob_type_idx = torch.arange(1, 1 + self.n_global_tokens, dtype=torch.long, device=device).view(1, -1)
            glob_type_idx = glob_type_idx.expand(B, -1)  # [B,G]
            glob_type_emb = self.special_emb(glob_type_idx)  # [B,G,d_model]

            x = torch.cat([bos_emb, glob_type_emb + glob_val_emb, x_res], dim=1)  # [B, 1+G+L, d_model]
        else:
            x = x_res

        x = self.pos_enc(x)
        attn_mask = self._build_causal_mask(x.size(1), x.device)
        h = self.encoder(x, key_padding_mask=None, attn_mask=attn_mask)
        return h

    def _encode_context(
        self,
        torsions_shifted: torch.Tensor,
        residue_types: Optional[torch.Tensor] = None,
        angle_mask_shifted: Optional[torch.Tensor] = None,
        global_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode teacher-forced context sequence.

        Args:
            torsions_shifted: [B, L, 3] previous torsions per residue (shifted right).
            residue_types:    [B, L] residue type indices, or None (all zeros).
            angle_mask_shifted: [B, L, 3] mask bits for shifted context, or None (all ones).
        Returns:
            h: [B, L, d_model] contextual embeddings.
        """
        h_full = self._encode_full_context(
            torsions_shifted,
            residue_types=residue_types,
            angle_mask_shifted=angle_mask_shifted,
            global_values=global_values,
        )
        if self.n_global_tokens > 0:
            prefix_len = 1 + self.n_global_tokens
            return h_full[:, prefix_len:, :]
        return h_full

    def _global_params(
        self,
        h_full: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Map hidden states to mixture params for global scalar tokens in log1p-space.

        Global token j is predicted from hidden state at prefix position j:
          j=0 -> BOS predicts global token 0
          j=1 -> global token 0 predicts global token 1
          ...

        Returns:
            logits, mu, sigma each of shape [B, G, K]
        """
        if self.n_global_tokens <= 0:
            raise RuntimeError("_global_params called but n_global_tokens=0")
        B, T, _ = h_full.shape
        G = int(self.n_global_tokens)
        if T < 1 + G:
            raise ValueError(f"Expected full context length >= {1+G}, got {T}")
        h_src = h_full[:, :G, :]  # [B, G, d_model] (BOS, global0, ...)
        out = self.global_head(h_src)  # [B, G, 3*K]
        K = int(self.global_kmix)
        logits = out[..., 0 * K : 1 * K]
        mu = out[..., 1 * K : 2 * K]
        log_sigma = out[..., 2 * K : 3 * K]
        sigma = F.softplus(log_sigma) + 1e-3
        sigma = torch.clamp(sigma, min=1e-6, max=50.0)
        return logits, mu, sigma

    def _angle_params(
        self,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Map hidden states to mixture params for (φ, ψ, ω).

        Returns:
            logits_phi, mu_phi, kappa_phi,
            logits_psi, mu_psi, kappa_psi,
            logits_omega, mu_omega, kappa_omega
        Each of shape [B, L, K].
        """
        B, L, _ = h.shape
        out = self.head(h)  # [B, L, 9*K]
        K = self.kmix

        # Slice per-angle segments
        phi_logits = out[..., 0 * K : 1 * K]
        phi_mu     = out[..., 1 * K : 2 * K]
        phi_kappa  = out[..., 2 * K : 3 * K]

        psi_logits = out[..., 3 * K : 4 * K]
        psi_mu     = out[..., 4 * K : 5 * K]
        psi_kappa  = out[..., 5 * K : 6 * K]

        omg_logits = out[..., 6 * K : 7 * K]
        omg_mu     = out[..., 7 * K : 8 * K]
        omg_kappa  = out[..., 8 * K : 9 * K]

        # Concentrations must be positive; clamp to avoid degenerate or overly sharp peaks.
        phi_kappa = F.softplus(phi_kappa) + 1.0
        psi_kappa = F.softplus(psi_kappa) + 1.0
        omg_kappa = F.softplus(omg_kappa) + 1.0

        phi_kappa = torch.clamp(phi_kappa, max=200.0)
        psi_kappa = torch.clamp(psi_kappa, max=200.0)
        omg_kappa = torch.clamp(omg_kappa, max=200.0)

        # Wrap means into [-π, π] to keep them in canonical range
        pi = math.pi
        phi_mu = (phi_mu + pi) % (2 * pi) - pi
        psi_mu = (psi_mu + pi) % (2 * pi) - pi
        omg_mu = (omg_mu + pi) % (2 * pi) - pi

        return (
            phi_logits, phi_mu, phi_kappa,
            psi_logits, psi_mu, psi_kappa,
            omg_logits, omg_mu, omg_kappa,
        )

    def log_prob(
        self,
        torsions: torch.Tensor,
        residue_types: Optional[torch.Tensor] = None,
        angle_mask: Optional[torch.Tensor] = None,
        global_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log π(Θ) for a batch of torsion sequences using teacher forcing.

        Args:
            torsions:      [B, L, 3] angles (φ, ψ, ω) in radians.
            residue_types: [B, L] residue indices (optional).
            angle_mask:    optional [B, L, 3] bool mask (True = valid angle).

        Returns:
            log_probs: [B] total log-probability per sequence.
        """
        if torsions.ndim != 3 or torsions.shape[2] != 3:
            raise ValueError("torsions must have shape [B, L, 3] (φ, ψ, ω)")

        B, L, _ = torsions.shape
        if L != self.num_residues:
            raise ValueError(f"Expected torsions length {self.num_residues}, got {L}")

        # Teacher forcing: shift torsions and masks right by one residue
        torsions_shift = torsions.clone()
        torsions_shift[:, 1:, :] = torsions[:, :-1, :]
        torsions_shift[:, 0, :] = 0.0

        if angle_mask is None:
            angle_mask = torch.ones_like(torsions, dtype=torch.bool)
        if angle_mask.shape != torsions.shape:
            raise ValueError("angle_mask must have same shape as torsions [B, L, 3]")

        angle_mask_shift = angle_mask.clone()
        angle_mask_shift[:, 1:, :] = angle_mask[:, :-1, :]
        angle_mask_shift[:, 0, :] = False

        h_full = self._encode_full_context(
            torsions_shift,
            residue_types=residue_types,
            angle_mask_shifted=angle_mask_shift,
            global_values=global_values,
        )
        h = h_full
        if self.n_global_tokens > 0:
            prefix_len = 1 + self.n_global_tokens
            h = h_full[:, prefix_len:, :]
        (
            phi_logits, phi_mu, phi_kappa,
            psi_logits, psi_mu, psi_kappa,
            omg_logits, omg_mu, omg_kappa,
        ) = self._angle_params(h)

        phi = torsions[..., 0]
        psi = torsions[..., 1]
        omg = torsions[..., 2]

        lp_phi = mixture_vonmises_log_prob(phi, phi_logits, phi_mu, phi_kappa)
        lp_psi = mixture_vonmises_log_prob(psi, psi_logits, psi_mu, psi_kappa)
        lp_omg = mixture_vonmises_log_prob(omg, omg_logits, omg_mu, omg_kappa)

        mask_phi = angle_mask[..., 0].to(lp_phi.dtype)
        mask_psi = angle_mask[..., 1].to(lp_psi.dtype)
        mask_omg = angle_mask[..., 2].to(lp_omg.dtype)

        # Sum over residues and angles where masks are active
        torsion_lp = (lp_phi * mask_phi + lp_psi * mask_psi + lp_omg * mask_omg).sum(dim=1)  # [B]

        if self.n_global_tokens > 0:
            if global_values is None:
                global_values = torsions.new_zeros((B, self.n_global_tokens), dtype=torch.float32)
            y = self._globals_to_y(global_values.to(dtype=torch.float32, device=torsions.device))  # [B, G]
            gl_logits, gl_mu, gl_sigma = self._global_params(h_full)
            lp_g = mixture_normal_log_prob(y, gl_logits, gl_mu, gl_sigma)  # [B, G]
            global_lp = lp_g.sum(dim=1)  # [B]
            return torsion_lp + global_lp

        return torsion_lp  # [B]

    def _vonmises_entropy(self, kappa: torch.Tensor) -> torch.Tensor:
        """
        Differential entropy of VonMises(μ, κ) (independent of μ).

        H = log(2π I0(κ)) - κ * I1(κ) / I0(κ)

        Uses exponentially scaled Bessel functions (i0e/i1e) for numerical stability.
        """
        kappa = torch.clamp(kappa, min=1e-6)
        i0e = torch.special.i0e(kappa)
        i1e = torch.special.i1e(kappa)
        i0e = torch.clamp(i0e, min=1e-12)
        ratio = i1e / i0e
        log_i0 = torch.log(i0e) + kappa  # log(I0) = log(i0e) + kappa
        return math.log(2.0 * math.pi) + log_i0 - kappa * ratio

    def mean_token_entropy(
        self,
        torsions: torch.Tensor,
        residue_types: Optional[torch.Tensor] = None,
        angle_mask: Optional[torch.Tensor] = None,
        include_omega: bool = True,
        global_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Approximate per-sequence mean token entropy (nats/token) for the torsion AR policy.

        We use a differentiable upper bound on the mixture entropy:
          H(mixture) <= H(weights) + E_w[ H(VonMises(κ)) ].

        Args:
            torsions:      [B, L, 3] angles (φ, ψ, ω) in radians.
            residue_types: [B, L] residue indices (optional).
            angle_mask:    [B, L, 3] bool mask (True = valid).
            include_omega: if False, compute entropy over φ/ψ only.

        Returns:
            mean_ent: [B] mean token entropy per sequence (averaged over valid tokens).
        """
        if torsions.ndim != 3 or torsions.shape[2] != 3:
            raise ValueError("torsions must have shape [B, L, 3] (φ, ψ, ω)")

        B, L, _ = torsions.shape
        if L != self.num_residues:
            raise ValueError(f"Expected torsions length {self.num_residues}, got {L}")

        if angle_mask is None:
            angle_mask = torch.ones_like(torsions, dtype=torch.bool)
        if angle_mask.shape != torsions.shape:
            raise ValueError("angle_mask must have same shape as torsions [B, L, 3]")

        torsions_shift = torsions.clone()
        torsions_shift[:, 1:, :] = torsions[:, :-1, :]
        torsions_shift[:, 0, :] = 0.0

        angle_mask_shift = angle_mask.clone()
        angle_mask_shift[:, 1:, :] = angle_mask[:, :-1, :]
        angle_mask_shift[:, 0, :] = False

        h_full = self._encode_full_context(
            torsions_shift,
            residue_types=residue_types,
            angle_mask_shifted=angle_mask_shift,
            global_values=global_values,
        )
        h = h_full
        if self.n_global_tokens > 0:
            prefix_len = 1 + self.n_global_tokens
            h = h_full[:, prefix_len:, :]
        (
            phi_logits, _, phi_kappa,
            psi_logits, _, psi_kappa,
            omg_logits, _, omg_kappa,
        ) = self._angle_params(h)

        def _mixture_entropy_upper(logits: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
            w = torch.softmax(logits, dim=-1)
            logw = torch.log_softmax(logits, dim=-1)
            ent_w = -(w * logw).sum(dim=-1)  # [B, L]
            ent_vm = self._vonmises_entropy(kappa)  # [B, L, K]
            ent_x_upper = ent_w + (w * ent_vm).sum(dim=-1)  # [B, L]
            return ent_x_upper

        ent_phi = _mixture_entropy_upper(phi_logits, phi_kappa)
        ent_psi = _mixture_entropy_upper(psi_logits, psi_kappa)
        ent_omg = _mixture_entropy_upper(omg_logits, omg_kappa)

        mask_phi = angle_mask[..., 0].to(ent_phi.dtype)
        mask_psi = angle_mask[..., 1].to(ent_psi.dtype)
        mask_omg = angle_mask[..., 2].to(ent_omg.dtype) if include_omega else torch.zeros_like(mask_phi)

        total_ent = (ent_phi * mask_phi + ent_psi * mask_psi + ent_omg * mask_omg).sum(dim=1)  # [B]
        total_tokens = (mask_phi + mask_psi + mask_omg).sum(dim=1).clamp_min(1.0)  # [B]

        if self.n_global_tokens > 0:
            # Global-token entropy upper bound: H(w) + E_w[H(N(σ))].
            try:
                gl_logits, _, gl_sigma = self._global_params(h_full)
                w = torch.softmax(gl_logits, dim=-1)
                logw = torch.log_softmax(gl_logits, dim=-1)
                ent_w = -(w * logw).sum(dim=-1)  # [B, G]
                ent_n = 0.5 * (math.log(2.0 * math.pi * math.e) + 2.0 * torch.log(gl_sigma))  # [B,G,K]
                ent_g = ent_w + (w * ent_n).sum(dim=-1)  # [B,G]
                total_ent = total_ent + ent_g.sum(dim=1)
                total_tokens = total_tokens + float(self.n_global_tokens)
            except Exception:
                pass

        mean_ent = total_ent / total_tokens
        mean_ent = torch.where(torch.isfinite(mean_ent), mean_ent, torch.zeros_like(mean_ent))
        return mean_ent

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        residue_types: Optional[torch.Tensor] = None,
        angle_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        global_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressively sample torsion sequences Θ ~ π(·).

        Args:
            batch_size: number of sequences to sample.
            residue_types: optional [B, L] residue indices; if None, zeros.
            angle_mask: optional [B, L, 3] bool mask for valid angles.
            temperature: sampling temperature applied to mixture logits.

        Returns:
            torsions: [B, L, 3] sampled angles (φ, ψ, ω) in radians.
        """
        B = batch_size
        L = self.num_residues
        device = next(self.parameters()).device

        if residue_types is None:
            residue_types = torch.zeros((B, L), dtype=torch.long, device=device)
        if angle_mask is None:
            angle_mask = torch.ones((B, L, 3), dtype=torch.bool, device=device)

        torsions = torch.zeros((B, L, 3), dtype=torch.float32, device=device)
        torsions_shift = torch.zeros_like(torsions)

        # Build a static shifted mask so that the transformer can
        # distinguish termini/undefined angles in the context.
        angle_mask_shift = angle_mask.clone()
        angle_mask_shift[:, 1:, :] = angle_mask[:, :-1, :]
        angle_mask_shift[:, 0, :] = False

        for i in range(L):
            h = self._encode_context(
                torsions_shift,
                residue_types,
                angle_mask_shifted=angle_mask_shift,
                global_values=global_values,
            )
            (
                phi_logits, phi_mu, phi_kappa,
                psi_logits, psi_mu, psi_kappa,
                omg_logits, omg_mu, omg_kappa,
            ) = self._angle_params(h)

            # Select parameters for position i
            phi_logits_i = phi_logits[:, i, :] / max(temperature, 1e-3)
            psi_logits_i = psi_logits[:, i, :] / max(temperature, 1e-3)
            omg_logits_i = omg_logits[:, i, :] / max(temperature, 1e-3)

            phi_mu_i = phi_mu[:, i, :]
            psi_mu_i = psi_mu[:, i, :]
            omg_mu_i = omg_mu[:, i, :]

            phi_kappa_i = phi_kappa[:, i, :]
            psi_kappa_i = psi_kappa[:, i, :]
            omg_kappa_i = omg_kappa[:, i, :]

            # Sample mixture component indices
            cat_phi = Categorical(logits=phi_logits_i)
            cat_psi = Categorical(logits=psi_logits_i)
            cat_omg = Categorical(logits=omg_logits_i)

            idx_phi = cat_phi.sample()  # [B]
            idx_psi = cat_psi.sample()  # [B]
            idx_omg = cat_omg.sample()  # [B]

            def _gather_params(mu_all, kappa_all, idx):
                gather_idx = idx.unsqueeze(-1)
                mu_sel = torch.gather(mu_all, dim=-1, index=gather_idx).squeeze(-1)
                kappa_sel = torch.gather(kappa_all, dim=-1, index=gather_idx).squeeze(-1)
                return mu_sel, kappa_sel

            mu_phi_i, kappa_phi_i = _gather_params(phi_mu_i, phi_kappa_i, idx_phi)
            mu_psi_i, kappa_psi_i = _gather_params(psi_mu_i, psi_kappa_i, idx_psi)
            mu_omg_i, kappa_omg_i = _gather_params(omg_mu_i, omg_kappa_i, idx_omg)

            vm_phi = VonMises(mu_phi_i, kappa_phi_i)
            vm_psi = VonMises(mu_psi_i, kappa_psi_i)
            vm_omg = VonMises(mu_omg_i, kappa_omg_i)

            phi_sample = vm_phi.sample()
            psi_sample = vm_psi.sample()
            omg_sample = vm_omg.sample()

            # Respect angle masks at this position: if invalid, keep zero.
            valid_phi = angle_mask[:, i, 0]
            valid_psi = angle_mask[:, i, 1]
            valid_omg = angle_mask[:, i, 2]

            phi_sample = torch.where(valid_phi, phi_sample, torsions[:, i, 0])
            psi_sample = torch.where(valid_psi, psi_sample, torsions[:, i, 1])
            omg_sample = torch.where(valid_omg, omg_sample, torsions[:, i, 2])

            torsions[:, i, 0] = phi_sample
            torsions[:, i, 1] = psi_sample
            torsions[:, i, 2] = omg_sample

            # Update shifted context for next residue
            if i + 1 < L:
                torsions_shift[:, i + 1, :] = torsions[:, i, :]

        return torsions

    @torch.no_grad()
    def sample_with_globals(
        self,
        batch_size: int,
        residue_types: Optional[torch.Tensor] = None,
        angle_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        global_temperature: float = 1.0,
        global_ranges: Optional[torch.Tensor] = None,
        max_rejection_iters: int = 100,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample (global RMSD/E2E, then torsions) when global tokens are enabled.

        Args:
            batch_size: Number of samples to generate
            residue_types: [B, L] residue type indices
            angle_mask: [B, L, 3] mask for which angles exist
            temperature: Temperature for torsion sampling
            global_temperature: Temperature for global token sampling
            global_ranges: [B, G, 2] optional (min, max) ranges for each global token in nm.
                          If provided, uses rejection sampling to enforce constraints.
                          Shape: [batch_size, n_global_tokens, 2] where [..., i, 0] = min, [..., i, 1] = max
            max_rejection_iters: Maximum rejection sampling attempts per token

        Returns:
            torsions: [B, L, 3]
            globals:  [B, G] in raw space (nm), or None if G==0
        """
        if self.n_global_tokens <= 0:
            tors = self.sample(
                batch_size=batch_size,
                residue_types=residue_types,
                angle_mask=angle_mask,
                temperature=temperature,
                global_values=None,
            )
            return tors, None

        B = int(batch_size)
        L = int(self.num_residues)
        device = next(self.parameters()).device
        if residue_types is None:
            residue_types = torch.zeros((B, L), dtype=torch.long, device=device)
        if angle_mask is None:
            angle_mask = torch.ones((B, L, 3), dtype=torch.bool, device=device)

        # Sample globals sequentially in y=log1p(space), then map to raw nm.
        globals_raw = torch.zeros((B, self.n_global_tokens), dtype=torch.float32, device=device)
        torsions_shift = torch.zeros((B, L, 3), dtype=torch.float32, device=device)
        angle_mask_shift = angle_mask.clone()
        angle_mask_shift[:, 1:, :] = angle_mask[:, :-1, :]
        angle_mask_shift[:, 0, :] = False

        # Check if we have range constraints for this batch
        use_ranges = global_ranges is not None and global_ranges.shape[0] == B and global_ranges.shape[1] >= self.n_global_tokens
        
        for j in range(self.n_global_tokens):
            h_full = self._encode_full_context(
                torsions_shift,
                residue_types=residue_types,
                angle_mask_shifted=angle_mask_shift,
                global_values=globals_raw,
            )
            h_src = h_full[:, j, :]  # BOS (j=0) or global0 (j=1)
            out = self.global_head(h_src)  # [B, 3*K]
            K = int(self.global_kmix)
            logits = out[:, 0 * K : 1 * K] / max(float(global_temperature), 1e-3)
            mu = out[:, 1 * K : 2 * K]
            log_sigma = out[:, 2 * K : 3 * K]
            sigma = F.softplus(log_sigma) + 1e-3
            sigma = torch.clamp(sigma, min=1e-6, max=50.0)

            # Sample for each item in batch, with optional rejection sampling for range constraints
            raw_samples = torch.zeros(B, dtype=torch.float32, device=device)
            
            for b in range(B):
                # Extract range constraints for this batch item and global token
                if use_ranges:
                    range_min = float(global_ranges[b, j, 0].item())
                    range_max = float(global_ranges[b, j, 1].item())
                    has_constraint = torch.isfinite(global_ranges[b, j, 0]) and torch.isfinite(global_ranges[b, j, 1]) and range_min < range_max
                else:
                    has_constraint = False
                
                accepted = False
                for attempt in range(max_rejection_iters if has_constraint else 1):
                    cat = Categorical(logits=logits[b:b+1])
                    idx = cat.sample()  # [1]
                    mu_sel = mu[b, idx]
                    sigma_sel = sigma[b, idx]
                    y_samp = Normal(mu_sel, sigma_sel).sample()
                    raw = torch.expm1(y_samp)
                    raw = torch.clamp(raw, min=0.0)
                    
                    if has_constraint:
                        if range_min <= raw.item() <= range_max:
                            raw_samples[b] = raw
                            accepted = True
                            break
                    else:
                        raw_samples[b] = raw
                        accepted = True
                        break
                
                # If rejection sampling failed, fall back to clamping to range
                if not accepted and has_constraint:
                    raw_samples[b] = torch.clamp(raw, min=range_min, max=range_max)
            
            globals_raw[:, j] = raw_samples

        torsions = self.sample(
            batch_size=batch_size,
            residue_types=residue_types,
            angle_mask=angle_mask,
            temperature=temperature,
            global_values=globals_raw,
        )
        return torsions, globals_raw


class ResidueTransformerPolicy(nn.Module):
    """Transformer-based policy for per-residue torsion and pairwise distance prediction.
    
    Architecture Overview:
    =====================
    This policy maintains per-residue structure throughout for consistency with input features.
    
    Input:  [B, num_residues, feature_dim]
        - feature_dim = 4 (torsions) + max_partners (pairwise) in hybrid mode
        - Each residue has its own feature vector with local context
    
    Output: Two distributions over per-residue predictions
        1. torsion_dist: Normal([B, num_residues, 4])
           - Predicts (sin φ, cos φ, sin ψ, cos ψ) per residue
        2. pair_dist: Normal([B, num_residues, max_partners]) or None
           - Predicts pairwise distances to neighboring residues (if hybrid mode)
           - Each residue predicts distances to its max_partners neighbors
    
    Dimensional Flow (Hybrid Mode):
    ================================
    Policy:     [B, N, 4+K] → Transformer → [B, N, 4] + [B, N, K]
    Flatten:    [B, N, 4] → [B, N*4],  [B, N, K] → [B, N*K]
    Generator:  Concatenate → [B, N*(4+K)] (flattened API)
    Diffusion:  Reshape internally → [B, N, 4+K] (per-residue processing)
    
    Where:
        B = batch size
        N = num_residues
        K = max_partners (per-residue pairwise dimension)
        
    Note: The flattening to [B, N*K] happens in the training loop before passing to
    the generator. The generator expects flattened input but internally reshapes it
    for per-residue processing (when cond_injection_mode='per_residue').
    
    Args:
        num_residues: Number of residues in the peptide
        feature_dim: Input feature dimension per residue (4 for torsion-only, 4+K for hybrid)
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout probability
        pair_dim: Per-residue pairwise dimension (0 for torsion-only, K for hybrid mode)
                  This is max_partners, NOT hybrid_pair_dim (which is N*K)
    """

    def __init__(
        self,
        num_residues: int,
        feature_dim: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        init_log_std: float = math.log(0.3),
        pair_dim: int = 0,
    ):
        super().__init__()
        self.num_residues = num_residues
        self.feature_dim = feature_dim
        self.pair_dim = pair_dim

        # Normalize per-residue feature channels to have roughly zero mean
        # and unit variance before projection so torsion and pairwise
        # components contribute on comparable scales.
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=num_residues)
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=4 * d_model,
            dropout=dropout,
        )
        # Torsion head:
        #   Outputs 4 parameters per residue:
        #     [mu_delta_phi, kappa_phi_raw, mu_delta_psi, kappa_psi_raw]
        #   which are converted into two VonMises distributions over
        #   residual angle changes Δφ and Δψ in forward().
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 4),
        )
        
        # Head for pairwise distances (per-residue predictions)
        # Output max_partners distances per residue, matching input structure
        if self.pair_dim > 0:
            # Pairwise head:
            #   Outputs per-residue log-scale means μ_log_scale for a
            #   LogNormal distribution over multiplicative distance
            #   factors. The global vector pair_log_std parameterizes
            #   the log-scale standard deviations σ_log_scale.
            self.pair_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, self.pair_dim),  # pair_dim = max_partners
            )
            self.pair_log_std = nn.Parameter(torch.full((self.pair_dim,), init_log_std, dtype=torch.float32))
        else:
            self.pair_head = None
        
        # Kept for backward compatibility with existing optimizer
        # code which clamps this parameter as a generic exploration
        # floor, although the torsion VonMises head below uses its
        # own kappa parameterization directly.
        self.log_std = nn.Parameter(torch.full((num_residues, 4), init_log_std, dtype=torch.float32))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        if self.pair_head is not None:
            for module in self.pair_head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, residue_features: torch.Tensor) -> Tuple[VonMises, Optional[LogNormal]]:
        """
        Args:
            residue_features: Tensor of shape (B, num_residues, feature_dim).
        Returns:
            torsion_dist: Normal over (B, num_residues, 4) for (sinφ, cosφ, sinψ, cosψ).
            pair_dist: Normal over (B, num_residues, pair_dim) for per-residue pairwise, or None.
        """
        if residue_features.ndim != 3:
            raise ValueError("residue_features must be 3D (B, num_residues, feature_dim)")
        if residue_features.shape[1] != self.num_residues:
            raise ValueError(
                f"Expected residue_features second dim {self.num_residues}, got {residue_features.shape[1]}"
            )
        # Per-residue feature normalization across channels
        h = self.input_norm(residue_features)
        h = self.input_proj(h)
        h = self.pos_enc(h)
        h = self.encoder(h)
        
        # ------------------------------------------------------------------
        # Torsion Head: residual Δ angles with VonMises distributions
        # ------------------------------------------------------------------
        # head(h) → [B, N, 4] = [mu_phi, kappa_phi_raw, mu_psi, kappa_psi_raw]
        torsion_params = self.head(h)
        mu_phi = torsion_params[..., 0]
        kappa_phi_raw = torsion_params[..., 1]
        mu_psi = torsion_params[..., 2]
        kappa_psi_raw = torsion_params[..., 3]

        # Concentrations: ensure strictly positive and cap maximum
        # precision to maintain an exploration floor.
        kappa_phi = F.softplus(kappa_phi_raw) + 1.0
        kappa_psi = F.softplus(kappa_psi_raw) + 1.0
        kappa_phi = torch.clamp(kappa_phi, max=100.0)
        kappa_psi = torch.clamp(kappa_psi, max=100.0)

        # Stack into [B, N, 2] tensors for Δφ and Δψ
        loc = torch.stack([mu_phi, mu_psi], dim=-1)
        concentration = torch.stack([kappa_phi, kappa_psi], dim=-1)
        torsion_dist = VonMises(loc, concentration)
        
        # ------------------------------------------------------------------
        # Pairwise Head: multiplicative log-normal scales for distances
        # ------------------------------------------------------------------
        pair_dist = None
        if self.pair_head is not None:
            # Mean of log-scale (μ_log_scale) per residue/partner
            mu_log_scale = self.pair_head(h)  # [B, num_residues, pair_dim]
            # Global log-scale std parameter vector → broadcast
            sigma_log = torch.exp(self.pair_log_std)
            sigma_log = torch.where(sigma_log.isfinite(), sigma_log, torch.full_like(sigma_log, 1.0))
            sigma_log = torch.clamp(sigma_log, min=1e-3)
            sigma_log = sigma_log.unsqueeze(0).unsqueeze(0).expand_as(mu_log_scale)
            pair_dist = LogNormal(mu_log_scale, sigma_log)
            
        return torsion_dist, pair_dist
