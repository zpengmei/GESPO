"""Autoregressive transformer policy over per-residue torsions."""
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, VonMises

from gespo.models.transformer import PositionalEncoding, TransformerEncoder
from .von_mises_start import mixture_normal_log_prob, mixture_vonmises_log_prob

class TorsionARPolicy(nn.Module):
    """
    Autoregressive transformer policy over per-residue torsions.

    Default angle layout is backbone (φ, ψ, ω) and optional sidechain torsions
    (χ1..χK) padded to a fixed `n_angles` with an `angle_mask` indicating which
    angles are defined for each residue.

    - Factorization: π(Θ) = Π_i π(θ_i | Θ_<i, seq)
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
        n_angles: int = 3,
        angle_names: Optional[Sequence[str]] = None,
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
        self.n_angles = int(n_angles)
        if self.n_angles <= 0:
            raise ValueError(f"n_angles must be > 0, got {self.n_angles}")

        if angle_names is None:
            if self.n_angles >= 3:
                extra = [f"chi{i}" for i in range(1, self.n_angles - 2 + 1)]
                self.angle_names = ["phi", "psi", "omega", *extra][: self.n_angles]
            else:
                self.angle_names = [f"angle{i}" for i in range(self.n_angles)]
        else:
            names = [str(x) for x in angle_names]
            if len(names) != self.n_angles:
                raise ValueError(f"angle_names length {len(names)} != n_angles {self.n_angles}")
            self.angle_names = names

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

        # Previous-torsion embedding:
        #   [sinθ_1, cosθ_1, ..., sinθ_A, cosθ_A, m_1, ..., m_A] -> d_model
        # where m* are binary mask bits indicating whether each angle is defined.
        self.torsion_proj = nn.Linear(3 * self.n_angles, d_model)

        # Causal transformer encoder (GPT-style via attention mask)
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=4 * d_model,
            dropout=dropout,
        )

        # Shared head that outputs mixture params for all angles per residue:
        # n_angles × 3*K parameters (logits_K, mu_K, kappa_raw_K).
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3 * self.n_angles * kmix),
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
        # Ensure all inputs are on the same device as the torsion sequence.
        # This avoids subtle multi-GPU issues where masks/types are materialized on a
        # different device than the torsion tensor (e.g., when using multiple CUDA
        # devices for env rollouts or policy evaluation).
        device = torsions_shifted.device
        if residue_types is None:
            B, L, _ = torsions_shifted.shape
            residue_types = torsions_shifted.new_zeros((B, L), dtype=torch.long)
        elif torch.is_tensor(residue_types) and residue_types.device != device:
            residue_types = residue_types.to(device=device)
        if angle_mask_shifted is None:
            angle_mask_shifted = torch.ones_like(torsions_shifted, dtype=torch.bool)
        elif torch.is_tensor(angle_mask_shifted) and angle_mask_shifted.device != device:
            angle_mask_shifted = angle_mask_shifted.to(device=device)

        # Encode torsions as sin/cos per angle and concatenate mask bits
        if torsions_shifted.ndim != 3 or torsions_shifted.shape[2] != self.n_angles:
            raise ValueError(
                f"torsions_shifted must be [B,L,{self.n_angles}], got {tuple(torsions_shifted.shape)}"
            )
        if angle_mask_shifted.shape != torsions_shifted.shape:
            raise ValueError(
                f"angle_mask_shifted must match torsions_shifted shape {tuple(torsions_shifted.shape)}, "
                f"got {tuple(angle_mask_shifted.shape)}"
            )
        B, L, A = torsions_shifted.shape
        sin_cos = torch.stack([torch.sin(torsions_shifted), torch.cos(torsions_shifted)], dim=-1)  # [B,L,A,2]
        sin_cos = sin_cos.reshape(B, L, 2 * A)  # [B,L,2A]
        mask_feat = angle_mask_shifted.to(sin_cos.dtype)  # [B,L,A]
        torsion_input = torch.cat([sin_cos, mask_feat], dim=-1)  # [B,L,3A]
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

    def _angle_params(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Map hidden states to mixture params for all torsion angles.

        Returns:
            logits, mu, kappa each of shape [B, L, A, K].
        """
        B, L, _ = h.shape
        out = self.head(h)  # [B, L, 3*A*K]
        K = int(self.kmix)
        A = int(self.n_angles)
        if out.shape[-1] != 3 * A * K:
            raise RuntimeError(
                f"Internal error: expected head dim {3*A*K}, got {int(out.shape[-1])}. "
                f"(A={A}, K={K})"
            )
        out = out.view(B, L, A, 3, K)  # [B,L,A,3,K]
        logits = out[:, :, :, 0, :]  # [B,L,A,K]
        mu = out[:, :, :, 1, :]      # [B,L,A,K]
        kappa = out[:, :, :, 2, :]   # [B,L,A,K]

        # Concentrations must be positive; clamp to avoid degenerate or overly sharp peaks.
        kappa = F.softplus(kappa) + 1.0
        kappa = torch.clamp(kappa, max=200.0)

        # Wrap means into [-π, π] to keep them in canonical range.
        pi = math.pi
        mu = (mu + pi) % (2 * pi) - pi
        return logits, mu, kappa

    def log_prob(
        self,
        torsions: torch.Tensor,
        residue_types: Optional[torch.Tensor] = None,
        angle_mask: Optional[torch.Tensor] = None,
        global_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score a batch of proposals by teacher forcing.

        Positive global descriptors are scored in log1p coordinates. The raw-
        coordinate Jacobian is independent of policy parameters and omitted, so
        it cancels in GRPO likelihood ratios. Zero descriptors use probability
        mass, not density. This score does not normalize range-constrained laws.

        Args:
            torsions:      [B, L, 3] angles (φ, ψ, ω) in radians.
            residue_types: [B, L] residue indices (optional).
            angle_mask:    optional [B, L, 3] bool mask (True = valid angle).

        Returns:
            log_probs: [B] total log-probability per sequence.
        """
        # Align all inputs to the policy device. In multi-GPU workflows it's common
        # for torsions/masks to be produced on a different CUDA device than the
        # policy network (or the "old" policy used for PPO ratios).
        device = self.device
        if torch.is_tensor(torsions) and torsions.device != device:
            torsions = torsions.to(device=device)
        if residue_types is not None and torch.is_tensor(residue_types) and residue_types.device != device:
            residue_types = residue_types.to(device=device)
        if angle_mask is not None and torch.is_tensor(angle_mask) and angle_mask.device != device:
            angle_mask = angle_mask.to(device=device)
        if global_values is not None and torch.is_tensor(global_values) and global_values.device != device:
            global_values = global_values.to(device=device)

        if torsions.ndim != 3 or torsions.shape[2] != self.n_angles:
            raise ValueError(f"torsions must have shape [B, L, {self.n_angles}]")

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
            raise ValueError(f"angle_mask must have same shape as torsions [B, L, {self.n_angles}]")

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
        logits, mu, kappa = self._angle_params(h)  # [B,L,A,K]

        # Sum over residues and angles where masks are active.
        torsion_lp = torch.zeros((B,), dtype=mu.dtype, device=mu.device)
        for a in range(int(self.n_angles)):
            lp_a = mixture_vonmises_log_prob(torsions[..., a], logits[..., a, :], mu[..., a, :], kappa[..., a, :])
            m_a = angle_mask[..., a].to(lp_a.dtype)
            torsion_lp = torsion_lp + (lp_a * m_a).sum(dim=1)

        if self.n_global_tokens > 0:
            if global_values is None:
                global_values = torsions.new_zeros((B, self.n_global_tokens), dtype=torch.float32)
            gv = global_values.to(dtype=torch.float32, device=torsions.device)
            y = self._globals_to_y(gv)  # [B, G]
            gl_logits, gl_mu, gl_sigma = self._global_params(h_full)
            lp_g = mixture_normal_log_prob(y, gl_logits, gl_mu, gl_sigma)  # [B, G]
            # Sampling draws z on the real line and reports g = max(exp(z) - 1, 0),
            # so g == 0 is an atom carrying the mixture mass at z <= 0. Scoring it
            # with a density at z = 0 is the wrong quantity and makes the likelihood
            # ratio inconsistent with the action actually taken.
            at_zero = gv <= 0.0
            if bool(at_zero.any()):
                w = torch.softmax(gl_logits, dim=-1)                       # [B, G, K]
                cdf0 = 0.5 * torch.erfc(gl_mu / (gl_sigma * math.sqrt(2.0)))  # Phi((0-mu)/sigma)
                lp_atom = torch.log((w * cdf0).sum(dim=-1).clamp_min(1e-30))  # [B, G]
                lp_g = torch.where(at_zero, lp_atom, lp_g)
            global_lp = lp_g.sum(dim=1)  # [B]
            return torsion_lp + global_lp

        return torsion_lp  # [B]

    @torch.no_grad()
    def mode_torsions(
        self,
        torsions: torch.Tensor,
        residue_types: Optional[torch.Tensor] = None,
        angle_mask: Optional[torch.Tensor] = None,
        global_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute a deterministic "policy suggested" torsion sequence by taking,
        at each residue and angle, the mean (μ) of the most-probable mixture
        component under teacher forcing.

        This is intended for diagnostics (e.g., how far sampled actions deviate
        from the policy's greedy/component-mode suggestion).

        Returns:
            mode_mu: [B, L, A] angles in radians.
        """
        device = self.device
        if torch.is_tensor(torsions) and torsions.device != device:
            torsions = torsions.to(device=device)
        if residue_types is not None and torch.is_tensor(residue_types) and residue_types.device != device:
            residue_types = residue_types.to(device=device)
        if angle_mask is not None and torch.is_tensor(angle_mask) and angle_mask.device != device:
            angle_mask = angle_mask.to(device=device)
        if global_values is not None and torch.is_tensor(global_values) and global_values.device != device:
            global_values = global_values.to(device=device)

        if torsions.ndim != 3 or torsions.shape[2] != self.n_angles:
            raise ValueError(f"torsions must have shape [B, L, {self.n_angles}]")
        B, L, _ = torsions.shape
        if L != self.num_residues:
            raise ValueError(f"Expected torsions length {self.num_residues}, got {L}")

        if angle_mask is None:
            angle_mask = torch.ones_like(torsions, dtype=torch.bool)
        if angle_mask.shape != torsions.shape:
            raise ValueError(f"angle_mask must have same shape as torsions [B, L, {self.n_angles}]")

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

        logits, mu, _ = self._angle_params(h)  # [B,L,A,K]
        idx = torch.argmax(logits, dim=-1)  # [B,L,A]
        mode = torch.gather(mu, dim=-1, index=idx.unsqueeze(-1)).squeeze(-1).to(dtype=torsions.dtype)  # [B,L,A]
        mode = torch.where(angle_mask, mode, torsions)
        return mode

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
        if torsions.ndim != 3 or torsions.shape[2] != self.n_angles:
            raise ValueError(f"torsions must have shape [B, L, {self.n_angles}]")

        B, L, _ = torsions.shape
        if L != self.num_residues:
            raise ValueError(f"Expected torsions length {self.num_residues}, got {L}")

        if angle_mask is None:
            angle_mask = torch.ones_like(torsions, dtype=torch.bool)
        if angle_mask.shape != torsions.shape:
            raise ValueError(f"angle_mask must have same shape as torsions [B, L, {self.n_angles}]")

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
        logits, _, kappa = self._angle_params(h)  # [B,L,A,K]

        def _mixture_entropy_upper(logits: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
            w = torch.softmax(logits, dim=-1)
            logw = torch.log_softmax(logits, dim=-1)
            ent_w = -(w * logw).sum(dim=-1)  # [...,]
            ent_vm = self._vonmises_entropy(kappa)  # [..., K]
            ent_x_upper = ent_w + (w * ent_vm).sum(dim=-1)  # [...]
            return ent_x_upper

        ent = _mixture_entropy_upper(logits, kappa)  # [B,L,A]
        mask = angle_mask.to(ent.dtype)  # [B,L,A]
        if (not include_omega) and self.n_angles >= 3:
            mask = mask.clone()
            mask[..., 2] = 0.0
        total_ent = (ent * mask).sum(dim=(1, 2))  # [B]
        total_tokens = mask.sum(dim=(1, 2)).clamp_min(1.0)  # [B]

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
            torsions: [B, L, A] sampled angles in radians.
        """
        B = int(batch_size)
        L = int(self.num_residues)
        A = int(self.n_angles)
        device = next(self.parameters()).device

        if residue_types is None:
            residue_types = torch.zeros((B, L), dtype=torch.long, device=device)
        if angle_mask is None:
            angle_mask = torch.ones((B, L, A), dtype=torch.bool, device=device)
        if angle_mask.shape != (B, L, A):
            raise ValueError(f"angle_mask must have shape [B,L,A]=[{B},{L},{A}], got {tuple(angle_mask.shape)}")

        torsions = torch.zeros((B, L, A), dtype=torch.float32, device=device)
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
            logits, mu, kappa = self._angle_params(h)  # [B,L,A,K]
            logits_i = logits[:, i, :, :] / max(float(temperature), 1e-3)  # [B,A,K]
            mu_i = mu[:, i, :, :]  # [B,A,K]
            kappa_i = kappa[:, i, :, :]  # [B,A,K]

            # Guardrail: prevent NaNs/Infs / extreme concentrations from triggering
            # CUDA device-side asserts in Categorical/VonMises sampling.
            logits_i = torch.nan_to_num(logits_i, nan=0.0, posinf=0.0, neginf=0.0)
            mu_i = torch.nan_to_num(mu_i, nan=0.0, posinf=0.0, neginf=0.0)
            kappa_i = torch.nan_to_num(kappa_i, nan=1.0, posinf=50.0, neginf=1.0)
            logits_i = torch.clamp(logits_i, min=-80.0, max=80.0)
            # Sampling-only cap: VonMises rejection sampling can overflow for very large kappa.
            kappa_i = torch.clamp(kappa_i, min=1e-6, max=50.0)

            # Sample each angle independently at residue i.
            samples_i = torch.zeros((B, A), dtype=torsions.dtype, device=device)
            for a in range(A):
                valid = angle_mask[:, i, a]
                if not bool(torch.any(valid)):
                    samples_i[:, a] = torsions[:, i, a]
                    continue

                cat = Categorical(logits=logits_i[:, a, :])
                idx = cat.sample()  # [B]
                mu_sel = torch.gather(mu_i[:, a, :], dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
                kappa_sel = torch.gather(kappa_i[:, a, :], dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
                mu_sel = torch.nan_to_num(mu_sel, nan=0.0, posinf=0.0, neginf=0.0)
                kappa_sel = torch.nan_to_num(kappa_sel, nan=1.0, posinf=50.0, neginf=1.0)
                kappa_sel = torch.clamp(kappa_sel, min=1e-6, max=50.0)
                vm = VonMises(mu_sel, kappa_sel)
                samp = vm.sample()
                samp = torch.where(valid, samp, torsions[:, i, a])
                samples_i[:, a] = samp

            torsions[:, i, :] = samples_i

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
            torsions: [B, L, A]
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
        A = int(self.n_angles)
        device = next(self.parameters()).device
        if residue_types is None:
            residue_types = torch.zeros((B, L), dtype=torch.long, device=device)
        if angle_mask is None:
            angle_mask = torch.ones((B, L, A), dtype=torch.bool, device=device)
        if angle_mask.shape != (B, L, A):
            raise ValueError(f"angle_mask must have shape [B,L,A]=[{B},{L},{A}], got {tuple(angle_mask.shape)}")

        # Sample globals sequentially in y=log1p(space), then map to raw nm.
        globals_raw = torch.zeros((B, self.n_global_tokens), dtype=torch.float32, device=device)
        torsions_shift = torch.zeros((B, L, A), dtype=torch.float32, device=device)
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

            # Guardrail: prevent NaNs/Infs from triggering CUDA device-side asserts
            # in Categorical/Normal sampling (common if the policy explodes).
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
            log_sigma = torch.nan_to_num(log_sigma, nan=0.0, posinf=0.0, neginf=0.0)
            logits = torch.clamp(logits, min=-80.0, max=80.0)
            sigma = F.softplus(log_sigma) + 1e-3
            sigma = torch.clamp(sigma, min=1e-6, max=50.0)
            sigma = torch.nan_to_num(sigma, nan=1.0, posinf=50.0, neginf=1.0)

            # Sample for each item in batch, with optional rejection sampling for range constraints
            raw_samples = torch.zeros(B, dtype=torch.float32, device=device)

            bad_rows = (
                (~torch.isfinite(logits)).any(dim=1)
                | (~torch.isfinite(mu)).any(dim=1)
                | (~torch.isfinite(sigma)).any(dim=1)
                | (sigma <= 0.0).any(dim=1)
            )
            if bool(torch.any(bad_rows)):
                # Replace invalid mixture parameters with a benign, well-defined distribution.
                logits = logits.clone()
                mu = mu.clone()
                sigma = sigma.clone()
                logits[bad_rows] = 0.0
                mu[bad_rows] = 0.0
                sigma[bad_rows] = 1.0

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
                    if not torch.isfinite(mu_sel).all() or not torch.isfinite(sigma_sel).all() or torch.any(sigma_sel <= 0.0):
                        mu_sel = torch.zeros_like(mu_sel)
                        sigma_sel = torch.ones_like(sigma_sel)
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
