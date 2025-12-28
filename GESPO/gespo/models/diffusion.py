"""Conditional hierarchical Cartesian diffusion network."""
from .._imports import *
from .transformer import PositionalEncoding, TransformerEncoder
from .timestep import TimestepEmbedder

try:
    from fla.layers.gated_deltanet import GatedDeltaNet
except ImportError:  # pragma: no cover - optional dependency
    GatedDeltaNet = None


class GDNBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        layer_idx: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        mode: str = "chunk",
    ) -> None:
        super().__init__()
        if GatedDeltaNet is None:
            raise ImportError("fla is required for GatedDeltaNet support. Install fla to use atom_mixer='gated_deltanet'.")
        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.mixer = GatedDeltaNet(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            mode=mode,
            expand_v=2.0,
            use_gate=True,
            use_short_conv=True,
            conv_size=4,
            conv_bias=False,
            layer_idx=layer_idx,
            norm_eps=norm_eps,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, *_ = self.mixer(self.norm(x))
        x = x + y
        return x + self.mlp(x)


class GDNStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [GDNBlock(hidden_size, num_heads, head_dim=head_dim, dropout=dropout, layer_idx=i, norm_eps=norm_eps) for i in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class ConditionalHierarchicalCartesianDiffusion(nn.Module):
    def __init__(
        self,
        n_atoms: int,
        n_atom_types: int,
        R: int,
        max_atoms_res: int,
        res_idx_pad: torch.Tensor,
        res_mask: torch.Tensor,
        atom_types: torch.Tensor,
        d_model: int = 128,
        latent_dim: int = 16,
        nhead: int = 8,
        num_layers_intra: int = 4,
        num_layers_inter: int = 4,
        dropout: float = 0.0,
        latent_level: str = 'atom',
        use_atom_id: bool = True,
        use_intra_pos_enc: bool = True,
        use_dec_intra_pos_enc: bool = True,
        ungroup_mode: str = 'split',
        learn_sigma: bool = True,
        cond_dim: int = 0,
        cond_injection_mode: str = 'global',
        atom_mixer: str = 'transformer',
        num_residue_sinks: int = 3,
    ) -> None:
        super().__init__()
        assert latent_level in ('atom', 'residue')
        assert ungroup_mode in ('broadcast', 'split')
        self.n_atoms = n_atoms
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.R = R
        self.max_atoms_res = max_atoms_res
        self.latent_level = latent_level
        self.use_atom_id = use_atom_id
        self.use_intra_pos_enc = use_intra_pos_enc
        self.use_dec_intra_pos_enc = use_dec_intra_pos_enc
        self.ungroup_mode = ungroup_mode
        self.learn_sigma = learn_sigma
        self.cond_dim = cond_dim
        self.atom_mixer = atom_mixer
        self.cond_injection_mode = cond_injection_mode
        # Optional attention sink tokens at the residue level for the inter-residue encoders
        self.num_residue_sinks = max(0, int(num_residue_sinks))
        if self.cond_injection_mode not in ('global', 'per_residue'):
            raise ValueError(f"Invalid cond_injection_mode '{self.cond_injection_mode}'")
        if self.atom_mixer not in ('transformer', 'gated_deltanet'):
            raise ValueError(f"Unsupported atom_mixer '{self.atom_mixer}'")

        res_idx_pad = res_idx_pad.long()
        res_mask = res_mask.bool()
        self.register_buffer('res_idx_pad', res_idx_pad)
        self.register_buffer('res_mask', res_mask)
        self.register_buffer('res_idx_flat', res_idx_pad.reshape(-1))
        self.register_buffer('key_mask_base', ~res_mask)

        A = int(res_mask.sum().item())
        assert A == n_atoms, f'Cached map atoms ({A}) != n_atoms ({n_atoms})'
        inv_r = torch.full((n_atoms,), -1, dtype=torch.long)
        inv_p = torch.full((n_atoms,), -1, dtype=torch.long)
        for r in range(R):
            Lr = int(res_mask[r].sum().item())
            idxs = res_idx_pad[r, :Lr]
            inv_r[idxs] = r
            inv_p[idxs] = torch.arange(Lr, dtype=torch.long)
        self.register_buffer('inv_r', inv_r)
        self.register_buffer('inv_p', inv_p)
        self.register_buffer('atom_types', atom_types.long())

        self.atom_type_emb = nn.Embedding(n_atom_types + 1, d_model)
        self.atom_id_emb = nn.Embedding(n_atoms, d_model)
        self.coord_proj = nn.Linear(3, d_model)
        if self.atom_mixer == 'transformer':
            self.intra_pos_enc = PositionalEncoding(d_model, max_len=max_atoms_res)
        else:
            self.intra_pos_enc = None

        if self.atom_mixer == 'transformer':
            self.intra_encoder = TransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers_intra, dim_feedforward=4 * d_model, dropout=dropout)
            self._gdn_head_dim = None
        else:
            if GatedDeltaNet is None:
                raise ImportError("fla is required for atom_mixer='gated_deltanet'. Install fla to enable this option.")
            key_dim_required = (3 * d_model) // 4
            if (3 * d_model) % 4 != 0 or key_dim_required % nhead != 0:
                raise ValueError("d_model must satisfy (3*d_model/4) divisible by nhead for gated_deltanet mixer.")
            head_dim = key_dim_required // nhead
            self._gdn_head_dim = head_dim
            self.atom_gdn = GDNStack(num_layers_intra, d_model, nhead, head_dim=head_dim, dropout=dropout)
            self._gdn_dtype = torch.bfloat16
            self.atom_gdn = self.atom_gdn.to(dtype=self._gdn_dtype)
        # Positional encodings sized to include optional sink tokens
        self.res_pos_enc = PositionalEncoding(d_model, max_len=R + self.num_residue_sinks)
        self.inter_encoder = TransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers_inter, dim_feedforward=4 * d_model, dropout=dropout)

        self.to_latent_atom = nn.Linear(d_model * 2, latent_dim)
        self.to_latent_res = nn.Linear(d_model, latent_dim)

        self.dec_z_proj = nn.Linear(latent_dim, d_model)
        self.dec_z_res_proj = nn.Linear(latent_dim, d_model)
        self.dec_res_pos_enc = PositionalEncoding(d_model, max_len=R + self.num_residue_sinks)
        self.dec_inter_encoder = TransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers_inter, dim_feedforward=4 * d_model, dropout=dropout)
        self.dec_ungroup = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, max_atoms_res * d_model),
        )
        self.dec_fuse = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, 2 * d_model),
        )
        with torch.no_grad():
            last_linear = self.dec_fuse[2]
            half = last_linear.bias.numel() // 2
            nn.init.constant_(last_linear.bias[:half], 0.0)
            nn.init.constant_(last_linear.bias[half:], 2.0)

        self.dec_res_skip = nn.Linear(self.d_model, self.d_model)
        nn.init.xavier_uniform_(self.dec_res_skip.weight, gain=0.1)
        nn.init.zeros_(self.dec_res_skip.bias)

        if self.atom_mixer == 'transformer':
            self.dec_intra_pos_enc = PositionalEncoding(d_model, max_len=max_atoms_res)
        else:
            self.dec_intra_pos_enc = None
        if self.atom_mixer == 'transformer':
            self.dec_intra_encoder = TransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers_intra, dim_feedforward=4 * d_model, dropout=dropout)
        else:
            self.dec_atom_gdn = GDNStack(num_layers_intra, d_model, nhead, head_dim=self._gdn_head_dim, dropout=dropout).to(dtype=self._gdn_dtype)
        self.dec_out = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3),
        )
        if self.learn_sigma:
            self.dec_out_sigma = nn.Sequential(
                nn.Linear(d_model, 2 * d_model),
                nn.ReLU(),
                nn.Linear(2 * d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 3),
            )
        else:
            self.dec_out_sigma = None

        # Initialize conditioning modules to None; selectively create based on mode
        # Learnable sink token parameters for encoder/decoder (used only if num_residue_sinks > 0)
        if self.num_residue_sinks > 0:
            self.sink_tokens_enc = nn.Parameter(torch.zeros(self.num_residue_sinks, d_model))
            self.sink_tokens_dec = nn.Parameter(torch.zeros(self.num_residue_sinks, d_model))
            nn.init.xavier_uniform_(self.sink_tokens_enc)
            nn.init.xavier_uniform_(self.sink_tokens_dec)
        else:
            self.sink_tokens_enc = None
            self.sink_tokens_dec = None

        self.cond_per_res_dim = None
        self.cond_proj_enc = None
        self.cond_proj_dec = None
        self.cond_proj_res_enc = None
        self.cond_proj_res_dec = None
        self.residue_cond_fuse_enc = None
        self.residue_cond_fuse_dec = None
        if self.cond_dim > 0:
            if self.cond_injection_mode == 'global':
                self.cond_proj_enc = nn.Sequential(
                    nn.Linear(cond_dim, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, d_model),
                )
                self.cond_proj_dec = nn.Sequential(
                    nn.Linear(cond_dim, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, d_model),
                )
                # Per-residue modules remain None in global mode
            else:
                if cond_dim % R != 0:
                    raise ValueError(f"cond_dim {cond_dim} must be divisible by number of residues {R} for per_residue conditioning.")
                self.cond_per_res_dim = cond_dim // R
                self.cond_proj_res_enc = nn.Sequential(
                    nn.Linear(self.cond_per_res_dim, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, d_model),
                )
                self.cond_proj_res_dec = nn.Sequential(
                    nn.Linear(self.cond_per_res_dim, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, d_model),
                )
                self.residue_cond_fuse_enc = nn.Sequential(
                    nn.Linear(2 * d_model, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, d_model),
                )
                self.residue_cond_fuse_dec = nn.Sequential(
                    nn.Linear(2 * d_model, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, d_model),
                )
        else:
            # No conditioning: all modules remain None
            pass

        self.time_embedder = TimestepEmbedder(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.SiLU(),
            nn.Linear(4*d_model, d_model),
        )

    def _gather_res(self, tensor_flat: torch.Tensor) -> torch.Tensor:
        B, A, D = tensor_flat.shape
        R, Lmax = self.R, self.max_atoms_res
        idx = self.res_idx_flat.unsqueeze(0).expand(B, -1)
        gathered = torch.gather(tensor_flat, 1, idx.unsqueeze(-1).expand(B, R * Lmax, D))
        return gathered.reshape(B, R, Lmax, D)

    def _scatter_flat(self, tensor_res: torch.Tensor) -> torch.Tensor:
        B, R, Lmax, D = tensor_res.shape
        b_idx = torch.arange(B, device=tensor_res.device).unsqueeze(1)
        return tensor_res[b_idx, self.inv_r.unsqueeze(0), self.inv_p.unsqueeze(0)]

    def _run_gdn_stack(self, stack: nn.Module, seq: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        seq_dtype = seq.dtype
        seq_cast = seq.to(self._gdn_dtype)
        if mask is not None:
            seq_cast = seq_cast.masked_fill(mask.unsqueeze(-1), 0.0)
        out = stack(seq_cast)
        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(-1), 0.0)
        return out.to(seq_dtype)

    def _apply_residue_condition(self, base: torch.Tensor, cond_embed: Optional[torch.Tensor], fuse: Optional[nn.Module]) -> torch.Tensor:
        if cond_embed is None:
            return base
        if fuse is None:
            return base + cond_embed
        fused = torch.cat([base, cond_embed], dim=-1)
        return fuse(fused)

    def _prepare_atom_tokens(
        self, x: torch.Tensor, cond: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, A, _ = x.shape
        device = x.device
        type_emb = self.atom_type_emb(self.atom_types.clamp(min=0))
        if self.use_atom_id:
            id_emb = self.atom_id_emb(torch.arange(A, device=device))
        else:
            id_emb = torch.zeros((A, self.d_model), device=device)
        atom_token = self.coord_proj(x) + type_emb.unsqueeze(0) + id_emb.unsqueeze(0)
        cond_embed_global = None
        cond_embed_res = None
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("Conditioning vector is required but missing.")
            if self.cond_injection_mode == "global":
                cond_embed_global = self.cond_proj_enc(cond)
                atom_token = atom_token + cond_embed_global.unsqueeze(1)
            else:
                cond_res = cond.view(B, self.R, self.cond_per_res_dim)
                cond_embed_res = self.cond_proj_res_enc(cond_res)
        return atom_token, cond_embed_global, cond_embed_res

    def _encode_atom_mixer(self, atom_token: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.atom_mixer == "transformer":
            h_res = self._gather_res(atom_token)
            B_, R, Lmax, D = h_res.shape
            if self.use_intra_pos_enc:
                intra_in = self.intra_pos_enc(h_res.reshape(B_ * R, Lmax, D))
            else:
                intra_in = h_res.reshape(B_ * R, Lmax, D)
            key_mask = (
                self.key_mask_base.unsqueeze(0)
                .expand(atom_token.size(0), -1, -1)
                .reshape(B_ * R, Lmax)
            )
            h_intra = self.intra_encoder(intra_in, key_padding_mask=key_mask).reshape(
                atom_token.size(0), R, Lmax, D
            )
            return h_intra, h_res
        h_atom = self._run_gdn_stack(self.atom_gdn, atom_token, mask=None)
        h_res = self._gather_res(h_atom)
        return h_res, h_res

    def _decode_atom_mixer(self, atom_res_ctx: torch.Tensor) -> torch.Tensor:
        B_, R_, Lmax_, D_ = atom_res_ctx.shape
        if self.atom_mixer == "transformer":
            if self.use_dec_intra_pos_enc:
                dec_intra_in = self.dec_intra_pos_enc(atom_res_ctx.reshape(B_ * R_, Lmax_, D_))
            else:
                dec_intra_in = atom_res_ctx.reshape(B_ * R_, Lmax_, D_)
            key_mask = (
                self.key_mask_base.unsqueeze(0)
                .expand(B_, -1, -1)
                .reshape(B_ * R_, Lmax_)
            )
            h_dec = self.dec_intra_encoder(dec_intra_in, key_padding_mask=key_mask).reshape(
                B_, R_, Lmax_, D_
            )
            return self._scatter_flat(h_dec)
        atom_seq_flat = self._scatter_flat(atom_res_ctx)
        return self._run_gdn_stack(self.dec_atom_gdn, atom_seq_flat, mask=None)

    def encode_latent(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        atom_token, cond_embed_global, cond_embed_res = self._prepare_atom_tokens(x, cond)
        h_intra, h_res = self._encode_atom_mixer(atom_token)
        B = x.size(0)

        m = (~self.key_mask_base).to(h_intra.dtype).unsqueeze(0)
        sum_tok = (h_intra * m.unsqueeze(-1)).sum(dim=2)
        cnt = m.sum(dim=2).clamp(min=1.0)
        res_tok = sum_tok / cnt.unsqueeze(-1)

        if cond_embed_global is not None:
            res_tok = res_tok + cond_embed_global.unsqueeze(1)
        res_tok = self._apply_residue_condition(res_tok, cond_embed_res, self.residue_cond_fuse_enc)
        # Optionally prepend attention sink tokens to residue sequence
        if self.num_residue_sinks > 0 and self.sink_tokens_enc is not None:
            B, R, D = res_tok.shape
            sink = self.sink_tokens_enc.unsqueeze(0).expand(B, -1, -1)
            res_seq = torch.cat([sink, res_tok], dim=1)
        else:
            res_seq = res_tok
        res_seq = self.res_pos_enc(res_seq)
        h_inter_full = self.inter_encoder(res_seq)
        # Drop sink outputs; keep only residue tokens
        h_inter = h_inter_full[:, -self.R :, :]

        if self.latent_level == 'atom':
            h_inter_b = h_inter.unsqueeze(2).expand(B, self.R, self.max_atoms_res, self.d_model)
            h_cat_res = torch.cat([h_intra, h_inter_b], dim=-1)
            h_cat_flat = self._scatter_flat(h_cat_res)
            return self.to_latent_atom(h_cat_flat)
        else:
            return self.to_latent_res(h_inter)

    def decode_latent(self, z: torch.Tensor, time_embed: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = z.device
        D = self.d_model
        A = self.n_atoms
        R = self.R
        Lmax = self.max_atoms_res

        type_emb = self.atom_type_emb(self.atom_types.clamp(min=0))
        if self.use_atom_id:
            id_emb = self.atom_id_emb(torch.arange(A, device=device))
        else:
            id_emb = torch.zeros((A, D), device=device)
        atom_base = (type_emb + id_emb).unsqueeze(0)

        cond_embed_dec = None
        cond_embed_res_dec = None
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError('Conditioning vector is required but missing.')
            if self.cond_injection_mode == 'global':
                cond_embed_dec = self.cond_proj_dec(cond)
                atom_base = atom_base + cond_embed_dec.unsqueeze(1)
            else:
                cond_res = cond.view(z.size(0), R, self.cond_per_res_dim)
                cond_embed_res_dec = self.cond_proj_res_dec(cond_res)

        B = z.size(0)
        if self.latent_level == 'atom':
            z_feat = self.dec_z_proj(z)
            z_feat = z_feat + time_embed.unsqueeze(1)
            atom_init = z_feat + atom_base
            atom_res = self._gather_res(atom_init)
            m = (~self.key_mask_base).to(atom_res.dtype).unsqueeze(0)
            sum_tok = (atom_res * m.unsqueeze(-1)).sum(dim=2)
            cnt = m.sum(dim=2).clamp(min=1.0)
            res_tok0 = sum_tok / cnt.unsqueeze(-1)
        else:
            res_tok0 = self.dec_z_res_proj(z) + time_embed.unsqueeze(1)
            atom_res = self._gather_res(atom_base.expand(B, -1, -1))

        if cond_embed_dec is not None:
            res_tok0 = res_tok0 + cond_embed_dec.unsqueeze(1)
        res_tok0 = self._apply_residue_condition(res_tok0, cond_embed_res_dec, self.residue_cond_fuse_dec)

        # Optionally prepend sink tokens to decoder residue tokens
        if self.num_residue_sinks > 0 and self.sink_tokens_dec is not None:
            B, R_cur, D = res_tok0.shape
            sink = self.sink_tokens_dec.unsqueeze(0).expand(B, -1, -1)
            res_seq0 = torch.cat([sink, res_tok0], dim=1)
        else:
            res_seq0 = res_tok0
        res_seq0 = self.dec_res_pos_enc(res_seq0)
        res_ctx_full = self.dec_inter_encoder(res_seq0)
        res_ctx = res_ctx_full[:, -self.R :, :]

        if self.ungroup_mode == 'broadcast':
            res_ctx_atoms = res_ctx.unsqueeze(2).expand(B, R, Lmax, D)
        else:
            res_ctx_split = self.dec_ungroup(res_ctx)
            res_ctx_atoms = res_ctx_split.reshape(B, R, Lmax, D)

        res_ctx_atoms = res_ctx_atoms * (~self.key_mask_base).unsqueeze(0).unsqueeze(-1).to(res_ctx_atoms.dtype)
        atom_res = atom_res + self.dec_res_skip(res_ctx_atoms)
        fuse_in = torch.cat([atom_res, res_ctx_atoms], dim=-1)
        gates = self.dec_fuse(fuse_in)
        delta, gate = torch.chunk(gates, 2, dim=-1)
        atom_res_ctx = atom_res + torch.sigmoid(gate) * delta

        h_flat = self._decode_atom_mixer(atom_res_ctx)
        return h_flat

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        squeeze_last = False
        if x.dim() == 4:
            squeeze_last = True
            x = x.squeeze(-1)
        if x.dim() != 3:
            raise ValueError(f'Expected input of shape [B, 3, N] or [B, 3, N, 1], got {tuple(x.shape)}')
        B, C, N = x.shape
        if C != 3:
            raise ValueError(f'Expected 3 coordinate channels, got {C}')

        coords = x.permute(0, 2, 1)
        z = self.encode_latent(coords, cond)
        time_embed = self.time_mlp(self.time_embedder(timesteps))
        h_flat = self.decode_latent(z, time_embed, cond)

        noise = self.dec_out(h_flat).permute(0, 2, 1)
        if self.learn_sigma and self.dec_out_sigma is not None:
            log_sigma = self.dec_out_sigma(h_flat).permute(0, 2, 1)
            out = torch.cat([noise, log_sigma], dim=1)
        else:
            out = noise

        if squeeze_last:
            out = out.unsqueeze(-1)
        return out

    @torch.no_grad()
    def predict_coords(self, x: torch.Tensor, timesteps: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        preds = self.forward(x, timesteps, cond=cond)
        if self.learn_sigma and self.dec_out_sigma is not None:
            preds = preds[:, :3]
        if preds.dim() == 4:
            preds = preds.squeeze(-1)
        return preds.permute(0, 2, 1)
