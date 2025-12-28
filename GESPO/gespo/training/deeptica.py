import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Callable
from torch.jit import Final

# =============================================================================
# Utils / EarlyStopping (Simplified)
# =============================================================================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, consecutive=True, log=False, save_best_model=True):
        self.patience = patience
        self.min_delta = min_delta
        self.consecutive = consecutive
        self.log = log
        self.save_best_model = save_best_model
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model=None, epoch=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.save_best_model and model is not None:
                self.best_model = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            if self.consecutive:
                self.counter += 1
            else:
                # cumulative logic could be here
                self.counter += 1
            if self.log:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            if self.save_best_model and model is not None:
                self.best_model = model.state_dict()
            self.counter = 0

# =============================================================================
# Base Neural Network Class
# =============================================================================

class BaseNNCV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.normIn = False
        self.normOut = False
        self.opt_ = None
        self.earlystopping_ = None
        self.lrscheduler_ = None
        self.log_header = True
        # TorchScript needs buffers to exist up front even if normalization is disabled.
        self.register_buffer('MeanIn', torch.zeros(1))
        self.register_buffer('RangeIn', torch.ones(1))
        self.register_buffer('MeanOut', torch.zeros(1))
        self.register_buffer('RangeOut', torch.ones(1))

    def standardize_inputs(self, x: torch.Tensor, print_values=False):
        '''
        Enable the standardization of inputs based on max and min over set.

        Parameters:
            x (tensor): reference set over which compute the standardization
        '''
        
        #compute mean and range
        Max, _=torch.max(x,dim=0)
        Min, _=torch.min(x,dim=0)

        Mean=(Max+Min)/2.
        Range=(Max-Min)/2.

        if print_values:
            print( "Input standardization enabled.")
            print('Mean:',Mean.shape,'-->',Mean)
            print('Range:',Range.shape,'-->',Range)
        
        # Avoid division by zero
        mask = Range < 1e-6
        if mask.sum() > 0:
            if print_values:
                print( "[Warninig] Input normalization: the following features have a range of values < 1e-6:", mask.nonzero() )
            Range[mask]=1.

        self.normIn = True
        # Buffers are pre-registered in __init__, so we simply replace them here.
        self.MeanIn = Mean
        self.RangeIn = Range

    def standardize_outputs(self, input: torch.Tensor, print_info=False):
        '''
        Enable the standardization of outputs based on max and min over set.

        Parameters:
            x (tensor): input set over which compute the standardization
        '''
        #disable normout for cv evaluation
        self.normOut = False
        with torch.no_grad():
            x = self.forward(input)

        #compute mean and range
        Max, _=torch.max(x,dim=0)
        Min, _=torch.min(x,dim=0)

        Mean=(Max+Min)/2.
        Range=(Max-Min)/2.

        if print_info:
            print( "Output standardization enabled.")
            print('Mean:',Mean.shape,'-->',Mean)
            print('Range:',Range.shape,'-->',Range)
        
        mask = Range < 1e-6
        if mask.sum() > 0:
            if print_info:
                print( "[Warninig] Output normalization: the following features have a range of values < 1e-6:", mask.nonzero() )
            Range[mask]=1.

        self.normOut = True
        # Buffers are pre-registered in __init__, so we simply replace them here.
        self.MeanOut = Mean
        self.RangeOut = Range

    def disable_standardize_outputs(self,print_info=False):
        '''
        Disable the standardization of outputs.
        '''
        if print_info:
            print( "Output standardization disabled.")
        self.normOut = False

    def _normalize(self, x: torch.Tensor, Mean: torch.Tensor, Range: torch.Tensor) -> (torch.Tensor):
        ''' Compute standardized outputs '''

        # Handle broadcasting if needed
        # Reference implementation uses unsqueeze and expand
        # But register_buffer handles device, we just need shapes
        
        # If x is [batch, dim] and Mean is [dim]
        # x.sub(Mean) works by broadcasting
        # But reference does explicit expansion:
        # Mean_ = Mean.unsqueeze(0).expand(batch_size, x_size)
        # This is safer if x has more dims?
        
        # Let's stick to reference logic but be careful about dimensions
        if x.ndim == 2:
            batch_size = x.size(0)
            x_size = x.size(1)
            
            # Ensure Mean/Range are on same device (handled by register_buffer)
            # But we need to make sure they are 1D
            if Mean.ndim == 1:
                Mean_ = Mean.unsqueeze(0).expand(batch_size, x_size)
                Range_ = Range.unsqueeze(0).expand(batch_size, x_size)
                return x.sub(Mean_).div(Range_)
        
        # Fallback to broadcasting which is usually fine
        return x.sub(Mean).div(Range)
    
    def set_optimizer(self,opt):
        self.opt_ = opt
        
    def default_optimizer(self):
        self.opt_ = torch.optim.Adam(self.parameters())
        
    def set_earlystopping(self,patience=5, min_delta = 0, consecutive=True, log=False, save_best_model=True):
        # Use local EarlyStopping class
        self.earlystopping_ = EarlyStopping(patience,min_delta,consecutive,log,save_best_model)
        self.best_valid = None
        self.best_model = None
        
    def set_lrscheduler(self, opt):
        # utils.LRScheduler is missing, so we skip or implement a basic one
        # For now, just set to None or log warning
        print("Warning: LRScheduler not available (utils module missing).")
        self.lrscheduler_ = None
        
    def print_info(self):
        print('================INFO================')
        print('[MODEL]')
        print(self)
        print('\n[OPTIMIZER]')
        print(self.opt_)
        # print('\n[PARAMETERS]')
        # print(self.get_params()) # DeepTICA has get_params, but BaseNNCV doesn't? 
        # Reference calls self.get_params() so it assumes subclass has it.
        if hasattr(self, 'get_params'):
            print('\n[PARAMETERS]')
            print(self.get_params())
        print('====================================')
        
    def print_log(self,log_dict,spacing=None,decimals=3):
        if spacing is None:
            spacing = [16 for i in range(len(log_dict))]
        if self.log_header:
            for i,key in enumerate(log_dict.keys()):
                print("{0:<{width}s}".format(key,width=spacing[i]),end='')
            print('')
            self.log_header = False
            
        for i,value in enumerate(log_dict.values()):
            if type(value) == int:
                print("{0:<{width}d}".format(value,width=spacing[i]),end='')
                
            if (type(value) == torch.Tensor) or (type(value) == torch.nn.parameter.Parameter) :
                value = value.detach().cpu().numpy() # Detach and cpu for numpy
                if value.shape == ():
                    print("{0:<{width}.{dec}f}".format(value,width=spacing[i],dec=decimals),end='')
                else:
                    for v in value:
                        print("{0:<6.3f}".format(v),end=' ')
        print('')

# =============================================================================
# Graphormer3D Components
# =============================================================================

@torch.jit.script
def softmax_dropout(input, dropout_prob: float, is_training: bool):
    return F.dropout(F.softmax(input, -1), dropout_prob, is_training)

class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(
            embed_dim, embed_dim * 3, bias=bias
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query: torch.Tensor,
        attn_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        n_node, n_graph, embed_dim = query.size()
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        _shape = (-1, n_graph * self.num_heads, self.head_dim)
        q = q.contiguous().view(_shape).transpose(0, 1) * self.scaling
        k = k.contiguous().view(_shape).transpose(0, 1)
        v = v.contiguous().view(_shape).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) + attn_bias
        attn_probs = softmax_dropout(attn_weights, self.dropout, self.training)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(n_node, n_graph, embed_dim)
        attn = self.out_proj(attn)
        return attn


class Graphormer3DEncoderLayer(nn.Module):
    """
    Implements a Graphormer-3D Encoder Layer.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.self_attn = SelfMultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
        )
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor = None,
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            attn_bias=attn_bias,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

@torch.jit.script
def compute_pairwise_distances(pos: torch.Tensor):
    """
    Compute pairwise distances between positions.
    Args:
        pos: [batch, n_atoms, 3] positions
    Returns:
        delta_pos: [batch, n_atoms, n_atoms, 3] position differences
        dist: [batch, n_atoms, n_atoms] pairwise distances
    """
    delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
    dist = delta_pos.norm(p=2, dim=-1)
    delta_pos = delta_pos / (dist.unsqueeze(-1) + 1e-5)
    return delta_pos, dist

@torch.jit.script
def mask_edge_features(edge_features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """
    Mask edge features using padding mask.
    Args:
        edge_features: [batch, n_atoms, n_atoms, K] edge features
        padding_mask: [batch, n_atoms] mask (True for padded positions)
    Returns:
        masked edge features
    """
    # Ensure mask lives on the same device as edge_features to avoid
    # CPU/GPU mismatches in TorchScript (OpenMM may execute this on CUDA).
    mask = padding_mask.to(edge_features.device).unsqueeze(1).unsqueeze(-1)
    return edge_features.masked_fill(mask, 0.0)

class GaussianLayer(nn.Module):
    def __init__(self, K=128):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        # Removed edge_types dependency
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x):
        # x is distances
        if x.dim() == 1:
            # Sparse edge list: shape [E]
            x = x.unsqueeze(-1)  # [E, 1]
            # Broadcast to [E, K] by expanding last dim
            x = x.expand(-1, self.K)
        else:
            # Dense matrix: [B, n, n]
            x = x.unsqueeze(-1)
            x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()
        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = self.layer2(x)
        return x

class Graphormer3D(nn.Module):
    def __init__(self, 
                 output_dim,
                 embed_dim=64, 
                 ffn_embed_dim=128, 
                 attention_heads=4, 
                 dropout=0.1, 
                 attention_dropout=0.1, 
                 activation_dropout=0.1, 
                 num_kernel=16, 
                 num_layers=3, 
                 blocks=1,
                 atom_types=64,
                 node_feature_dim=0):
        super().__init__()
        
        self.atom_types = atom_types
        self.node_feature_dim = node_feature_dim
        
        self.atom_encoder = nn.Embedding(
            self.atom_types, embed_dim, padding_idx=0
        )
        
        if self.node_feature_dim > 0:
            self.node_feature_proj = nn.Linear(self.node_feature_dim, embed_dim)
        else:
            self.node_feature_proj = None
        
        self.input_dropout = dropout
        self.layers = nn.ModuleList(
            [
                Graphormer3DEncoderLayer(
                    embed_dim,
                    ffn_embed_dim,
                    num_attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_ln: Callable[[torch.Tensor], torch.Tensor] = nn.LayerNorm(embed_dim)        
        K = num_kernel

        self.gbf: Callable[[torch.Tensor], torch.Tensor] = GaussianLayer(K)
        self.bias_proj: Callable[[torch.Tensor], torch.Tensor] = NonLinear(
            K, attention_heads
        )
        self.edge_proj: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(K, embed_dim)        
        self.blocks = blocks
        self.output_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, atoms: torch.Tensor, pos: torch.Tensor, real_mask: torch.Tensor, node_features: Optional[torch.Tensor] = None):
        # Ensure all floating-point inputs use the same dtype as the model
        # parameters to avoid float32/float64 mismatches when embedded in
        # TorchScript / OpenMM contexts.
        base_dtype = self.atom_encoder.weight.dtype
        pos = pos.to(dtype=base_dtype)
        if node_features is not None:
            node_features = node_features.to(dtype=base_dtype)

        padding_mask = atoms.eq(0)

        n_graph, n_node = atoms.size()
        
        # Use JIT-compiled distance computation
        delta_pos, dist = compute_pairwise_distances(pos)

        # Removed edge_type computation as we don't use it in GaussianLayer anymore
        
        gbf_feature = self.gbf(dist)
        # Use JIT-compiled masking
        edge_features = mask_edge_features(gbf_feature, padding_mask)

        graph_node_feature = (
            self.atom_encoder(atoms)
            + self.edge_proj(edge_features.sum(dim=-2))
        )
        
        if self.node_feature_proj is not None and node_features is not None:
            graph_node_feature = graph_node_feature + self.node_feature_proj(node_features)
        # Removed tag_encoder usage

        # ===== MAIN MODEL =====
        output = F.dropout(
            graph_node_feature, p=self.input_dropout, training=self.training
        )
        output = output.transpose(0, 1).contiguous()

        graph_attn_bias = self.bias_proj(gbf_feature).permute(0, 3, 1, 2).contiguous()
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        for _ in range(self.blocks):
            for enc_layer in self.layers:
                output = enc_layer(output, attn_bias=graph_attn_bias)

        output = self.final_ln(output)
        output = output.transpose(0, 1) # (bsz, n_node, embed_dim)

        # Pooling and Projection for DeepTICA
        # Mask out padding
        output = output * real_mask.unsqueeze(-1)
        # Sum pooling
        output_sum = output.sum(dim=1)
        # Count real nodes
        num_nodes = real_mask.sum(dim=1, keepdim=True).clamp(min=1)
        output_mean = output_sum / num_nodes
        
        z = self.output_proj(output_mean)
        
        return z


class SparseGraphormer3D(nn.Module):
    """
    Graphormer-3D variant with interleaved sparse attention:
      - Local block attention within windows of block_size residues.
      - Strided global links every sparse_stride residues (i±k*sparse_stride).
    This avoids building a full N×N edge embedding for long sequences by
    computing GBF/bias only on the sparse edge set.
    """

    def __init__(
        self,
        output_dim,
        embed_dim=64,
        ffn_embed_dim=128,
        attention_heads=4,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        num_kernel=16,
        num_layers=4,
        blocks=1,
        atom_types=64,
        node_feature_dim=0,
        block_size: int = 20,
        sparse_stride: int = 10,
    ):
        super().__init__()
        self.atom_types = atom_types
        self.node_feature_dim = node_feature_dim
        self.block_size = max(1, int(block_size))
        self.sparse_stride = max(1, int(sparse_stride))
        self.num_heads = attention_heads

        self.atom_encoder = nn.Embedding(self.atom_types, embed_dim, padding_idx=0)
        self.input_dropout = dropout
        self.node_feature_proj = (
            nn.Linear(self.node_feature_dim, embed_dim) if self.node_feature_dim > 0 else None
        )

        self.layers = nn.ModuleList(
            [
                Graphormer3DEncoderLayer(
                    embed_dim,
                    ffn_embed_dim,
                    num_attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_ln: Callable[[torch.Tensor], torch.Tensor] = nn.LayerNorm(embed_dim)
        self.gbf: Callable[[torch.Tensor], torch.Tensor] = GaussianLayer(num_kernel)
        self.bias_proj: Callable[[torch.Tensor], torch.Tensor] = NonLinear(num_kernel, attention_heads)
        self.edge_proj: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(num_kernel, embed_dim)
        self.blocks = blocks
        self.output_proj = nn.Linear(embed_dim, output_dim)

    def _build_sparse_edges(self, n_node: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return edge index tensors (ei, ej) for block + strided links."""
        ei: list = []
        ej: list = []
        bs = self.block_size
        stride = self.sparse_stride
        for i in range(n_node):
            blk_start = (i // bs) * bs
            blk_end = min(blk_start + bs, n_node)
            for j in range(blk_start, blk_end):
                ei.append(i)
                ej.append(j)
            # Forward strides
            j = i + stride
            while j < n_node:
                ei.append(i)
                ej.append(j)
                j += stride
            # Backward strides
            j = i - stride
            while j >= 0:
                ei.append(i)
                ej.append(j)
                j -= stride
        return (
            torch.tensor(ei, device=device, dtype=torch.long),
            torch.tensor(ej, device=device, dtype=torch.long),
        )

    def forward(
        self,
        atoms: torch.Tensor,
        pos: torch.Tensor,
        real_mask: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
    ):
        base_dtype = self.atom_encoder.weight.dtype
        pos = pos.to(dtype=base_dtype)
        if node_features is not None:
            node_features = node_features.to(dtype=base_dtype)

        padding_mask = atoms.eq(0)
        n_graph, n_node = atoms.size()

        # Sparse edge list (shared across batch length n_node)
        edges_i, edges_j = self._build_sparse_edges(n_node, pos.device)

        bias_batches = []
        agg_batches = []
        for b in range(n_graph):
            valid_nodes = (~padding_mask[b]) & (real_mask[b] > 0)
            ei = edges_i
            ej = edges_j
            edge_mask = valid_nodes[ei] & valid_nodes[ej]
            if edge_mask.any():
                ei = ei[edge_mask]
                ej = ej[edge_mask]
            else:
                ei = ej = torch.tensor([], device=pos.device, dtype=torch.long)

            if ei.numel() == 0:
                bias_b = torch.full(
                    (self.num_heads, n_node, n_node),
                    float("-inf"),
                    device=pos.device,
                    dtype=base_dtype,
                )
                agg_b = torch.zeros((n_node, self.gbf.K), device=pos.device, dtype=base_dtype)
                bias_batches.append(bias_b)
                agg_batches.append(agg_b)
                continue

            vec_i = pos[b, ei]
            vec_j = pos[b, ej]
            dist_edge = torch.norm(vec_i - vec_j, dim=-1)
            gbf_edge = self.gbf(dist_edge)  # [E, K]

            bias_edge = self.bias_proj(gbf_edge)  # [E, heads]
            bias_b = torch.full(
                (self.num_heads, n_node, n_node),
                float("-inf"),
                device=pos.device,
                dtype=base_dtype,
            )
            bias_b[:, ei, ej] = bias_edge.transpose(0, 1)

            agg_b = torch.zeros((n_node, gbf_edge.shape[-1]), device=pos.device, dtype=base_dtype)
            agg_b.index_add_(0, ej, gbf_edge)

            bias_batches.append(bias_b)
            agg_batches.append(agg_b)

        graph_attn_bias = torch.stack(bias_batches, dim=0)  # [B, heads, n, n]
        graph_node_feature = self.atom_encoder(atoms) + self.edge_proj(torch.stack(agg_batches, dim=0))

        if self.node_feature_proj is not None and node_features is not None:
            graph_node_feature = graph_node_feature + self.node_feature_proj(node_features)

        output = F.dropout(graph_node_feature, p=self.input_dropout, training=self.training)
        output = output.transpose(0, 1).contiguous()

        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        for _ in range(self.blocks):
            for enc_layer in self.layers:
                output = enc_layer(output, attn_bias=graph_attn_bias)

        output = self.final_ln(output)
        output = output.transpose(0, 1)

        output = output * real_mask.unsqueeze(-1)
        output_sum = output.sum(dim=1)
        num_nodes = real_mask.sum(dim=1, keepdim=True).clamp(min=1)
        output_mean = output_sum / num_nodes
        z = self.output_proj(output_mean)
        return z

# =============================================================================
# DeepTICA Class
# =============================================================================

class DeepTICA(BaseNNCV):
    ''' Neural Network class for DeepTica '''

    def __init__(self, layers, activation='tanh', output='linear', beta_output=1., reg_cholesky=1e-6, model_type='mlp', mode=None, **kwargs):
        super().__init__()
        
        self.model_type: Final[str] = model_type
        # Treat both dense and sparse Graphormer variants as graphormer backends.
        self.is_graphormer: Final[bool] = (model_type in ('graphormer', 'graphormer_sparse'))
        self.graph_stride: Final[int] = int(kwargs.pop("graph_stride", 1) or 1)
        # TorchScript does not like Optional[str] comparisons; keep mode as str.
        self.mode: Final[str] = mode or ''
        self.nn: Optional[nn.Module] = None
        self.graphormer: Optional[Graphormer3D] = None

        if self.model_type == 'mlp':
            # get activation function
            activ = None
            if activation == 'relu':
                activ = torch.nn.ReLU(True)
            elif activation == 'elu':
                activ = torch.nn.ELU(True)
            elif activation == 'tanh':
                activ = torch.nn.Tanh()
            elif activation == 'linear':
                pass
            else:
                raise ValueError("unknown activation. options: 'relu','elu','tanh','linear'. ")

            # Create architecture
            modules = []
            for i in range(len(layers) - 1):
                if i < len(layers) - 2:
                    modules.append(torch.nn.Linear(layers[i], layers[i+1]))
                    if activ is not None:
                        modules.append(activ)
                else:
                    modules.append(torch.nn.Linear(layers[i], layers[i+1]))
            
            self.nn = torch.nn.Sequential(*modules)
            
        elif self.model_type == 'graphormer':
            # kwargs should contain graphormer config
            # We assume layers[-1] is the output dimension
            output_dim = layers[-1]

            # For CA-based modes, allow optional per-residue torsion features
            # as node features (4 dims: sinφ, cosφ, sinψ, cosψ).
            if self.mode in ("ca", "ca_torsion"):
                if "node_feature_dim" not in kwargs:
                    kwargs["node_feature_dim"] = 4

            self.graphormer = Graphormer3D(output_dim=output_dim, **kwargs)
        elif self.model_type == 'graphormer_sparse':
            # Sparse Graphormer variant with block + strided attention
            output_dim = layers[-1]

            if self.mode in ("ca", "ca_torsion"):
                if "node_feature_dim" not in kwargs:
                    kwargs["node_feature_dim"] = 4

            self.graphormer = SparseGraphormer3D(output_dim=output_dim, **kwargs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.output = output
        
        # TICA parameters
        # average of hidden layer
        self.register_buffer('average', torch.zeros(layers[-1]))
        # tica evals and evecs
        self.register_buffer('evals_', torch.zeros(layers[-1]))
        self.register_buffer('evecs_', torch.zeros(layers[-1], layers[-1]))

        self.n_input = layers[0]
        self.beta_output = beta_output
        
        self.reg_cholesky = reg_cholesky
        self.epochs = 0
        self.loss_train = []
        self.loss_valid = []

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input dtype matches the model parameters (e.g., when OpenMM
        # provides double-precision inputs but the network is in float32, or
        # vice versa). We use evecs_/average as a proxy for the network dtype.
        target_dtype = self.evecs_.dtype if hasattr(self, "evecs_") else self.average.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        # Apply stride preprocessing if needed (for graphormer models that expect strided input)
        # This handles cases where get_hidden() is called directly (e.g., from rewards.py)
        # rather than going through DeepTICAProjector.transform()
        if self.is_graphormer and self.graph_stride > 1 and self.mode in ('ca', 'ca_torsion'):
            # Check if input needs striding by comparing with expected input dimension
            expected_dim = self.n_input  # This is the strided dimension set during __init__
            current_dim = x.shape[-1]
            
            if current_dim != expected_dim:
                # Input is not yet strided, apply stride preprocessing
                feat_per_atom = 7 if self.mode == 'ca_torsion' else 3
                if current_dim % feat_per_atom == 0:
                    bsz = x.shape[0] if x.ndim >= 2 else 1
                    if x.ndim == 1:
                        x = x.unsqueeze(0)
                        squeeze_output = True
                    else:
                        squeeze_output = False
                    
                    n_atoms = current_dim // feat_per_atom
                    x = x.view(bsz, n_atoms, feat_per_atom)
                    x = x[:, ::self.graph_stride, :]
                    x = x.reshape(bsz, -1)
                    
                    if squeeze_output:
                        x = x.squeeze(0)

        if self.normIn:
            x = self._normalize(x, self.MeanIn, self.RangeIn)
        
        # Ensure input is at least 2D
        if x.ndim == 1:
            x = x.unsqueeze(0)
            
        if self.is_graphormer:
            # x is expected to be flattened coordinates or coordinates+node features:
            #   - mode='ca'        : [n_atoms * 3]
            #   - mode='ca_torsion': [n_atoms * 7]  -> [x, y, z, sinφ, cosφ, sinψ, cosψ]
            bsz = x.size(0)
            n_features = x.size(1)
            
            node_features: Optional[torch.Tensor] = None
            
            if self.mode == 'ca':
                # CA mode: use only aligned CA coordinates as positions. If the
                # input accidentally carries interleaved torsions (7 per atom),
                # we drop the extra 4 features and keep positions only.
                if n_features % 3 == 0:
                    n_atoms = n_features // 3
                    pos = x.view(bsz, n_atoms, 3)
                    node_features = None
                elif n_features % 7 == 0:
                    n_atoms = n_features // 7
                    x_reshaped = x.view(bsz, n_atoms, 7)
                    pos = x_reshaped[:, :, :3]
                    node_features = None
                else:
                    raise ValueError(f"Input features size {n_features} is not divisible by 3 (coords) or 7 (coords+torsions) for mode='ca'.")
            elif self.mode == 'ca_torsion':
                # CA+torsion mode: interpret input as [x, y, z, sinφ, cosφ, sinψ, cosψ].
                # We keep the runtime shape check for normal execution, but skip
                # it under TorchScript tracing to avoid TracerWarnings about
                # Tensor-dependent control flow.
                if not torch.jit.is_tracing():
                    if n_features % 7 != 0:
                        raise ValueError(
                            f"Input features size {n_features} is not divisible by 7 for mode='ca_torsion'."
                        )
                n_atoms = n_features // 7
                x_reshaped = x.view(bsz, n_atoms, 7)
                pos = x_reshaped[:, :, :3]
                node_features = x_reshaped[:, :, 3:]
            else:
                n_atoms = n_features // 3
                pos = x.view(bsz, n_atoms, 3)
            
            # Stride is applied during preprocessing in fit(), not here during forward pass.
            # Keeping this block commented out to document the decision.
            # if self.graph_stride > 1:
            #     n_atoms_before = n_atoms
            #     pos = pos[:, ::self.graph_stride, :]
            #     if node_features is not None:
            #         node_features = node_features[:, ::self.graph_stride, :]
            #     n_atoms = pos.shape[1]
            
            # Create atom type and mask tensors on the same device as x in a
            # way that TorchScript tracing does not hard-code a specific device.
            atoms = x.new_ones((bsz, n_atoms), dtype=torch.long)
            real_mask = x.new_ones((bsz, n_atoms), dtype=torch.bool)
            
            if self.graphormer is None:
                raise RuntimeError("Graphormer model requested but graphormer module is not initialized.")
            z = self.graphormer(atoms, pos, real_mask, node_features=node_features)
        else:
            if self.nn is None:
                raise RuntimeError("MLP model requested but nn module is not initialized.")
            z = self.nn(x)

        # Ensure downstream TICA buffers (average/evecs_) see a consistent dtype.
        if z.dtype != target_dtype:
            z = z.to(target_dtype)
        
        if self.output == 'softmax':
            z = torch.nn.functional.softmax(torch.mul(z, self.beta_output), dim=1)
        elif self.output == 'sigmoid':
            z = torch.sigmoid(torch.mul(z, self.beta_output))
            
        return z

    def _apply_tica(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.evecs_)

    def _remove_average(self, x: torch.Tensor) -> torch.Tensor:
        return x.sub(self.average)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.get_hidden(x)
        z = self._remove_average(z)
        z = self._apply_tica(z)
        if self.normOut:
            z = self._normalize(z, self.MeanOut, self.RangeOut)
        return z

    def compute_correlation_matrix(self, x, y, w, symmetrize=True):
        d = x.shape[1]
        # C = x.T @ (w * y) / sum(w)
        corr = torch.einsum('ij, ik, i -> jk', x, y, w)
        norm = torch.sum(w)
        corr /= norm
        if symmetrize:
            corr = 0.5 * (corr + corr.T)
        return corr

    def _inv_sqrt(self, x, epsilon=1e-10, mode='regularize'):
        '''
        Compute the inverse square root of a matrix using eigenvalue decomposition.
        Modes:
        - 'regularize': Add epsilon to diagonal (default in geom2vec)
        - 'trunc': Filter out eigenvalues < epsilon

        This implementation is made robust against ill-conditioned inputs by
        falling back to an SVD-based inverse square root if torch.linalg.eigh
        fails to converge.
        '''
        def _svd_inv_sqrt(mat, eps: float) -> torch.Tensor:
            # mat is assumed symmetric; SVD is robust even for ill-conditioned matrices.
            U, S, _ = torch.linalg.svd(mat, full_matrices=False)
            S_clamped = torch.clamp(S, min=eps)
            inv_sqrt = torch.diag(torch.sqrt(1.0 / S_clamped))
            return U @ inv_sqrt @ U.T

        if mode == 'regularize':
            x_reg = x + epsilon * torch.eye(x.shape[0], device=x.device)
            try:
                eigvals, eigvecs = torch.linalg.eigh(x_reg)
                # Ensure positive eigenvalues
                eigvals = torch.abs(eigvals)
            except RuntimeError as exc:
                # Fallback: SVD-based inverse square root
                print(f"[DeepTICA] Warning: eigh failed in _inv_sqrt(regularize, eps={epsilon:.1e}); "
                      f"falling back to SVD-based inverse sqrt. Error: {exc}")
                return _svd_inv_sqrt(x_reg, epsilon)
            
        elif mode == 'trunc':
            try:
                eigvals, eigvecs = torch.linalg.eigh(x)
            except RuntimeError as exc:
                print(f"[DeepTICA] Warning: eigh failed in _inv_sqrt(trunc, eps={epsilon:.1e}); "
                      f"falling back to SVD-based inverse sqrt. Error: {exc}")
                return _svd_inv_sqrt(x, epsilon)
            # Filter out small eigenvalues
            mask = eigvals > epsilon
            if not torch.any(mask):
                # All eigenvalues are below threshold; fall back to SVD
                print("[DeepTICA] Warning: all eigenvalues below epsilon in _inv_sqrt(trunc); "
                      "falling back to SVD-based inverse sqrt.")
                return _svd_inv_sqrt(x, epsilon)
            eigvals = eigvals[mask]
            eigvecs = eigvecs[:, mask]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Inverse square root of eigenvalues
        inv_sqrt_eigvals = torch.diag(torch.sqrt(1.0 / eigvals))
        
        # Reconstruct: E @ D^{-1/2} @ E.T
        x_inv_sqrt = torch.matmul(eigvecs, torch.matmul(inv_sqrt_eigvals, eigvecs.T))
        
        return x_inv_sqrt

    def solve_tica_eigenproblem(self, C_00, C_01, C_11, n_eig=0, save_params=False):
        '''
        Solve the TICA eigenproblem using VAMP-2 approach (SVD on whitened matrix).
        V = C_00^{-1/2} C_01 C_11^{-1/2}
        '''
        # Use reg_cholesky as epsilon if provided, otherwise default
        epsilon = self.reg_cholesky if self.reg_cholesky is not None else 1e-6
        # Use regularize mode by default as in geom2vec, or make it configurable?
        # For now hardcode 'regularize' to match geom2vec default, or add self.mode
        mode = 'regularize' 
        
        C_00_inv_sqrt = self._inv_sqrt(C_00, epsilon=epsilon, mode=mode)
        C_11_inv_sqrt = self._inv_sqrt(C_11, epsilon=epsilon, mode=mode)
        
        # VAMP matrix
        V = torch.matmul(C_00_inv_sqrt, torch.matmul(C_01, C_11_inv_sqrt))
        
        # SVD of V
        try:
            U, S, Vh = torch.linalg.svd(V, full_matrices=False)
        except RuntimeError:
            # Fallback to eigh if SVD fails
            print("[DeepTICA] SVD failed, falling back to eigh on symmetrized matrix")
            V_sym = 0.5 * (V + V.T)
            eigvals, eigvecs = torch.linalg.eigh(V_sym)
            # Sort descending
            idx = torch.argsort(eigvals, descending=True)
            S = eigvals[idx]
            U = eigvecs[:, idx]
        
        eigvals = S
        
        # Eigenvectors (projectors)
        # We need W = C_00^{-1/2} U
        eigvecs = torch.matmul(C_00_inv_sqrt, U)
        
        if n_eig > 0:
            eigvals = eigvals[:n_eig]
            eigvecs = eigvecs[:, :n_eig]
        
        if save_params:
            self.evals_.data = eigvals.detach()
            self.evecs_.data = eigvecs.detach()
    
        return eigvals

    def forward_and_tica(self, data, n_eig=0, return_corr=False, save_params=False):
        x_t, x_lag, w_t, w_lag = data   
        f_t = self.get_hidden(x_t)
        f_lag = self.get_hidden(x_lag)
        
        # Separate centering using w_lag (consistent with VAMPnet ops.py)
        sum_w = torch.sum(w_lag)
        ave_f_t = torch.einsum('ij,i ->j', f_t, w_lag) / sum_w
        ave_f_lag = torch.einsum('ij,i ->j', f_lag, w_lag) / sum_w
        
        f_t_centered = f_t.sub(ave_f_t)
        f_lag_centered = f_lag.sub(ave_f_lag)
        
        # Compute Covariances using w_lag
        # C_00 and C_11 should be symmetrized (auto-covariance)
        # C_01 should NOT be symmetrized (cross-covariance)
        C_00 = self.compute_correlation_matrix(f_t_centered, f_t_centered, w_lag, symmetrize=True)
        C_01 = self.compute_correlation_matrix(f_t_centered, f_lag_centered, w_lag, symmetrize=False)
        C_11 = self.compute_correlation_matrix(f_lag_centered, f_lag_centered, w_lag, symmetrize=True)
        
        evals = self.solve_tica_eigenproblem(C_00, C_01, C_11, n_eig=n_eig, save_params=save_params) 
        
        if save_params:
            self.average.data = ave_f_t.detach()

        if return_corr:
            return evals, C_01
        else:
            return evals

    def loss_function(self, evals, objective='sum2', n_eig=0):
        if n_eig > 0 and len(evals) < n_eig:
            raise ValueError("n_eig must be lower than the number of eigenvalues.")
        elif n_eig == 0:
            n_eig = len(evals)

        if objective == 'sum':
            loss = -torch.sum(evals[:n_eig])
        elif objective == 'sum2':
            loss = -torch.sum(torch.pow(evals[:n_eig], 2))
        elif objective == 'gap':
            loss = -(evals[0] - evals[1])
        elif objective == 'single':
            loss = -evals[n_eig-1]
        elif objective == 'single2':
            loss = -torch.pow(evals[n_eig-1], 2)
        else:
            raise ValueError("unknown objective")

        return loss

    def train_epoch(self, dataset, loss_type, n_eig):
        for batch in dataset: 
            # Move batch to device
            batch = [b.to(self.average.device) for b in batch]
            evals = self.forward_and_tica(batch, n_eig=n_eig, save_params=True) 
            loss = self.loss_function(evals, objective=loss_type, n_eig=n_eig)
            
            self.opt_.zero_grad()
            loss.backward()
            self.opt_.step()
        self.epochs += 1
    
    def evaluate_dataset(self, data, loss_type, n_eig, save_params=False):   
        with torch.no_grad():
            # Move data to device
            data = [d.to(self.average.device) for d in data]
            evals = self.forward_and_tica(data, n_eig=n_eig, save_params=save_params)
            loss = self.loss_function(evals, objective=loss_type, n_eig=n_eig)       
        return loss.item()

# =============================================================================
# Dataset Helpers
# =============================================================================

from bisect import bisect_left

def closest_idx_torch(array, value):
        '''
        Find index of the element of 'array' which is closest to 'value'.
        The array is first converted to a np.array in case of a tensor.
        Note: it does always round to the lowest one.

        Parameters:
            array (tensor/np.array)
            value (float)

        Returns:
            pos (int): index of the closest value in array
        '''
        if type(array) is np.ndarray:
            pos = bisect_left(array, value)
        else:
            pos = bisect_left(array.numpy(), value)
        if pos == 0:
            return 0
        elif pos == len(array):
            return -1
        else:
            return pos-1

def look_for_configurations(x,t,lag):
    '''
    Searches for all the pairs which are distant 'lag' in time, and returns the weights associated to lag=lag as well as the weights for lag=0.

    Parameters:
        x (tensor): array whose columns are the descriptors and rows the time evolution
        time (tensor): array with the simulation time
        lag (float): lag-time

    Returns:
        x_t (tensor): array of descriptors at time t
        x_lag (tensor): array of descriptors at time t+lag
        w_t (tensor): weights at time t
        w_lag (tensor): weights at time t+lag
    '''
    # Guard against empty or too-short trajectories (e.g., zero-length segments
    # when some walkers produced no frames). We need at least 2 time points to
    # form any lagged pair.
    try:
        t_len = len(t)
    except Exception:
        t_len = 0
    try:
        x_len = len(x)
    except Exception:
        x_len = 0
    if t is None or t_len < 2 or x is None or x_len < 2:
        empty = torch.tensor([], dtype=torch.float32)
        return empty, empty, empty, empty

    #define lists
    x_t = []
    x_lag = []
    w_t = []
    w_lag = []
    #find maximum time idx
    idx_end = closest_idx_torch(t,t[-1]-lag)
    start_j = 0
    #loop over time array and find pairs which are far away by lag
    for i in range(idx_end):
        stop_condition = lag+t[i+1]
        n_j = 0
        for j in range(i,len(t)):
            if ( t[j] < stop_condition ) and (t[j+1]>t[i]+lag):
                x_t.append(x[i])
                x_lag.append(x[j])
                deltaTau=min(t[i+1]+lag,t[j+1]) - max(t[i]+lag,t[j])
                w_lag.append(deltaTau)
                if n_j == 0: #assign j as the starting point for the next loop
                    start_j = j
                n_j +=1
            elif t[j] > stop_condition:
                break
        for k in range(n_j):
            w_t.append((t[i+1]-t[i])/float(n_j))

    # Convert to tensor if not empty
    if len(x_t) > 0:
        x_t = torch.stack(x_t) if isinstance(x_t[0], torch.Tensor) else torch.tensor(x_t, dtype=torch.float32)
        x_lag = torch.stack(x_lag) if isinstance(x_lag[0], torch.Tensor) else torch.tensor(x_lag, dtype=torch.float32)
        w_t = torch.tensor(w_t, dtype=torch.float32)
        w_lag = torch.tensor(w_lag, dtype=torch.float32)
    else:
        # Return empty tensors
        x_t = torch.tensor([], dtype=torch.float32)
        x_lag = torch.tensor([], dtype=torch.float32)
        w_t = torch.tensor([], dtype=torch.float32)
        w_lag = torch.tensor([], dtype=torch.float32)

    return x_t,x_lag,w_t,w_lag

def create_tica_dataset(x, t, lag_time, n_train, n_valid, every=1, batch_tr=-1):
    # Split
    x_train = x[:n_train*every:every]
    t_train = t[:n_train*every:every]
    
    x_valid = x[n_train*every : (n_train+n_valid)*every : every]
    t_valid = t[n_train*every : (n_train+n_valid)*every : every]
    
    # Create pairs
    # Note: x and t should be tensors
    if not isinstance(x_train, torch.Tensor):
        x_train = torch.tensor(x_train, dtype=torch.float32)
        t_train = torch.tensor(t_train, dtype=torch.float32)
    if not isinstance(x_valid, torch.Tensor):
        x_valid = torch.tensor(x_valid, dtype=torch.float32)
        t_valid = torch.tensor(t_valid, dtype=torch.float32)

    # This function is now deprecated by the new DeepTICAProjector.fit logic
    # train_configs = look_for_configurations(x_train, t_train, lag=lag_time)
    # valid_configs = look_for_configurations(x_valid, t_valid, lag=lag_time)
    
    # Batching
    # if batch_tr == -1:
    #     batch_tr = len(train_configs[0])
    
    # Split into batches
    # n_samples = len(train_configs[0])
    # train_batches = []
    # if n_samples > 0:
    #     for i in range(0, n_samples, batch_tr):
    #         batch = [
    #             train_configs[0][i:i+batch_tr],
    #             train_configs[1][i:i+batch_tr],
    #             train_configs[2][i:i+batch_tr],
    #             train_configs[3][i:i+batch_tr]
    #         ]
    #         train_batches.append(batch)
            
    # return train_batches, valid_configs
    raise NotImplementedError("create_tica_dataset is deprecated. Use DeepTICAProjector.fit directly.")

# =============================================================================
# Projector Wrapper for GESPO
# =============================================================================

class DeepTICAProjector:
    """
    Wrapper for DeepTICA to be used in PCAScorer.
    Exposes fit(data) and transform(data).

    The `fuzzy` flag controls how downstream code should interpret the
    DeepTICA outputs for sampling / rewards:
      - fuzzy=False (default): prefer hard argmax assignments over VAMP
        states when drawing samples.
      - fuzzy=True: prefer using the raw softmax probabilities (soft
        state weights) for sampling / weighting.

    Note: this flag does not change the DeepTICA training objective;
    it is a behavioural hint for consumers of the projector
    (e.g., rewards, discrete start policy).
    """
    def __init__(self, n_components=2, lagtime=10, n_epochs=100, max_steps=10000, val_steps=200, patience=5, batch_size=500, device='cpu', weighted=False, reg_cholesky=1e-6, model_type='mlp', mode=None, fuzzy: bool = False, freeze_encoder: bool = False, **kwargs):
        self.n_components = n_components
        self.lagtime = lagtime
        self.n_epochs = n_epochs  # Kept for backward compatibility, but max_steps takes precedence
        self.max_steps = max_steps
        self.val_steps = val_steps
        self.patience = patience  # Early stopping: stop if no improvement for this many validations
        self.batch_size = batch_size
        self.device = device
        self.weighted = weighted
        self.reg_cholesky = reg_cholesky
        self.model_type = model_type
        self.mode = mode
        # When False (default), downstream code should treat DeepTICA
        # VAMP outputs as hard state assignments (via argmax) for
        # sampling; when True, it should use the full softmax
        # distribution (fuzzy states).
        self.fuzzy = bool(fuzzy)
        # Snapshot stride for Graphormer inputs (capture before kwargs mutation).
        self.graph_stride = int(kwargs.get("graph_stride", 1) or 1)

        # Allow optional validation split configuration via kwargs without
        # passing them down into the underlying DeepTICA model. Default:
        # use 20% of trajectories for validation to monitor VAMP score.
        self.val_fraction = float(kwargs.pop("val_fraction", 0.2) or 0.0)
        self.min_val_trajectories = int(kwargs.pop("min_val_trajectories", 1) or 1)
        # Keep an internal copy so in-model pops (e.g., graph_stride) don't erase
        # the value for subsequent refreshes.
        self.model_kwargs = dict(kwargs)
        self.model = None
        self.optimizer = None # Added optimizer attribute
        self.input_dim = None
        self.last_val_vamp2 = None  # Track validation VAMP-2 from last fit
        self.last_val_loss = None   # Track best validation loss from last fit
        self.last_best_step = None  # Track best validation step from last fit
        self.last_effective_batch_size = None  # Effective batch size after OOM back-off
        self.freeze_encoder = freeze_encoder
        self.is_anchoring = False  # Track whether we're in anchoring mode
        self._anchoring_encoder_state = None  # Store encoder weights from anchoring

        # Graphormer backend currently supports CA-only, CA+torsion, and heavy-atom modes.
        if self.model_type == 'graphormer' and self.mode not in ['ca', 'ca_torsion', 'heavy']:
            raise ValueError(
                "Graphormer model is only allowed in 'ca', 'ca_torsion', or 'heavy' mode, "
                f"but got '{self.mode}'."
            )

    def fit(self, X: List[np.ndarray], times: Optional[List[np.ndarray]] = None, is_anchoring: bool = False):
        """
        Fit the DeepTICA model.
        
        Parameters
        ----------
        X : List[np.ndarray]
            List of trajectories (arrays of shape [T, features]).
        times : List[np.ndarray], optional
        X: List of trajectories (arrays) OR single array (pooled data).
        is_anchoring : bool, optional
            Whether this fit is during the anchoring round (default: False).
            When freeze_encoder=True and is_anchoring=False, the encoder will be frozen.
        """
        # Update anchoring state
        self.is_anchoring = is_anchoring
        
        # Handle single array input (pooled data) -> treat as one trajectory
        if isinstance(X, np.ndarray):
            X = [X]
            if times is not None and isinstance(times, np.ndarray):
                times = [times]

        # If using Graphormer with stride>1 on CA/CA_torsion inputs, downsample
        # the residue axis up front so input_dim reflects the strided graph.
        # Only do this once; the model forward will no longer stride.
        if self.model_type in ("graphormer", "graphormer_sparse") and not getattr(self, "_graph_stride_applied", False):
            stride = int(self.graph_stride)
            if stride > 1 and self.mode in ("ca", "ca_torsion"):
                feat_per_atom = 7 if self.mode == "ca_torsion" else 3
                new_X = []
                n_atoms_before = None
                for traj in X:
                    if traj.shape[1] % feat_per_atom != 0:
                        new_X.append(traj)
                        continue
                    n_atoms = traj.shape[1] // feat_per_atom
                    if n_atoms_before is None:
                        n_atoms_before = n_atoms
                    traj_rs = traj.reshape(traj.shape[0], n_atoms, feat_per_atom)
                    traj_rs = traj_rs[:, ::stride, :]
                    traj_ds = traj_rs.reshape(traj.shape[0], -1)
                    new_X.append(traj_ds.astype(traj.dtype, copy=False))
                X = new_X
                self._graph_stride_applied = True
                n_atoms_after = (n_atoms_before + stride - 1) // stride if n_atoms_before else None
                print(f"[DeepTICA] Applied graph_stride={stride} during preprocessing (mode={self.mode}): {n_atoms_before} atoms -> {n_atoms_after} atoms")
        
        # Basic data checks
        if not X:
            print("[DeepTICA] Warning: No data provided for training.")
            return self
            
        # Initialize model if needed
        input_dim = X[0].shape[-1]
        print(f"[DeepTICA Debug] X type: {type(X)}, len: {len(X) if isinstance(X, list) else 'N/A'}")
        print(f"[DeepTICA Debug] X[0] shape: {X[0].shape}, input_dim: {input_dim}")
        
        # Check coverage of existing states with new data before reinitializing
        if self.model is not None and getattr(self.model, 'output', 'linear') == 'softmax':
            print("[DeepTICA] Checking coverage of existing states with new data...")
            self.model.eval()
            total_counts = None
            n_states = self.n_components
            
            try:
                with torch.no_grad():
                    for traj in X:
                        # Convert to tensor
                        t_tens = torch.tensor(traj, dtype=torch.float32, device=self.device)
                        
                        # Run get_hidden
                        probs = self.model.get_hidden(t_tens)
                        
                        # Hard assignment
                        preds = torch.argmax(probs, dim=1)
                        counts = torch.bincount(preds, minlength=n_states)
                        
                        if total_counts is None:
                            total_counts = counts
                        else:
                            if counts.shape[0] > total_counts.shape[0]:
                                # Resize total_counts if needed
                                new_total = torch.zeros_like(counts)
                                new_total[:total_counts.shape[0]] = total_counts
                                total_counts = new_total
                            elif counts.shape[0] < total_counts.shape[0]:
                                # Resize counts
                                new_counts = torch.zeros_like(total_counts)
                                new_counts[:counts.shape[0]] = counts
                                counts = new_counts
                                
                            total_counts += counts
                
                if total_counts is not None:
                    zeros = (total_counts == 0).nonzero()
                    if zeros.numel() > 0:
                        zero_indices = zeros.flatten().tolist()
                        print(f"[DeepTICA] WARNING: The following clusters have NO configurations in the new data: {zero_indices}")
                        print(f"[DeepTICA] This might cause instability or collapse during refitting.")
                    else:
                        print(f"[DeepTICA] All {len(total_counts)} clusters are covered by at least one configuration.")
            except Exception as exc_check:
                print(f"[DeepTICA] Failed to check state coverage: {exc_check}")

        # Reinitialize model when needed, but preserve encoder weights if freeze_encoder is enabled.
        # For post-anchoring refreshes (is_anchoring=False), we intentionally start
        # from scratch at each fit so that every DeepTICA refit is independent of
        # previous refresh weights. For the anchoring round, we keep the original
        # "reinit only if needed" behavior.
        if is_anchoring:
            should_reinit = (self.model is None or self.input_dim != input_dim)
        else:
            should_reinit = True
        
        # When freeze_encoder=True and not in anchoring mode, we want to:
        # 1. Reinitialize the full model architecture
        # 2. But restore the encoder weights from anchoring
        # This ensures TICA eigenvalue layers are fresh while encoder is frozen
        if should_reinit:
            self.input_dim = input_dim
            # Default architecture: [input, 64, 64, n_components]
            layers = [self.input_dim, 64, 64, self.n_components]
            # Ensure graph_stride is passed through even if it was popped in a prior init.
            model_kwargs = dict(self.model_kwargs)
            model_kwargs.setdefault("graph_stride", self.graph_stride)
            self.model = DeepTICA(layers, reg_cholesky=self.reg_cholesky, model_type=self.model_type, mode=self.mode, **model_kwargs).to(self.device)
            print(f"[DeepTICA] Reinitialized model with layers {layers}")
            
            # Restore encoder weights if available and not in anchoring mode
            # Only applicable to Graphormer: restore encoder, keep output_proj fresh
            if self.freeze_encoder and not is_anchoring and self._anchoring_encoder_state is not None:
                try:
                    if self.model_type in ('graphormer', 'graphormer_sparse') and self.model.graphormer is not None:
                        # Load all parameters except output_proj
                        current_state = self.model.graphormer.state_dict()
                        for key in self._anchoring_encoder_state:
                            if 'output_proj' not in key:
                                current_state[key] = self._anchoring_encoder_state[key].to(self.device)
                        self.model.graphormer.load_state_dict(current_state)
                        n_restored = sum(1 for k in self._anchoring_encoder_state if 'output_proj' not in k)
                        print(f"[DeepTICA] Restored Graphormer encoder weights ({n_restored} params, excluding output_proj MLP) from anchoring")
                    elif self.model_type == 'mlp':
                        # MLP mode: no encoder restoration needed
                        print(f"[DeepTICA] Skipping encoder restoration in MLP mode")
                except Exception as exc:
                    print(f"[DeepTICA] Warning: Failed to restore encoder weights: {exc}")
            
        # Prepare data for training
        # We need to generate time-lagged pairs from the trajectories.
        # To guard against overfitting, we use a random subset of whole trajectories for validation.
        # This avoids intra-trajectory correlation leakage (since trajectories are short).
        
        x_t_train_list = []
        x_lag_train_list = []
        w_t_train_list = []
        w_lag_train_list = []

        x_t_val_list = []
        x_lag_val_list = []
        w_t_val_list = []
        w_lag_val_list = []
        
        # Collect all valid trajectory data first
        all_traj_data = []
        for i, traj in enumerate(X):
            t_vals = times[i] if times is not None else None
            traj_np = traj.astype(np.float32)
            
            x_t, x_lag, w_t, w_lag = look_for_configurations(
                traj_np,
                t_vals if t_vals is not None else np.arange(len(traj_np)),
                self.lagtime,
            )
            
            if x_t.numel() > 0:
                all_traj_data.append((x_t, x_lag, w_t, w_lag))

        # Determine split based on number of valid trajectories
        n_valid = len(all_traj_data)
        
        # Random Trajectory Split
        # We shuffle the list of trajectories and pick a subset for validation.
        # This ensures:
        # 1. Randomness: The validation set covers the entire state space (unlike Chronological).
        # 2. No Intra-Trajectory Leakage: We don't split within a trajectory (unlike Frame-wise).
        # Note: This may include trajectories seen by previous epochs' models in the validation set.
        # This is a necessary trade-off to ensure the validation set is representative of the
        # global kinetics, preventing the "overly hard" OOD problem of chronological splitting.
        
        # Use a dynamic seed or global state to ensure different splits across refreshes
        indices = np.random.permutation(n_valid)
        
        n_val = 0
        if self.val_fraction > 0.0 and n_valid > 0:
            n_val = int(n_valid * self.val_fraction)
            # Enforce min_val_trajectories
            if n_val < self.min_val_trajectories and n_valid >= self.min_val_trajectories:
                n_val = self.min_val_trajectories
            # Ensure we don't take everything for validation
            if n_val >= n_valid:
                n_val = n_valid - 1
        
        val_indices = set(indices[:n_val])
        
        for i in range(n_valid):
            data = all_traj_data[i]
            if i in val_indices:
                x_t_val_list.append(data[0])
                x_lag_val_list.append(data[1])
                w_t_val_list.append(data[2])
                w_lag_val_list.append(data[3])
            else:
                x_t_train_list.append(data[0])
                x_lag_train_list.append(data[1])
                w_t_train_list.append(data[2])
                w_lag_train_list.append(data[3])

        # If we ended up with no training pairs but some validation pairs,
        # fall back to using validation data as training to avoid a hard error.
        if not x_t_train_list and x_t_val_list:
            x_t_train_list, x_lag_train_list = x_t_val_list, x_lag_val_list
            w_t_train_list, w_lag_train_list = w_t_val_list, w_lag_val_list
            x_t_val_list, x_lag_val_list, w_t_val_list, w_lag_val_list = [], [], [], []

        if not x_t_train_list:
            print("[DeepTICA] Warning: No valid pairs found (trajectories too short or lagtime too large?).")
            return self

        # Concatenate all training pairs
        x_t_data = torch.cat(x_t_train_list, dim=0)
        x_lag_data = torch.cat(x_lag_train_list, dim=0)
        w_t_data = torch.cat(w_t_train_list, dim=0)
        w_lag_data = torch.cat(w_lag_train_list, dim=0)

        # Concatenate validation pairs (if any)
        x_t_val = x_lag_val = w_t_val = w_lag_val = None
        if x_t_val_list:
            x_t_val = torch.cat(x_t_val_list, dim=0)
            x_lag_val = torch.cat(x_lag_val_list, dim=0)
            w_t_val = torch.cat(w_t_val_list, dim=0)
            w_lag_val = torch.cat(w_lag_val_list, dim=0)
            # Shuffle validation pairs so that early-stopping metrics are not
            # biased by deterministic ordering (trajectory concat + time order).
            # Keep this shuffle fixed for the whole fit call for stable
            # early stopping.
            if x_t_val.shape[0] > 1:
                perm = torch.randperm(x_t_val.shape[0])
                x_t_val = x_t_val[perm]
                x_lag_val = x_lag_val[perm]
                w_t_val = w_t_val[perm]
                w_lag_val = w_lag_val[perm]
        
        # Handle weighting override
        if not self.weighted:
            w_t_data = torch.ones_like(w_t_data)
            w_lag_data = torch.ones_like(w_lag_data)
            if w_t_val is not None:
                w_t_val = torch.ones_like(w_t_val)
            if w_lag_val is not None:
                w_lag_val = torch.ones_like(w_lag_val)
            
        # Standardize inputs (MeanIn/RangeIn)
        # We compute stats on x_t_data (all data)
        self.model.standardize_inputs(x_t_data)
        # Ensure model buffers are on device
        self.model.to(self.device)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(x_t_data, x_lag_data, w_t_data, w_lag_data)
        
        # Train loop with OOM-aware batch size back-off (important for Graphormer inputs)
        current_bs = max(1, int(self.batch_size))
        print(f"[DeepTICA] Starting fit with requested batch size {self.batch_size}, effective {current_bs}")
        self.last_effective_batch_size = current_bs
        trained = False
        while not trained:
            loader = torch.utils.data.DataLoader(dataset, batch_size=current_bs, shuffle=True)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

            self.model.train()
            best_val_loss = None
            best_val_vamp2 = None
            best_state = None
            best_step = 0
            steps_without_improvement = 0  # Early stopping counter

            try:
                # Freeze encoder if configured and not in anchoring mode
                # freeze_encoder only applies to Graphormer models (geometric encoder)
                # For MLP, the entire network should remain trainable
                encoder_frozen = False
                if self.freeze_encoder and not self.is_anchoring:
                    if self.model_type in ('graphormer', 'graphormer_sparse') and self.model.graphormer is not None:
                        # For Graphormer: freeze all encoder layers, keep output_proj trainable
                        # output_proj is the MLP head on top of the geometric encoder
                        for name, param in self.model.graphormer.named_parameters():
                            if 'output_proj' in name:
                                param.requires_grad = True
                            else:
                                param.requires_grad = False
                        encoder_frozen = True
                        n_frozen = sum(1 for p in self.model.graphormer.parameters() if not p.requires_grad)
                        n_trainable = sum(1 for p in self.model.graphormer.parameters() if p.requires_grad)
                        print(f"[DeepTICA] Graphormer encoder frozen ({n_frozen} params), output_proj MLP trainable ({n_trainable} params)")
                    elif self.model_type == 'mlp':
                        # For MLP mode: freeze_encoder has no effect (no separate geometric encoder)
                        # The entire MLP network should remain trainable for TICA eigenvalue fitting
                        print(f"[DeepTICA] freeze_encoder=True has no effect in MLP mode (entire network remains trainable)")
                else:
                    # Ensure all parameters are trainable (unfreeze in case frozen before)
                    if self.model_type == 'mlp' and self.model.nn is not None:
                        for param in self.model.nn.parameters():
                            param.requires_grad = True
                    elif self.model_type in ('graphormer', 'graphormer_sparse') and self.model.graphormer is not None:
                        for param in self.model.graphormer.parameters():
                            param.requires_grad = True
                    if self.freeze_encoder:
                        print(f"[DeepTICA] Full model training enabled for anchoring round (freeze_encoder=True, is_anchoring={self.is_anchoring})")
                
                step = 0
                total_loss = 0.0
                steps_since_val = 0

                # Create an infinite iterator for step-based training
                from itertools import cycle
                infinite_loader = cycle(loader)

                while step < self.max_steps:
                    batch = next(infinite_loader)
                    batch_x_t, batch_x_lag, batch_w_t, batch_w_lag = [b.to(self.device) for b in batch]
                    optimizer.zero_grad()
                    evals = self.model.forward_and_tica(
                        (batch_x_t, batch_x_lag, batch_w_t, batch_w_lag),
                        n_eig=self.n_components,
                        save_params=True,
                    )
                    loss = -torch.sum(evals**2)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    step += 1
                    steps_since_val += 1

                    # Standardize outputs periodically (every validation interval)
                    if steps_since_val >= self.val_steps:
                        with torch.no_grad():
                            self.model.standardize_outputs(x_t_data.to(self.device))

                    # Evaluate VAMP-2 score every val_steps on a held-out validation subset.
                    if steps_since_val >= self.val_steps:
                        avg_loss = total_loss / steps_since_val
                        steps_since_val = 0
                        total_loss = 0.0

                        n_eval = 0
                        try:
                            if x_t_val is not None and x_t_val.numel() > 0:
                                x_src = x_t_val
                                x_lag_src = x_lag_val
                                w_t_src = w_t_val
                                w_lag_src = w_lag_val
                                n_eval = min(len(x_src), current_bs)
                            else:
                                x_src = x_t_data
                                x_lag_src = x_lag_data
                                w_t_src = w_t_data
                                w_lag_src = w_lag_data
                                n_eval = min(len(x_src), current_bs)
                            if n_eval > 0:
                                val_subset = [
                                    x_src[:n_eval].to(self.device),
                                    x_lag_src[:n_eval].to(self.device),
                                    w_t_src[:n_eval].to(self.device),
                                    w_lag_src[:n_eval].to(self.device),
                                ]
                                evals_val = self.model.forward_and_tica(
                                    val_subset,
                                    n_eig=self.n_components,
                                    save_params=True,
                                )
                                val_loss = -torch.sum(evals_val**2).item()
                                val_vamp2 = torch.sum(evals_val**2).item()  # VAMP-2 score (positive)
                                self.last_val_vamp2 = val_vamp2  # Store for external access

                                evals_np = evals_val.detach().cpu().numpy()
                                ev_str = " ".join(f"{float(e):.3f}" for e in evals_np)
                                print(
                                    f"[DeepTICA] Step {step}/{self.max_steps} "
                                    f"AvgLoss: {avg_loss:.4f}  ValLoss: {val_loss:.4f}  VAMP-2: {val_vamp2:.4f}  Eigenvalues: {ev_str}"
                                )

                                if (best_val_loss is None) or (val_loss < best_val_loss):
                                    best_val_loss = val_loss
                                    best_val_vamp2 = val_vamp2
                                    best_step = step
                                    best_state = {
                                        k: v.detach().cpu().clone()
                                        for k, v in self.model.state_dict().items()
                                    }
                                    steps_without_improvement = 0  # Reset counter on improvement
                                else:
                                    steps_without_improvement += 1  # Increment if no improvement
                                    if steps_without_improvement >= self.patience:
                                        print(f"[DeepTICA] Early stopping at step {step}: no improvement for {self.patience} validations")
                                        break  # Exit training loop
                        except Exception as exc_log:
                            print(f"[DeepTICA] Logging failed at step {step}: {exc_log}")

                trained = True
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                new_bs = max(1, current_bs // 2)
                if new_bs == current_bs:
                    raise
                print(f"[DeepTICA] OOM during training; reducing batch size from {current_bs} to {new_bs} and retrying.")
                current_bs = new_bs
                self.last_effective_batch_size = current_bs
                continue
            except RuntimeError as exc_rt:
                # TorchScript dropout OOM often surfaces as RuntimeError, not OutOfMemoryError.
                if "out of memory" in str(exc_rt).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    new_bs = max(1, current_bs // 2)
                    if new_bs == current_bs:
                        raise
                    print(f"[DeepTICA] Runtime OOM during training; reducing batch size from {current_bs} to {new_bs} and retrying.")
                    current_bs = new_bs
                    self.last_effective_batch_size = current_bs
                    continue
                raise

        # Restore the best-validation model snapshot if we recorded one.
        if best_state is not None:
            try:
                self.model.load_state_dict(best_state)
                self.model.to(self.device)
                # Restore the VAMP-2 score from the best epoch (not the last epoch)
                if best_val_vamp2 is not None:
                    self.last_val_vamp2 = best_val_vamp2
                self.last_val_loss = best_val_loss
                self.last_best_step = best_step
                print(f"[DeepTICA] Restored best-validation model (ValLoss={best_val_loss:.4f}, VAMP-2={best_val_vamp2:.4f}).")
            except Exception as exc_load:
                print(f"[DeepTICA] Failed to restore best-validation model: {exc_load}")
        
        # Save encoder weights after successful anchoring training (for freeze_encoder mode)
        # Only applicable to Graphormer models - exclude the output_proj MLP head
        if is_anchoring and self.freeze_encoder:
            try:
                if self.model_type in ('graphormer', 'graphormer_sparse') and self.model.graphormer is not None:
                    # Save all parameters except output_proj (the MLP head)
                    full_state = self.model.graphormer.state_dict()
                    self._anchoring_encoder_state = {
                        k: v.cpu().clone() for k, v in full_state.items() if 'output_proj' not in k
                    }
                    print(f"[DeepTICA] Saved Graphormer encoder state (excluding output_proj) from anchoring ({len(self._anchoring_encoder_state)} parameters)")
                elif self.model_type == 'mlp':
                    # MLP mode: no separate geometric encoder, skip saving
                    print(f"[DeepTICA] Skipping encoder state save in MLP mode (no separate geometric encoder)")
            except Exception as exc:
                print(f"[DeepTICA] Warning: Failed to save encoder state: {exc}")
        
        # Store validation features for potential re-evaluation (fair comparison across refreshes)
        # Save the raw input features (x_t_val) that were used during training
        if x_t_val is not None and x_t_val.numel() > 0:
            self.X_val = x_t_val.detach().cpu().numpy()
            self.X_lag_val = x_lag_val.detach().cpu().numpy()
            self.w_t_val = w_t_val.detach().cpu().numpy() if self.weighted else None
            self.w_lag_val = w_lag_val.detach().cpu().numpy() if self.weighted else None
        else:
            self.X_val = None
            self.X_lag_val = None
            self.w_t_val = None
            self.w_lag_val = None
        
        # Persist the effective batch size actually used.
        self.batch_size = current_bs
        
        self.model.eval()
        return self

    def compute_validation_vamp2(self):
        """
        Compute VAMP-2 score on stored validation set.
        Used for re-evaluating old models on new validation data.
        """
        if self.model is None:
            print("[DeepTICA] Error: Model not fitted.")
            return None
        
        if self.X_val is None or self.X_lag_val is None:
            print("[DeepTICA] Warning: No validation data stored.")
            return None
        
        self.model.eval()
        with torch.no_grad():
            # Convert to tensor if stored as numpy (handles both cases)
            if isinstance(self.X_val, torch.Tensor):
                x_t = self.X_val.to(self.device)
                x_lag = self.X_lag_val.to(self.device)
            else:
                x_t = torch.from_numpy(self.X_val).to(self.device)
                x_lag = torch.from_numpy(self.X_lag_val).to(self.device)
            
            # Handle weights
            if self.w_t_val is not None:
                if isinstance(self.w_t_val, torch.Tensor):
                    w_t = self.w_t_val.to(self.device)
                    w_lag = self.w_lag_val.to(self.device)
                else:
                    w_t = torch.from_numpy(self.w_t_val).to(self.device)
                    w_lag = torch.from_numpy(self.w_lag_val).to(self.device)
            else:
                w_t = torch.ones(x_t.shape[0], device=self.device)
                w_lag = torch.ones(x_lag.shape[0], device=self.device)
            
            evals = self.model.forward_and_tica(
                (x_t, x_lag, w_t, w_lag),
                n_eig=self.n_components,
                save_params=False,
            )
            vamp2 = torch.sum(evals**2).item()
        return vamp2
    
    def transform(self, X: Union[List[np.ndarray], np.ndarray]):
        """
        Project data into TICA space.
        X: List of trajectories (arrays) OR single array (pooled data).
        Returns: List of arrays OR single array (matching input).
        """
        if self.model is None:
            print("[DeepTICA] Error: Model not fitted.")
            return None
        
        self.model.eval()
        
        # Handle single array input
        is_single = isinstance(X, np.ndarray)
        if is_single:
            X = [X]
        
        # Apply the same graph_stride preprocessing that was used during fit()
        if self.model_type in ("graphormer", "graphormer_sparse") and getattr(self, "_graph_stride_applied", False):
            stride = int(self.graph_stride)
            if stride > 1 and self.mode in ("ca", "ca_torsion"):
                feat_per_atom = 7 if self.mode == "ca_torsion" else 3
                new_X = []
                for traj in X:
                    if traj.shape[1] % feat_per_atom != 0:
                        new_X.append(traj)
                        continue
                    n_atoms = traj.shape[1] // feat_per_atom
                    traj_rs = traj.reshape(traj.shape[0], n_atoms, feat_per_atom)
                    traj_rs = traj_rs[:, ::stride, :]
                    traj_ds = traj_rs.reshape(traj.shape[0], -1)
                    new_X.append(traj_ds.astype(traj.dtype, copy=False))
                X = new_X
        
        # Use batching to avoid GPU OOM for long trajectories.
        max_bs = max(1, int(getattr(self, "batch_size", 1000) or 1000))
        all_proj = []
        for traj in X:
            traj_torch = torch.tensor(traj, dtype=torch.float32).to(self.device)
            proj_chunks = []
            with torch.no_grad():
                for start in range(0, traj_torch.shape[0], max_bs):
                    end = start + max_bs
                    proj = self.model(traj_torch[start:end])
                    proj_chunks.append(proj.cpu())
            all_proj.append(torch.cat(proj_chunks, dim=0).numpy())
            
        if is_single:
            return all_proj[0]
        return all_proj

    def plot_2d(self, X: List[np.ndarray], save_path: str = "tica_projection.png", title: str = "DeepTICA Projection"):
        """
        Plot the first two TICA components.
        """
        
        projs = self.transform(X)
        if projs is None:
            return
        
        # Concatenate for plotting
        data = np.concatenate(projs, axis=0)
        
        if data.shape[1] < 2:
            print("[DeepTICA] Warning: Less than 2 components, cannot plot 2D.")
            return
            
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
        plt.xlabel("TIC 1")
        plt.ylabel("TIC 2")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[DeepTICA] Projection plot saved to {save_path}")
    
    def __getstate__(self):
        """Custom serialization to handle PyTorch models properly."""
        state = self.__dict__.copy()
        # Save model state_dict instead of the model object itself
        if self.model is not None:
            state['_model_state_dict'] = {k: v.cpu() for k, v in self.model.state_dict().items()}
            state['model'] = None
        # Optimizer is not needed for checkpointing (will be recreated)
        state['optimizer'] = None
        return state
    
    def __setstate__(self, state):
        """Custom deserialization to restore PyTorch models."""
        # Restore model state if it was saved
        model_state = state.pop('_model_state_dict', None)
        self.__dict__.update(state)
        
        # Reconstruct model if we have saved state
        if model_state is not None and self.input_dim is not None:
            try:
                # Drop normalization buffers whose shapes may legitimately
                # change when the input/output dimensionality changes between
                # runs. These will be recomputed by standardize_inputs /
                # standardize_outputs after fitting.
                filtered_state = {}
                dropped_norm_keys = []
                for k, v in model_state.items():
                    base_name = k.split('.')[-1]
                    if base_name in ("MeanIn", "RangeIn", "MeanOut", "RangeOut"):
                        dropped_norm_keys.append(k)
                        continue
                    filtered_state[k] = v
                if dropped_norm_keys:
                    print(
                        "[DeepTICA] Skipping normalization buffers in checkpoint "
                        f"due to potential shape mismatch: {', '.join(dropped_norm_keys)}"
                    )

                layers = [self.input_dim, 64, 64, self.n_components]
                model_kwargs = dict(self.model_kwargs)
                model_kwargs.setdefault("graph_stride", self.graph_stride)
                self.model = DeepTICA(
                    layers,
                    reg_cholesky=self.reg_cholesky,
                    model_type=self.model_type,
                    mode=self.mode,
                    **model_kwargs
                ).to(self.device)
                # Use strict=False so that any remaining non-critical keys
                # that do not match (e.g., from older versions) are ignored.
                self.model.load_state_dict(filtered_state, strict=False)
                print(f"[DeepTICA] Restored model from checkpoint")
            except Exception as exc:
                print(f"[DeepTICA] Warning: Failed to restore model state: {exc}")
                self.model = None
