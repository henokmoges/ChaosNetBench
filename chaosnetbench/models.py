"""
ChaosNetBench Model Library

Self-contained implementations of the 13 benchmark models for the
Coupled Standard Map forecasting benchmark.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────
# DLinear: Trend-Seasonal Decomposition + Linear Projection
# ─────────────────────────────────────────────────────────


class _DLinearDecompose(nn.Module):
    """Moving-average trend/seasonal decomposition + linear projection.

    Input:  x [B, L_in, N]
    Output: y [B, L_out, N]
    """

    def __init__(self, seq_len: int, pred_len: int, kernel_size: int = 25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.kernel_size = kernel_size
        self.Linear_Trend = nn.Linear(seq_len, pred_len)
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)

    def _moving_average(self, x: torch.Tensor) -> torch.Tensor:
        ks = self.kernel_size
        if ks <= 1:
            return x
        pad = (ks - 1) // 2
        weight = torch.ones(1, 1, ks, device=x.device) / ks
        b, l, n = x.shape
        x_ = x.permute(0, 2, 1).reshape(b * n, 1, l)
        trend = torch.conv1d(F.pad(x_, (pad, pad), mode="replicate"), weight)
        trend = trend.squeeze(1).reshape(b, n, l).permute(0, 2, 1)
        return trend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self._moving_average(x)
        seasonal = x - trend
        out = self.Linear_Trend(trend.transpose(1, 2)).transpose(1, 2)
        out = out + self.Linear_Seasonal(seasonal.transpose(1, 2)).transpose(1, 2)
        return out


# ─────────────────────────────────────────────────────────
# TCN: Temporal Convolutional Network (Per-Node, No Graph)
# ─────────────────────────────────────────────────────────


class TCN(nn.Module):
    """Dilated causal temporal convolutional network for time series forecasting.

    Processes each node independently (no graph component).
    Serves as the "GWNet minus graph" ablation: isolates the TCN temporal backbone
    from GWNet's diffusion graph convolution.

    Architecture:
      - Gated dilated causal Conv1d layers (dilation = 2^i)
      - Skip connections aggregated across layers
      - Linear readout to pred_len

    Reference: Adapted from the WaveNet / Graph WaveNet temporal backbone.

    Input:  x [B, seq_len, N]
    Output: y [B, pred_len, N]
    """

    def __init__(
        self,
        n_nodes: int,
        seq_len: int,
        pred_len: int,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels

        # Input projection: per-node 1 → hidden_channels
        self.input_conv = nn.Conv1d(1, hidden_channels, kernel_size=1)

        # Gated dilated causal convolutions + skip connections
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        receptive_field = 1
        for i in range(n_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation  # causal padding
            receptive_field += (kernel_size - 1) * dilation

            self.filter_convs.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                          dilation=dilation, padding=padding)
            )
            self.gate_convs.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                          dilation=dilation, padding=padding)
            )
            self.skip_convs.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
            )
            self.residual_convs.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
            )
            self.bn.append(nn.BatchNorm1d(hidden_channels))

        self.receptive_field = receptive_field
        self.dropout = nn.Dropout(dropout)

        # Output: hidden_channels → pred_len via 2-layer 1x1 conv
        self.end_conv1 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=1)
        self.end_conv2 = nn.Conv1d(hidden_channels * 2, pred_len, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, N]
        B, T, N = x.shape

        # Process each node independently: reshape to [B*N, 1, T]
        h = x.transpose(1, 2).reshape(B * N, 1, T)  # [B*N, 1, T]
        h = self.input_conv(h)  # [B*N, C, T]

        skip_sum = 0
        for i in range(self.n_layers):
            residual = h

            # Gated temporal convolution with causal trimming
            f = self.filter_convs[i](h)[..., :T]  # trim to causal length
            g = self.gate_convs[i](h)[..., :T]
            h_tc = torch.tanh(f) * torch.sigmoid(g)  # [B*N, C, T]
            h_tc = self.dropout(h_tc)

            # Skip connection
            skip_sum = skip_sum + self.skip_convs[i](h_tc)

            # Residual connection
            h = residual + self.residual_convs[i](h_tc)
            h = self.bn[i](h)

        # Output: aggregate skip, take last timestep, project to pred_len
        out = F.relu(skip_sum[..., -1:])  # [B*N, C, 1]
        out = F.relu(self.end_conv1(out))  # [B*N, 2C, 1]
        out = self.end_conv2(out)  # [B*N, pred_len, 1]
        out = out.squeeze(-1)  # [B*N, pred_len]

        # Reshape back: [B*N, pred_len] → [B, N, pred_len] → [B, pred_len, N]
        out = out.reshape(B, N, self.pred_len).transpose(1, 2)
        return out  # [B, pred_len, N]

    def get_learned_adjacency(self) -> None:
        """No adjacency to extract (per-node model)."""
        return None

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total}


# ─────────────────────────────────────────────────────────
# LowRankTopKAdjacency: Learnable Graph Structure
# ─────────────────────────────────────────────────────────



class DLinearBaseline(nn.Module):
    """DLinear baseline: moving-average decomposition with per-node linear projection."""

    def __init__(self, seq_len: int, pred_len: int, n_nodes: int):
        super().__init__()
        self.backbone = _DLinearDecompose(seq_len=seq_len, pred_len=pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total}


# ─────────────────────────────────────────────────────────
# Baseline: MLP
# ─────────────────────────────────────────────────────────


class LSTMBaseline(nn.Module):
    """LSTM baseline for sequence-to-sequence forecasting.

    Encodes input sequence with LSTM, decodes with linear projection.
    No graph structure — temporal-only baseline.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_nodes: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.lstm = nn.LSTM(
            input_size=n_nodes,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, pred_len * n_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        _, (h_n, _) = self.lstm(x)  # h_n: [n_layers, B, hidden]
        h_last = h_n[-1]  # [B, hidden]
        y_flat = self.fc(h_last)  # [B, pred_len * n_nodes]
        return y_flat.reshape(B, self.pred_len, self.n_nodes)

    def get_learned_adjacency(self) -> None:
        return None

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total}


# ─────────────────────────────────────────────────────────
# Oracle STGNN: Uses Ground-Truth Adjacency
# ─────────────────────────────────────────────────────────


class OracleGCN(nn.Module):
    """Oracle upper-bound model using ground-truth adjacency with a GCN backbone.

    Architecture:
      - Temporal embedding: Linear(seq_len → d_model)
      - GCN layers with ground-truth A (2-hop)
      - Output projection: Linear(d_model → pred_len)

    Input:  x [B, seq_len, N]
    Output: y [B, pred_len, N]
    """

    def __init__(
        self,
        n_nodes: int,
        seq_len: int,
        pred_len: int,
        A_true: torch.Tensor,
        d_model: int = 32,
        n_gcn_layers: int = 2,
        dropout: float = 0.1,
        self_loop_alpha: float = 0.2,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Fixed ground-truth adjacency (symmetric normalization: D^{-1/2} A D^{-1/2})
        A_norm = A_true.float() + self_loop_alpha * torch.eye(n_nodes)
        D = A_norm.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D + 1e-6))
        A_norm = D_inv_sqrt @ A_norm @ D_inv_sqrt
        self.register_buffer("A_fixed", A_norm)

        # Temporal embedding: [B, N, seq_len] → [B, N, d_model]
        self.temp_embed = nn.Linear(seq_len, d_model)

        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_norms = nn.ModuleList()
        for _ in range(n_gcn_layers):
            self.gcn_layers.append(nn.Linear(d_model, d_model))
            self.gcn_norms.append(nn.LayerNorm(d_model))

        self.dropout = nn.Dropout(dropout)

        # Output: [B, N, d_model] → [B, N, pred_len]
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, N] → [B, N, seq_len]
        h = x.transpose(1, 2)  # [B, N, seq_len]
        h = self.temp_embed(h)  # [B, N, d_model]

        # GCN layers: h = σ(A_norm @ h @ W)
        for gcn, norm in zip(self.gcn_layers, self.gcn_norms):
            h_agg = torch.einsum("ij,bjd->bid", self.A_fixed, h)  # [B, N, d_model]
            h_new = gcn(h_agg)  # [B, N, d_model]
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            h = norm(h + h_new)  # residual + norm

        # Output projection
        out = self.output_proj(h)  # [B, N, pred_len]
        return out.transpose(1, 2)  # [B, pred_len, N]

    def get_learned_adjacency(self) -> np.ndarray:
        """Returns the fixed ground-truth adjacency."""
        return self.A_fixed.cpu().numpy()

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ─────────────────────────────────────────────────────────
# iTransformer: Variate-centric Transformer (ICLR 2024)
# ─────────────────────────────────────────────────────────


class iTransformer(nn.Module):
    """iTransformer baseline — inverted transformer for TSF.

    Key idea: treats each variate (node) as a token, applies attention
    across variates rather than across time. This makes the attention
    mechanism architecturally analogous to STGNN spatial modules.

    Reference: Liu et al., "iTransformer: Inverted Transformers Are
    Effective for Time Series Forecasting" (ICLR 2024).

    Input:  x [B, seq_len, N]
    Output: y [B, pred_len, N]
    """

    def __init__(
        self,
        n_nodes: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        # Embed each variate's time series into d_model
        self.variate_embed = nn.Linear(seq_len, d_model)

        # Transformer encoder (attention across variates)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Project back to prediction horizon
        self.output_proj = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, N]
        # Invert: treat variates as tokens with seq_len features
        x_inv = x.transpose(1, 2)  # [B, N, seq_len]

        # Embed each variate
        tokens = self.variate_embed(x_inv)  # [B, N, d_model]

        # Self-attention across variates
        encoded = self.encoder(tokens)  # [B, N, d_model]

        # Project to prediction horizon
        out = self.output_proj(encoded)  # [B, N, pred_len]
        return out.transpose(1, 2)  # [B, pred_len, N]

    def get_learned_adjacency(self) -> Optional[np.ndarray]:
        """Extract attention weights as implicit adjacency."""
        # Run a forward pass through encoder to get attention
        # For evaluation, we extract the last layer's attention
        return None  # iTransformer doesn't learn an explicit graph

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total}


# ─────────────────────────────────────────────────────────
# MTGNN: Graph Learning + Mix-Hop + Dilated Inception
# ─────────────────────────────────────────────────────────


class NBEATSBaseline(nn.Module):
    """N-BEATS baseline for time series forecasting.

    Architecture: stack of fully-connected blocks. Each block produces
    backcast (reconstruction of input) and forecast contributions.
    Final forecast is the sum of all block forecasts.

    Simplified generic version (no interpretable stacks).

    Reference: Oreshkin et al., "N-BEATS: Neural basis expansion analysis
    for interpretable time series forecasting" (ICLR 2020).

    Input:  x [B, seq_len, N]
    Output: y [B, pred_len, N]
    """

    def __init__(
        self,
        n_nodes: int,
        seq_len: int,
        pred_len: int,
        n_stacks: int = 2,
        n_blocks: int = 3,
        hidden_dim: int = 128,
        theta_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.stacks = nn.ModuleList()
        for _ in range(n_stacks):
            blocks = nn.ModuleList()
            for _ in range(n_blocks):
                blocks.append(
                    _NBEATSBlock(seq_len, pred_len, hidden_dim, theta_dim, dropout)
                )
            self.stacks.append(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, N]
        B, L, N = x.shape

        # Process each node independently
        forecast = torch.zeros(B, self.pred_len, N, device=x.device)
        backcast = x  # residual input

        for blocks in self.stacks:
            for block in blocks:
                # Process per-node: [B, N, seq_len]
                bc = backcast.transpose(1, 2)  # [B, N, seq_len]
                bc_flat = bc.reshape(B * N, self.seq_len)

                b, f = block(bc_flat)
                b = b.reshape(B, N, self.seq_len).transpose(1, 2)
                f = f.reshape(B, N, self.pred_len).transpose(1, 2)

                backcast = backcast - b
                forecast = forecast + f

        return forecast

    def get_learned_adjacency(self) -> None:
        return None

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total}


class _NBEATSBlock(nn.Module):
    """Single N-BEATS block: FC stack → backcast + forecast."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        hidden_dim: int,
        theta_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.fc_stack = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Theta parameters for basis expansion
        self.theta_b = nn.Linear(hidden_dim, theta_dim)
        self.theta_f = nn.Linear(hidden_dim, theta_dim)

        # Basis expansion to actual backcast/forecast
        self.backcast_basis = nn.Linear(theta_dim, seq_len)
        self.forecast_basis = nn.Linear(theta_dim, pred_len)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B*N, seq_len]
        h = self.fc_stack(x)
        backcast = self.backcast_basis(self.theta_b(h))
        forecast = self.forecast_basis(self.theta_f(h))
        return backcast, forecast


# ─────────────────────────────────────────────────────────
# AGCRN: Adaptive Graph Convolutional Recurrent Network
# (Bai et al., NeurIPS 2020)
#
# Verified against official repo: github.com/LeiBAI/AGCRN
# and TSL library: github.com/TorchSpatiotemporal/tsl
# See D33.4 in DECISIONS_LOG.md for three-way comparison.
# ─────────────────────────────────────────────────────────


class _AVWGCN(nn.Module):
    """Adaptive Virtual Weight Graph Convolution (Bai et al., NeurIPS 2020).

    Matches the official AGCRN implementation (LeiBAI/AGCRN/model/AGCN.py):
    - Chebyshev polynomial support set: [I, A, 2A·T1 - T0, ...]
    - Node-adaptive weights: einsum('nd,dkio->nkio', E, weights_pool)
    - Node-adaptive bias: E @ bias_pool
    """

    def __init__(self, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out)
        )
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weights_pool.size(-1))
        self.weights_pool.data.uniform_(-stdv, stdv)
        self.bias_pool.data.zero_()

    def forward(
        self, x: torch.Tensor, node_embeddings: torch.Tensor, supports: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, C_in]
            node_embeddings: [N, embed_dim]
            supports: [cheb_k, N, N] precomputed Chebyshev support set
        Returns:
            [B, N, C_out]
        """
        # Node-adaptive weights and bias
        weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)

        # Graph convolution: supports @ x → [B, K, N, C_in]
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)  # [B, N, K, C_in]

        # Apply node-adaptive weights: [B, N, C_out]
        x_gconv = torch.einsum("bnki,nkio->bno", x_g, weights) + bias
        return x_gconv


class _AGCRNCell(nn.Module):
    """Single AGCRN cell matching official AGCRNCell (LeiBAI/AGCRN).

    Uses two AVWGCN modules:
    - gate: produces reset (z) and update (r) gates
    - update: produces candidate hidden state
    """

    def __init__(
        self, dim_in: int, dim_out: int, n_nodes: int, cheb_k: int, embed_dim: int
    ):
        super().__init__()
        self.hidden_dim = dim_out
        self.gate = _AVWGCN(dim_in + dim_out, 2 * dim_out, cheb_k, embed_dim)
        self.update = _AVWGCN(dim_in + dim_out, dim_out, cheb_k, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        node_embeddings: torch.Tensor,
        supports: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, input_dim]
            state: [B, N, hidden_dim]
            node_embeddings: [N, embed_dim]
            supports: [cheb_k, N, N]
        Returns:
            h_new: [B, N, hidden_dim]
        """
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings, supports))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size: int, n_nodes: int, device: torch.device):
        return torch.zeros(batch_size, n_nodes, self.hidden_dim, device=device)


class AGCRNModel(nn.Module):
    """Adaptive Graph Convolutional Recurrent Network.

    Faithful reimplementation of Bai et al. (NeurIPS 2020), verified against
    official repo (LeiBAI/AGCRN) and TSL library. Extended with adj_mode
    parameter for controlled ablation studies (D33.2).

    Key differences from original:
    - adj_mode parameter enables ring/zero/learned adjacency for trichotomy
    - Supports are precomputed once per forward pass (not per-cell)
    - Layer processing follows official pattern (layer-first: all T for L0, then L1)

    Input:  x [B, seq_len, N]
    Output: y [B, pred_len, N]
    """

    def __init__(
        self,
        n_nodes: int,
        seq_len: int,
        pred_len: int,
        embed_dim: int = 10,
        rnn_units: int = 32,
        n_layers: int = 2,
        cheb_k: int = 2,
        dropout: float = 0.1,
        adj_mode: str = "learned",
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.rnn_units = rnn_units
        self.n_layers = n_layers
        self.adj_mode = adj_mode
        self.cheb_k = cheb_k

        # Node embeddings for adaptive graph
        self.node_embeddings = nn.Parameter(
            torch.randn(n_nodes, embed_dim), requires_grad=True
        )

        # Fixed adjacency for ablation modes (non-trainable)
        if adj_mode == "ring":
            A_fixed = torch.zeros(n_nodes, n_nodes)
            for i in range(n_nodes):
                A_fixed[i, (i - 1) % n_nodes] = 0.5
                A_fixed[i, (i + 1) % n_nodes] = 0.5
            self.register_buffer("fixed_adj", A_fixed)
        elif adj_mode == "random":
            A_fixed = torch.softmax(torch.rand(n_nodes, n_nodes), dim=-1)
            self.register_buffer("fixed_adj", A_fixed)
        else:
            self.register_buffer("fixed_adj", None)

        # Encoder: stacked AGCRN cells (matches official AVWDCRNN)
        self.encoder_cells = nn.ModuleList()
        input_dim = 1  # each node has 1 feature per timestep
        self.encoder_cells.append(
            _AGCRNCell(input_dim, rnn_units, n_nodes, cheb_k, embed_dim)
        )
        for _ in range(1, n_layers):
            self.encoder_cells.append(
                _AGCRNCell(rnn_units, rnn_units, n_nodes, cheb_k, embed_dim)
            )

        # Output: Conv2d matching official (1 → horizon*output_dim)
        self.end_conv = nn.Conv2d(
            1, pred_len, kernel_size=(1, rnn_units), bias=True
        )

    def _compute_supports(self, A: torch.Tensor) -> torch.Tensor:
        """Build Chebyshev support set [I, A, 2A·T1-T0, ...].

        Matches official: support_set = [I, A, 2*A@support[-1] - support[-2]]
        Returns: [cheb_k, N, N]
        """
        N = A.size(0)
        support_set = [torch.eye(N, device=A.device), A]
        for k in range(2, self.cheb_k):
            support_set.append(
                torch.matmul(2 * A, support_set[-1]) - support_set[-2]
            )
        return torch.stack(support_set[:self.cheb_k], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, N]
        B, T, N = x.shape

        # Compute adjacency (mode-dependent)
        if self.adj_mode == "learned":
            A = F.softmax(
                F.relu(torch.mm(self.node_embeddings, self.node_embeddings.t())),
                dim=1,
            )
        elif self.adj_mode == "zero":
            A = torch.zeros(N, N, device=x.device)
        else:  # ring or random: use pre-built fixed adj
            A = self.fixed_adj

        # Precompute Chebyshev supports once
        supports = self._compute_supports(A)

        # Layer-first processing (matches official AVWDCRNN):
        # Process all T timesteps through layer i, then pass outputs to layer i+1
        current_inputs = x.unsqueeze(-1)  # [B, T, N, 1]

        for i, cell in enumerate(self.encoder_cells):
            state = torch.zeros(B, N, self.rnn_units, device=x.device)
            inner_states = []
            for t in range(T):
                state = cell(
                    current_inputs[:, t, :, :], state, self.node_embeddings, supports
                )
                inner_states.append(state)
            current_inputs = torch.stack(inner_states, dim=1)  # [B, T, N, hidden]

        # Output from last timestep of last layer: [B, 1, N, hidden]
        output = current_inputs[:, -1:, :, :]
        # Conv2d: [B, 1, N, hidden] → [B, pred_len, N, 1] → [B, pred_len, N]
        output = self.end_conv(output).squeeze(-1)
        return output

    def get_learned_adjacency(self) -> np.ndarray:
        """Extract adaptive adjacency from node embeddings."""
        with torch.no_grad():
            if self.adj_mode == "learned":
                A = F.softmax(
                    F.relu(torch.mm(self.node_embeddings, self.node_embeddings.t())),
                    dim=1,
                )
            elif self.adj_mode == "zero":
                A = torch.zeros(self.n_nodes, self.n_nodes)
            else:
                A = self.fixed_adj
            A = A - torch.diag(torch.diag(A))
            return A.cpu().numpy()

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total}


# ─────────────────────────────────────────────────────────
# Graph WaveNet (Wu et al., IJCAI 2019)
# ─────────────────────────────────────────────────────────


class GraphWaveNet(nn.Module):
    """Graph WaveNet for multivariate time series forecasting.

    Key components:
      1. Adaptive adjacency matrix via node embeddings (self-attention style)
      2. Gated dilated causal convolution for temporal patterns
      3. Diffusion graph convolution (forward + backward)
      4. Skip connections across layers

    Reference: Wu et al., "Graph WaveNet for Deep Spatial-Temporal
    Graph Modeling" (IJCAI 2019).

    Input:  x [B, seq_len, N]
    Output: y [B, pred_len, N]
    """

    def __init__(
        self,
        n_nodes: int,
        seq_len: int,
        pred_len: int,
        residual_channels: int = 16,
        dilation_channels: int = 16,
        skip_channels: int = 32,
        end_channels: int = 64,
        n_layers: int = 4,
        adj_embed_dim: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_layers = n_layers
        self.skip_channels = skip_channels

        # Adaptive adjacency: E1 @ E2^T → softmax
        self.node_embed1 = nn.Parameter(torch.randn(n_nodes, adj_embed_dim) * 0.1)
        self.node_embed2 = nn.Parameter(torch.randn(n_nodes, adj_embed_dim) * 0.1)

        # Input projection: [B, 1, N, T] → [B, residual_channels, N, T]
        self.input_conv = nn.Conv2d(1, residual_channels, kernel_size=(1, 1))

        # WaveNet layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        # Diffusion GCN weights (forward + backward)
        self.gc_forward = nn.ModuleList()
        self.gc_backward = nn.ModuleList()

        receptive_field = 1
        for i in range(n_layers):
            dilation = 2 ** i
            receptive_field += dilation

            # Gated temporal convolution
            self.filter_convs.append(
                nn.Conv2d(residual_channels, dilation_channels, (1, 2), dilation=(1, dilation))
            )
            self.gate_convs.append(
                nn.Conv2d(residual_channels, dilation_channels, (1, 2), dilation=(1, dilation))
            )

            # Graph convolution (1x1 conv after spatial aggregation)
            self.gc_forward.append(
                nn.Conv2d(dilation_channels, residual_channels, (1, 1))
            )
            self.gc_backward.append(
                nn.Conv2d(dilation_channels, residual_channels, (1, 1))
            )

            self.residual_convs.append(
                nn.Conv2d(dilation_channels, residual_channels, (1, 1))
            )
            self.skip_convs.append(
                nn.Conv2d(dilation_channels, skip_channels, (1, 1))
            )
            self.bn.append(nn.BatchNorm2d(residual_channels))

        self.receptive_field = receptive_field

        # Output layers (matches official: end_conv2 projects to pred_len)
        self.end_conv1 = nn.Conv2d(skip_channels, end_channels, (1, 1))
        self.end_conv2 = nn.Conv2d(end_channels, pred_len, (1, 1))

        self.dropout = nn.Dropout(dropout)

    def _compute_adaptive_adj(self):
        """Compute adaptive adjacency from node embeddings."""
        A = F.softmax(F.relu(self.node_embed1 @ self.node_embed2.t()), dim=1)
        return A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, N] → [B, 1, N, seq_len]
        B, T, N = x.shape
        h = x.transpose(1, 2).unsqueeze(1)  # [B, 1, N, T]

        # Pad temporally if needed
        if T < self.receptive_field:
            h = F.pad(h, (self.receptive_field - T, 0))
            T_padded = self.receptive_field
        else:
            T_padded = T

        h = self.input_conv(h)  # [B, residual_channels, N, T]

        A_adaptive = self._compute_adaptive_adj()  # [N, N]
        A_T = A_adaptive.t()

        skip_sum = torch.zeros(B, self.skip_channels, N, T_padded, device=x.device)

        for i in range(self.n_layers):
            residual = h

            # Gated temporal convolution
            filter_out = torch.tanh(self.filter_convs[i](h))
            gate_out = torch.sigmoid(self.gate_convs[i](h))
            h_tc = filter_out * gate_out  # [B, dilation_channels, N, T']
            h_tc = self.dropout(h_tc)

            # Diffusion graph convolution (forward + backward)
            # Forward: A @ h_tc
            h_fwd = torch.einsum("ij,bcjt->bcit", A_adaptive, h_tc)
            h_fwd = self.gc_forward[i](h_fwd)

            # Backward: A^T @ h_tc
            h_bwd = torch.einsum("ij,bcjt->bcit", A_T, h_tc)
            h_bwd = self.gc_backward[i](h_bwd)

            h_gc = h_fwd + h_bwd  # [B, residual_channels, N, T']

            # Skip connection
            skip = self.skip_convs[i](h_tc)
            skip_sum = skip_sum[..., -skip.size(-1):] + skip

            # Residual connection
            h_res = self.residual_convs[i](h_tc)
            h = residual[..., -h_res.size(-1):] + h_gc
            h = self.bn[i](h)

        # Output: take last timestep from skip accumulation (matches official GWNet)
        out = F.relu(skip_sum[..., -1:])  # [B, skip_channels, N, 1]
        out = F.relu(self.end_conv1(out))  # [B, end_channels, N, 1]
        out = self.end_conv2(out)  # [B, pred_len, N, 1]
        return out.squeeze(-1)  # [B, pred_len, N]

    def get_learned_adjacency(self) -> np.ndarray:
        """Extract learned adaptive adjacency."""
        with torch.no_grad():
            A = self._compute_adaptive_adj()
            A = A - torch.diag(torch.diag(A))
            return A.cpu().numpy()

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total}


# ─────────────────────────────────────────────────────────
# ESN: Echo State Network (Reservoir Computing Baseline)
# ─────────────────────────────────────────────────────────


class PatchTST(nn.Module):
    """PatchTST — channel-independent patched transformer for TSF.

    Splits each variate's time series into patches, projects them into
    tokens, and applies a standard transformer encoder per variate.
    Channel-independent: no cross-variate attention (each node processed
    separately with shared weights).

    Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term
    Forecasting with Transformers" (ICLR 2023).

    Input:  x [B, seq_len, N]
    Output: y [B, pred_len, N]
    """

    def __init__(
        self,
        n_nodes: int,
        seq_len: int,
        pred_len: int,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride

        # Number of patches
        self.n_patches = max(1, (seq_len - patch_len) // stride + 1)

        # Patch embedding
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)

        # Output head: flatten encoded patches → pred_len
        self.head = nn.Linear(self.n_patches * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, N]
        B, L, N = x.shape

        # Channel-independent: reshape to [B*N, seq_len]
        x_ci = x.permute(0, 2, 1).reshape(B * N, L)  # [B*N, seq_len]

        # Extract patches: [B*N, n_patches, patch_len]
        patches = x_ci.unfold(dimension=1, size=self.patch_len, step=self.stride)

        # Embed patches
        tokens = self.patch_embed(patches) + self.pos_embed  # [B*N, n_patches, d_model]

        # Transformer encode
        encoded = self.encoder(tokens)  # [B*N, n_patches, d_model]
        encoded = self.layer_norm(encoded)

        # Flatten and project to pred_len
        flat = encoded.reshape(B * N, -1)  # [B*N, n_patches * d_model]
        out = self.head(flat)  # [B*N, pred_len]

        # Reshape back: [B, N, pred_len] → [B, pred_len, N]
        return out.reshape(B, N, self.pred_len).permute(0, 2, 1)

    def get_learned_adjacency(self) -> Optional[np.ndarray]:
        return None  # Channel-independent — no graph structure

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total}


# ─────────────────────────────────────────────────────────
# NHiTS: Neural Hierarchical Interpolation for TSF
# ─────────────────────────────────────────────────────────


class STAEformer(nn.Module):
    """STAEformer — Spatio-Temporal Adaptive Embedding transformer.

    Uses per-node adaptive embeddings concatenated with temporal input,
    then applies separate temporal and spatial transformer encoders.
    The adaptive node embeddings implicitly learn spatial relationships
    without requiring any predefined graph.

    Reference: Liu et al., "Spatio-Temporal Adaptive Embedding Makes
    Vanilla Transformer SOTA for Traffic Forecasting" (CIKM 2023).

    Input:  x [B, seq_len, N]
    Output: y [B, pred_len, N]
    """

    def __init__(
        self,
        n_nodes: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_t_layers: int = 2,
        n_s_layers: int = 1,
        d_ff: int = 128,
        adaptive_embed_dim: int = 16,
        dropout: float = 0.1,
        adj_mode: str = "learned",
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.adj_mode = adj_mode

        # Node-adaptive embeddings: [N, adaptive_embed_dim]
        self.node_embed = nn.Parameter(
            torch.randn(n_nodes, adaptive_embed_dim) * 0.02
        )

        # Input projection: 1 (value) + adaptive_embed_dim → d_model
        self.input_proj = nn.Linear(1 + adaptive_embed_dim, d_model)

        # Temporal positional encoding
        self.temporal_pos = nn.Parameter(torch.randn(1, seq_len, 1, d_model) * 0.02)

        # Temporal transformer (attention across time steps)
        t_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(t_layer, num_layers=n_t_layers)

        # Spatial transformer (attention across nodes)
        s_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.spatial_encoder = nn.TransformerEncoder(s_layer, num_layers=n_s_layers)

        # Output projection
        self.output_proj = nn.Linear(seq_len * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, N]
        B, T, N = x.shape

        # Expand node embeddings: [B, T, N, adaptive_embed_dim]
        # adj_mode="zero" or "no_both": zero out node embeddings → nodes indistinguishable
        if self.adj_mode in ("zero", "no_both"):
            ne = torch.zeros(B, T, N, self.node_embed.shape[-1], device=x.device)
        else:
            ne = self.node_embed.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        # Concatenate value + node embedding: [B, T, N, 1 + adapt_dim]
        x_cat = torch.cat([x.unsqueeze(-1), ne], dim=-1)

        # Project to d_model: [B, T, N, d_model]
        h = self.input_proj(x_cat) + self.temporal_pos

        # Temporal attention: reshape to [B*N, T, d_model]
        h_t = h.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)
        h_t = self.temporal_encoder(h_t)  # [B*N, T, d_model]
        h_t = h_t.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)

        # Spatial attention: reshape to [B*T, N, d_model]
        # adj_mode="no_spatial" or "no_both": skip spatial encoder
        if self.adj_mode in ("no_spatial", "no_both"):
            h_s = h_t
        else:
            h_s = h_t.reshape(B * T, N, self.d_model)
            h_s = self.spatial_encoder(h_s)  # [B*T, N, d_model]
            h_s = h_s.reshape(B, T, N, self.d_model)

        # Output: flatten temporal dim per node for projection
        # [B, N, T*d_model]
        h_out = h_s.permute(0, 2, 1, 3).reshape(B, N, T * self.d_model)
        out = self.output_proj(h_out)  # [B, N, pred_len]
        return out.permute(0, 2, 1)  # [B, pred_len, N]

    def get_learned_adjacency(self) -> Optional[np.ndarray]:
        """Extract implicit adjacency from node embeddings."""
        with torch.no_grad():
            ne = self.node_embed  # [N, adapt_dim]
            A = torch.softmax(ne @ ne.T, dim=-1)
            A = A - torch.diag(A.diag())  # remove self-loops
        return A.cpu().numpy()

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total}


# ─────────────────────────────────────────────────────────
# STID: Spatial-Temporal Identity — MLP with node identity
# ─────────────────────────────────────────────────────────


class STID(nn.Module):
    """STID — Spatial-Temporal Identity model.

    Demonstrates that a simple MLP with spatial and temporal identity
    embeddings can match or exceed complex STGNNs. Uses node identity
    embeddings to inject spatial awareness without any graph structure.

    The key insight: explicit graph may be unnecessary if the model
    can learn node-specific patterns through identity embeddings.

    Reference: Shao et al., "Spatial-Temporal Identity: A Simple yet
    Effective Baseline for Multivariate Time Series Forecasting"
    (CIKM 2022).

    Input:  x [B, seq_len, N]
    Output: y [B, pred_len, N]
    """

    def __init__(
        self,
        n_nodes: int,
        seq_len: int,
        pred_len: int,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Node identity embeddings
        self.node_embed = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.02)

        # Temporal identity embeddings (for each time step in the window)
        self.time_embed = nn.Parameter(torch.randn(seq_len, embed_dim) * 0.02)

        # Input projection: 1 (value) + embed_dim (node) + embed_dim (time) → hidden
        input_dim = 1 + 2 * embed_dim

        layers = []
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.mlp = nn.Sequential(*layers)

        # Output: project from hidden per-timestep to pred_len
        self.output_proj = nn.Linear(seq_len * hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, N]
        B, T, N = x.shape

        # Expand embeddings: node [B, T, N, embed_dim], time [B, T, N, embed_dim]
        ne = self.node_embed.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        te = self.time_embed.unsqueeze(0).unsqueeze(2).expand(B, -1, N, -1)

        # Concatenate: [B, T, N, 1 + 2*embed_dim]
        h = torch.cat([x.unsqueeze(-1), ne, te], dim=-1)

        # MLP per (timestep, node): [B, T, N, hidden_dim]
        h = self.mlp(h)

        # Flatten T and project: [B, N, T*hidden_dim] → [B, N, pred_len]
        h = h.permute(0, 2, 1, 3).reshape(B, N, T * h.shape[-1])
        out = self.output_proj(h)  # [B, N, pred_len]

        return out.permute(0, 2, 1)  # [B, pred_len, N]

    def get_learned_adjacency(self) -> Optional[np.ndarray]:
        """Extract implicit node similarity from identity embeddings."""
        with torch.no_grad():
            ne = self.node_embed  # [N, embed_dim]
            sim = F.cosine_similarity(
                ne.unsqueeze(0), ne.unsqueeze(1), dim=-1
            )  # [N, N]
            sim = sim - torch.diag(sim.diag())
        return sim.cpu().numpy()

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total}


# ─────────────────────────────────────────────────────────
# D2STGNN: Decoupled Dynamic Spatial-Temporal Graph Neural Network
# ─────────────────────────────────────────────────────────


class _EstimationGate(nn.Module):
    """Gate to estimate the proportion of diffusion vs inherent signals.

    Adapted from Shao et al. (VLDB 2022). For CML: uses node embeddings
    only (no time-of-day/day-of-week features).
    """

    def __init__(self, node_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(2 * node_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, node_emb_u, node_emb_d, history_data):
        # node_emb_u, node_emb_d: [N, d]
        # history_data: [B, T, N, D]
        B, T, N, _ = history_data.shape
        # Expand node embeddings to [B, T, N, d]
        eu = node_emb_u.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        ed = node_emb_d.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        gate_feat = torch.cat([eu, ed], dim=-1)
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(gate_feat))))
        return history_data * gate


class _DiffusionConv(nn.Module):
    """Localized spatial-temporal diffusion convolution.

    Multi-hop graph convolution with temporal causal conv.
    """

    def __init__(self, hidden_dim: int, k_s: int = 2, k_t: int = 3, dropout: float = 0.1):
        super().__init__()
        self.k_s = k_s
        self.k_t = k_t
        # Temporal causal conv
        self.temporal_conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=(1, k_t),
            padding=(0, k_t - 1)  # causal: pad left
        )
        # Spatial mixing: project multi-hop aggregated features
        self.spatial_fc = nn.Linear(hidden_dim * (k_s + 1), hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_list):
        # x: [B, T, N, D]
        B, T, N, D = x.shape
        # Temporal conv: [B, D, N, T]
        h = x.permute(0, 3, 2, 1)
        h = self.temporal_conv(h)[..., :T]  # causal: trim right
        h = h.permute(0, 3, 2, 1)  # back to [B, T, N, D]

        # Multi-hop spatial aggregation
        hops = [h]
        h_k = h
        for adj in adj_list:
            # adj: [N, N] or [B, N, N]
            if adj.dim() == 2:
                h_k = torch.einsum("btnd,mn->btmd", h_k, adj)
            else:
                h_k = torch.einsum("btnd,bmn->btmd", h_k, adj)
            hops.append(h_k)

        # Concatenate hops and project
        h_cat = torch.cat(hops, dim=-1)  # [B, T, N, D*(k_s+1)]
        out = self.spatial_fc(h_cat)
        out = self.dropout(F.relu(self.norm(out)))
        return out


class _InherentBlock(nn.Module):
    """Inherent signal block: GRU + self-attention (node-local dynamics).

    Captures temporal patterns that are inherent to each node,
    independent of diffusion along the graph.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, N, D]
        B, T, N, D = x.shape
        # Reshape to process each node's temporal series
        h = x.permute(0, 2, 1, 3).reshape(B * N, T, D)  # [B*N, T, D]

        # GRU
        h_gru, _ = self.gru(h)
        h = self.norm1(h + self.dropout(h_gru))

        # Self-attention over time
        h_attn, _ = self.attn(h, h, h)
        h = self.norm2(h + self.dropout(h_attn))

        # Feed-forward
        h = h + self.dropout(self.ff(h))

        return h.reshape(B, N, T, D).permute(0, 2, 1, 3)  # [B, T, N, D]


class _DecoupleLayer(nn.Module):
    """One layer of the Decoupled Spatial-Temporal Framework.

    Splits input into diffusion and inherent components via estimation gate,
    processes each with its specialized module, then recombines via
    residual decomposition.
    """

    def __init__(self, hidden_dim: int, node_dim: int, k_s: int = 2,
                 k_t: int = 3, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.est_gate = _EstimationGate(node_dim, hidden_dim=64)
        self.dif_block = _DiffusionConv(hidden_dim, k_s=k_s, k_t=k_t, dropout=dropout)
        self.inh_block = _InherentBlock(hidden_dim, n_heads=n_heads, dropout=dropout)
        self.fc_dif = nn.Linear(hidden_dim, hidden_dim)
        self.fc_inh = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, adj_list, node_emb_u, node_emb_d):
        # x: [B, T, N, D]
        gated = self.est_gate(node_emb_u, node_emb_d, x)

        # Diffusion branch
        dif_out = self.dif_block(gated, adj_list)
        dif_forecast = self.fc_dif(dif_out)

        # Residual: remove diffusion signal
        residual = x[:, -dif_out.shape[1]:, :, :] - dif_out

        # Inherent branch
        inh_out = self.inh_block(residual)
        inh_forecast = self.fc_inh(inh_out)

        # Output: residual for next layer
        next_input = residual - inh_out
        return next_input, dif_forecast, inh_forecast


class D2STGNN(nn.Module):
    """D2STGNN — Decoupled Dynamic Spatial-Temporal Graph Neural Network.

    Separates traffic (CML) signals into:
    1. Diffusion signals: propagated through graph via multi-hop GCN
    2. Inherent signals: node-local dynamics via GRU + attention

    An estimation gate learns the proportion of each component.
    A dynamic graph constructor learns time-varying adjacency.

    Adapted for CML: no time-of-day/day-of-week features; uses adaptive
    + static graph construction from node embeddings (like GWNet).

    Reference: Shao et al., "Decoupled Dynamic Spatial-Temporal Graph
    Neural Network for Traffic Forecasting" (VLDB 2022).

    Input:  x [B, seq_len, N]
    Output: y [B, pred_len, N]
    """

    def __init__(
        self,
        n_nodes: int,
        seq_len: int,
        pred_len: int,
        hidden_dim: int = 32,
        node_dim: int = 10,
        n_layers: int = 3,
        k_s: int = 2,
        k_t: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # Input embedding
        self.input_proj = nn.Linear(1, hidden_dim)

        # Node embeddings for adaptive adjacency and estimation gate
        self.node_emb_u = nn.Parameter(torch.randn(n_nodes, node_dim) * 0.1)
        self.node_emb_d = nn.Parameter(torch.randn(n_nodes, node_dim) * 0.1)

        # Dynamic graph constructor: node embeddings → attention-based adj
        self.dyn_graph_fc = nn.Linear(2 * node_dim, n_nodes)

        # Decouple layers
        self.layers = nn.ModuleList([
            _DecoupleLayer(hidden_dim, node_dim, k_s=k_s, k_t=k_t,
                           n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output: aggregate forecasts from all layers
        self.out_fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out_fc2 = nn.Linear(hidden_dim * 2, pred_len)

    def _build_graphs(self):
        """Construct static and dynamic adjacency matrices."""
        # Static graph: softmax(E_u @ E_d^T)
        static_adj = F.softmax(F.relu(self.node_emb_u @ self.node_emb_d.t()), dim=1)

        # Dynamic graph: learned from concatenated embeddings
        cat = torch.cat([self.node_emb_u, self.node_emb_d], dim=-1)  # [N, 2d]
        dyn_logits = self.dyn_graph_fc(cat)  # [N, N]
        dyn_adj = F.softmax(F.relu(dyn_logits), dim=1)

        return [static_adj, dyn_adj]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, N]
        B, T, N = x.shape

        # Input embedding: [B, T, N] → [B, T, N, D]
        h = self.input_proj(x.unsqueeze(-1))

        adj_list = self._build_graphs()

        dif_forecasts = []
        inh_forecasts = []

        for layer in self.layers:
            h, dif_fc, inh_fc = layer(h, adj_list, self.node_emb_u, self.node_emb_d)
            dif_forecasts.append(dif_fc)
            inh_forecasts.append(inh_fc)

        # Sum forecast hidden states from all layers, take mean over time
        dif_agg = sum(dif_forecasts)
        inh_agg = sum(inh_forecasts)
        forecast = dif_agg + inh_agg  # [B, T', N, D]

        # Pool over remaining time dimension
        forecast = forecast.mean(dim=1)  # [B, N, D]

        # Project to pred_len
        out = self.out_fc2(F.relu(self.out_fc1(forecast)))  # [B, N, pred_len]
        return out.permute(0, 2, 1)  # [B, pred_len, N]

    def get_learned_adjacency(self) -> Optional[np.ndarray]:
        with torch.no_grad():
            adj_list = self._build_graphs()
            A = sum(adj_list) / len(adj_list)
            A = A - torch.diag(A.diag())
            return A.cpu().numpy()

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total}


# ─────────────────────────────────────────────────────────
# DSformer: Double Sampling Transformer
# ─────────────────────────────────────────────────────────


class _RevIN(nn.Module):
    """Reversible Instance Normalization for time series."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._stdev = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self._mean) / self._stdev
            x = x * self.affine_weight + self.affine_bias
            return x
        else:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            x = x * self._stdev + self._mean
            return x


class _DSEmbed(nn.Module):
    """Double Sampling embedding: down-sampling + piecewise (interval) sampling.

    Produces two views of the input that focus on global patterns
    (down-sampling) and local patterns (interval sampling) respectively.
    """

    def __init__(self, input_len: int, num_id: int, num_samp: int, use_node: bool):
        super().__init__()
        self.num_samp = num_samp
        self.use_node = use_node
        if use_node:
            self.node_emb = nn.Parameter(torch.empty(num_id, input_len))
            nn.init.xavier_uniform_(self.node_emb)

    def _down_sample(self, data, n):
        # data: [B, N, L, 1]
        chunks = [data[:, :, i::n, :] for i in range(n)]
        result = torch.cat(chunks, dim=3)
        return result.transpose(2, 3)  # [B, N, n_samp_dim, chunk_len]

    def _interval_sample(self, data, n):
        # data: [B, N, L, 1]
        chunk_len = data.shape[2] // n
        chunks = [data[:, :, chunk_len * i:chunk_len * (i + 1), :] for i in range(n)]
        result = torch.cat(chunks, dim=3)
        return result.transpose(2, 3)  # [B, N, n_samp_dim, chunk_len]

    def forward(self, x):
        # x: [B, N, L]
        x = x.unsqueeze(-1)  # [B, N, L, 1]
        B = x.shape[0]

        if self.use_node:
            ne = self.node_emb.unsqueeze(0).expand(B, -1, -1).unsqueeze(-1)
            x1 = self._down_sample(x, self.num_samp)
            x1 = torch.cat([x1, self._down_sample(ne, self.num_samp)], dim=-1)
            x2 = self._interval_sample(x, self.num_samp)
            x2 = torch.cat([x2, self._interval_sample(ne, self.num_samp)], dim=-1)
        else:
            x1 = self._down_sample(x, self.num_samp)
            x2 = self._interval_sample(x, self.num_samp)

        return x1, x2


class _TemporalAttention(nn.Module):
    """Multi-head temporal attention."""

    def __init__(self, dim_input: int, dropout: float, num_head: int):
        super().__init__()
        self.num_head = num_head
        self.query = nn.Conv2d(dim_input, dim_input, 1)
        self.key = nn.Conv2d(dim_input, dim_input, 1)
        self.value = nn.Conv2d(dim_input, dim_input, 1)
        self.norm = nn.LayerNorm(dim_input)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head, 1)

    def forward(self, x):
        # x: [B, N, samp_dim, feat_dim]
        x_t = x.transpose(-3, -1)  # [B, feat_dim, samp_dim, N]
        heads = []
        for _ in range(self.num_head):
            q = self.dropout(self.query(x_t)).transpose(-3, -1)
            k = self.dropout(self.key(x_t)).transpose(-3, -1).transpose(-2, -1)
            v = self.dropout(self.value(x_t)).transpose(-3, -1)
            kd = (k.shape[-1] / self.num_head) ** 0.5
            attn = self.dropout(self.softmax(q @ k / kd)) @ v
            heads.append(attn.unsqueeze(-1))
        result = self.output(torch.cat(heads, dim=-1)).squeeze(-1)
        x = x + result
        return self.norm(x)


class _VariableAttention(nn.Module):
    """Multi-head variable (spatial) attention."""

    def __init__(self, num_id: int, dropout: float, num_head: int):
        super().__init__()
        self.num_head = num_head
        self.query = nn.Linear(num_id, num_id)
        self.key = nn.Linear(num_id, num_id)
        self.value = nn.Linear(num_id, num_id)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head, 1)

    def forward(self, x):
        # x: [B, N, samp_dim, feat_dim]
        x_t = x.transpose(1, 3)  # [B, feat_dim, samp_dim, N]
        q = self.dropout(self.query(x_t))
        k = self.dropout(self.key(x_t)).transpose(-2, -1)
        v = self.dropout(self.value(x_t))
        kd = (k.shape[-1] / self.num_head) ** 0.5
        heads = []
        for _ in range(self.num_head):
            attn = self.dropout(self.softmax(q @ k / kd)) @ v
            heads.append(attn.unsqueeze(-1))
        result = self.output(torch.cat(heads, dim=-1)).squeeze(-1)
        return result.transpose(1, 3)  # [B, N, samp_dim, feat_dim]


class _CrossAttention(nn.Module):
    """Cross-attention to fuse temporal and variable features."""

    def __init__(self, dim_input: int, dropout: float, num_head: int):
        super().__init__()
        self.num_head = num_head
        self.query = nn.Conv2d(dim_input, dim_input, 1)
        self.key = nn.Conv2d(dim_input, dim_input, 1)
        self.value = nn.Conv2d(dim_input, dim_input, 1)
        self.norm = nn.LayerNorm(dim_input)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head, 1)

    def forward(self, x_time, x_space):
        x_t = x_time.transpose(-3, -1)
        x_s = x_space.transpose(-3, -1)
        heads = []
        for _ in range(self.num_head):
            q = self.dropout(self.query(x_s)).transpose(-3, -1)
            k = self.dropout(self.key(x_t)).transpose(-3, -1).transpose(-2, -1)
            v = self.dropout(self.value(x_t)).transpose(-3, -1)
            kd = (k.shape[-1] / self.num_head) ** 0.5
            attn = self.dropout(self.softmax(q @ k / kd)) @ v
            heads.append(attn.unsqueeze(-1))
        result = self.output(torch.cat(heads, dim=-1)).squeeze(-1)
        x = x_time + result
        return self.norm(x)


class _TVAEncoder(nn.Module):
    """Temporal-Variable Attention encoder block."""

    def __init__(self, input_len: int, num_id: int, num_layer: int,
                 dropout: float, num_head: int, num_samp: int):
        super().__init__()
        self.num_layer = num_layer
        self.time_att = _TemporalAttention(input_len, dropout, num_head)
        self.space_att = _VariableAttention(num_id, dropout, num_head)
        self.cross_att = _CrossAttention(input_len, dropout, num_head)
        self.linear = nn.Conv2d(input_len, input_len, kernel_size=(num_samp, 1))

    def forward(self, x):
        # x: [B, N, samp_dim, feat_dim]
        for _ in range(self.num_layer):
            x = self.cross_att(self.time_att(x), self.space_att(x))
        x = self.linear(x.transpose(-3, -1)).squeeze(-2)  # [B, feat_dim, N]
        return x.transpose(-2, -1)  # [B, N, feat_dim]


class DSformer(nn.Module):
    """DSformer — Double Sampling Transformer for multivariate time series.

    Key components:
    1. Double Sampling (DS) block: down-sampling + interval sampling
       to extract global and local temporal patterns
    2. Temporal-Variable Attention (TVA) block: temporal attention,
       variable attention, and cross-attention fusion
    3. RevIN for normalization stability

    Adapted for CML: variable count = 2*N (x,p pairs).

    Reference: Yu et al., "DSformer: A Double Sampling Transformer for
    Multivariate Time Series Long-term Prediction" (CIKM 2023).

    Input:  x [B, seq_len, N]
    Output: y [B, pred_len, N]
    """

    def __init__(
        self,
        n_nodes: int,
        seq_len: int,
        pred_len: int,
        num_layer: int = 1,
        dropout: float = 0.15,
        num_head: int = 2,
        num_samp: int = 2,
        use_node_embed: bool = True,
        use_revin: bool = True,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_revin = use_revin

        if use_node_embed:
            self.input_len_eff = 2 * seq_len // num_samp
        else:
            self.input_len_eff = seq_len // num_samp

        self.revin = _RevIN(n_nodes) if use_revin else None
        self.embed = _DSEmbed(seq_len, n_nodes, num_samp, use_node_embed)
        self.encoder = _TVAEncoder(
            self.input_len_eff, n_nodes, num_layer, dropout, num_head, num_samp
        )
        self.norm = nn.LayerNorm(self.input_len_eff)
        self.output_conv = nn.Conv1d(self.input_len_eff, pred_len, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, N]
        x = self.revin(x, "norm") if self.use_revin else x
        x = x.transpose(-2, -1)           # [B, N, seq_len]

        x1, x2 = self.embed(x)            # two views: [B, N, samp_dim, feat_dim]

        x1 = self.encoder(x1)             # [B, N, feat_dim]
        x2 = self.encoder(x2)             # [B, N, feat_dim]
        h = self.norm(x1 + x2)            # [B, N, feat_dim]

        out = self.output_conv(h.transpose(-2, -1))  # [B, pred_len, N]
        out = self.revin(out, "denorm") if self.use_revin else out
        return out

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total}



def create_model(
    model_name: str,
    n_nodes: int,
    seq_len: int,
    pred_len: int,
    A_true: Optional[torch.Tensor] = None,
    **kwargs,
) -> nn.Module:
    """Create a model by name.

    Args:
        model_name: one of 'dlinear', 'tcn', 'lstm', 'nbeats', 'patchtst',
                    'stid', 'itransformer', 'dsformer', 'oracle_gcn', 'agcrn',
                    'graph_wavenet', 'd2stgnn', 'staeformer'
        n_nodes: number of nodes (2*N for standard encoding)
        seq_len: input sequence length
        pred_len: prediction horizon
        A_true: ground-truth adjacency for Oracle model
        **kwargs: model-specific arguments

    Returns:
        Initialized model
    """
    if model_name == "dlinear":
        return DLinearBaseline(seq_len=seq_len, pred_len=pred_len, n_nodes=n_nodes)
    elif model_name == "tcn":
        return TCN(
            n_nodes=n_nodes,
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_channels=kwargs.get("hidden_channels", 32),
            kernel_size=kwargs.get("kernel_size", 3),
            n_layers=kwargs.get("n_layers", 4),
            dropout=kwargs.get("dropout", 0.1),
        )
    elif model_name == "lstm":
        return LSTMBaseline(
            seq_len=seq_len,
            pred_len=pred_len,
            n_nodes=n_nodes,
            hidden_dim=kwargs.get("hidden_dim", 64),
            n_layers=kwargs.get("n_layers", 2),
        )
    elif model_name == "itransformer":
        return iTransformer(
            n_nodes=n_nodes,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=kwargs.get("d_model", 64),
            n_heads=kwargs.get("n_heads", 4),
            n_layers=kwargs.get("n_layers", 2),
            d_ff=kwargs.get("d_ff", 128),
            dropout=kwargs.get("dropout", 0.1),
        )
    elif model_name == "nbeats":
        return NBEATSBaseline(
            n_nodes=n_nodes,
            seq_len=seq_len,
            pred_len=pred_len,
            n_stacks=kwargs.get("n_stacks", 2),
            n_blocks=kwargs.get("n_blocks", 3),
            hidden_dim=kwargs.get("hidden_dim", 128),
            theta_dim=kwargs.get("theta_dim", 16),
            dropout=kwargs.get("dropout", 0.1),
        )
    elif model_name == "agcrn":
        return AGCRNModel(
            n_nodes=n_nodes,
            seq_len=seq_len,
            pred_len=pred_len,
            embed_dim=kwargs.get("embed_dim", 10),
            rnn_units=kwargs.get("rnn_units", 32),
            n_layers=kwargs.get("n_layers", 2),
            cheb_k=kwargs.get("cheb_k", 2),
            dropout=kwargs.get("dropout", 0.1),
            adj_mode=kwargs.get("adj_mode", "learned"),
        )
    elif model_name == "oracle_gcn":
        if A_true is None:
            raise ValueError("Oracle-GCN model requires ground-truth adjacency A_true")
        return OracleGCN(
            n_nodes=n_nodes,
            seq_len=seq_len,
            pred_len=pred_len,
            A_true=A_true,
            d_model=kwargs.get("d_model", 32),
            n_gcn_layers=kwargs.get("n_gcn_layers", 2),
            dropout=kwargs.get("dropout", 0.1),
            self_loop_alpha=kwargs.get("self_loop_alpha", 0.2),
        )
    elif model_name == "graph_wavenet":
        return GraphWaveNet(
            n_nodes=n_nodes,
            seq_len=seq_len,
            pred_len=pred_len,
            residual_channels=kwargs.get("residual_channels", 16),
            dilation_channels=kwargs.get("dilation_channels", 16),
            skip_channels=kwargs.get("skip_channels", 32),
            end_channels=kwargs.get("end_channels", 64),
            n_layers=kwargs.get("n_layers", 4),
            adj_embed_dim=kwargs.get("adj_embed_dim", 10),
            dropout=kwargs.get("dropout", 0.3),
        )
    elif model_name == "patchtst":
        return PatchTST(
            n_nodes=n_nodes,
            seq_len=seq_len,
            pred_len=pred_len,
            patch_len=kwargs.get("patch_len", 8),
            stride=kwargs.get("stride", 4),
            d_model=kwargs.get("d_model", 64),
            n_heads=kwargs.get("n_heads", 4),
            n_layers=kwargs.get("n_layers", 2),
            d_ff=kwargs.get("d_ff", 128),
            dropout=kwargs.get("dropout", 0.1),
        )
    elif model_name == "staeformer":
        return STAEformer(
            n_nodes=n_nodes,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=kwargs.get("d_model", 64),
            n_heads=kwargs.get("n_heads", 4),
            n_t_layers=kwargs.get("n_t_layers", 2),
            n_s_layers=kwargs.get("n_s_layers", 1),
            d_ff=kwargs.get("d_ff", 128),
            adaptive_embed_dim=kwargs.get("adaptive_embed_dim", 16),
            dropout=kwargs.get("dropout", 0.1),
            adj_mode=kwargs.get("adj_mode", "learned"),
        )
    elif model_name == "stid":
        return STID(
            n_nodes=n_nodes,
            seq_len=seq_len,
            pred_len=pred_len,
            embed_dim=kwargs.get("embed_dim", 32),
            hidden_dim=kwargs.get("hidden_dim", 128),
            n_layers=kwargs.get("n_layers", 3),
            dropout=kwargs.get("dropout", 0.1),
        )
    elif model_name == "d2stgnn":
        return D2STGNN(
            n_nodes=n_nodes,
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_dim=kwargs.get("hidden_dim", 32),
            node_dim=kwargs.get("node_dim", 10),
            n_layers=kwargs.get("n_layers", 3),
            k_s=kwargs.get("k_s", 2),
            k_t=kwargs.get("k_t", 3),
            n_heads=kwargs.get("n_heads", 4),
            dropout=kwargs.get("dropout", 0.1),
        )
    elif model_name == "dsformer":
        return DSformer(
            n_nodes=n_nodes,
            seq_len=seq_len,
            pred_len=pred_len,
            num_layer=kwargs.get("num_layer", 1),
            dropout=kwargs.get("dropout", 0.15),
            num_head=kwargs.get("num_head", 2),
            num_samp=kwargs.get("num_samp", 2),
            use_node_embed=kwargs.get("use_node_embed", True),
            use_revin=kwargs.get("use_revin", True),
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: dlinear, tcn, lstm, nbeats, patchtst, stid, itransformer, "
            f"dsformer, oracle_gcn, agcrn, graph_wavenet, d2stgnn, staeformer"
        )
