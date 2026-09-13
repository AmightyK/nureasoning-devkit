"""
GR00T-style Flow-Matching DiT Action Expert for VLA trajectory prediction.

Architecture adapted from NVIDIA GR00T (Isaac-GR00T):
  - ActionEncoder: fuses noisy trajectory with flow-matching timestep
    via sinusoidal positional encoding (action_proj || time_sinusoidal → MLP)
  - StateEncoder: projects ego state + history trajectory into state tokens
  - DiT: Transformer with interleaved cross-attention (to VLM features)
    and self-attention blocks, using AdaLN timestep conditioning
  - ActionDecoder: MLP projecting DiT action-token outputs to trajectory space

The DiT processes [state_tokens ; action_tokens] while cross-attending
to detached VLM backbone features.

Reference:
  https://github.com/NVIDIA/Isaac-GR00T/tree/main/gr00t/model

Flow matching trains a velocity field v_θ(x_t, t, c) with optimal-transport:
    x_t = (1 − t) · noise + t · actions
    loss = ‖ v_θ(x_t, t, c) − (actions − noise) ‖²
where t ~ Beta(α, β) scaled by noise_s.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

logger = logging.getLogger(__name__)


@dataclass
class ActionExpertConfig:
    vlm_feature_dim: int = 2048
    ego_state_dim: int = 4                # vx, vy, ax, ay
    max_history_traj_points: int = 6      # past waypoints on the 0.5 s grid (3 s)
    history_traj_dim: int = 3             # (x, y, θ) per history point
    num_waypoints: int = 10               # 5 s at 0.5 s
    trajectory_dim: int = 3               # (x, y, θ) per waypoint in ego frame

    # DiT architecture (GR00T-style)
    hidden_dim: int = 512
    num_heads: int = 8
    num_dit_layers: int = 12
    dropout: float = 0.1
    mlp_ratio: float = 4.0
    interleave_self_attention: bool = True

    # Flow matching (GR00T schedule)
    num_inference_steps: int = 10
    num_timestep_buckets: int = 1000
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 2.5
    noise_s: float = 0.999
    sigma_min: float = 1e-4
    trajectory_norm_scale: Tuple[float, float, float] = (50.0, 20.0, math.pi)

    add_pos_embed: bool = True
    max_seq_len: int = 64


# ============================================================================
# Building blocks
# ============================================================================


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding for a (B, T) grid of scalar values."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half = self.dim // 2
        exponent = -torch.arange(half, dtype=torch.float, device=timesteps.device) * (
            math.log(10000.0) / half
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # [B, T, half]
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class TimestepEncoder(nn.Module):
    """Discrete timestep → continuous embedding (sinusoidal + MLP)."""

    def __init__(self, hidden_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        exponent = -math.log(10000.0) / (half - 1)
        freqs = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float) * exponent
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(next(self.mlp.parameters()).dtype))


class FixedSinusoidalPositionEmbedding(nn.Module):
    """Pre-computed sinusoidal position embeddings added inside transformer blocks."""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]].to(x.dtype)


class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm conditioned on timestep embedding (GR00T pattern:
    SiLU → Linear → scale/shift)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(temb))           # [B, 2D]
        scale, shift = emb.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FeedForward(nn.Module):
    """Two-layer FFN with approximate GELU (matches GR00T's gelu-approximate)."""

    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single GR00T-style transformer block.

    Each block performs *either* cross-attention (to VLM encoder features)
    or self-attention — controlled by ``cross_attention_dim``.
    Timestep conditioning is applied via AdaLN before attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cross_attention_dim: Optional[int] = None,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.is_cross_attention = cross_attention_dim is not None

        self.norm1 = AdaLayerNorm(dim)
        self.pos_embed = FixedSinusoidalPositionEmbedding(dim, max_len=max_seq_len)

        if self.is_cross_attention:
            self.attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                kdim=cross_attention_dim,
                vdim=cross_attention_dim,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=mlp_ratio, dropout=dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed = self.pos_embed(self.norm1(hidden_states, temb))

        if self.is_cross_attention and encoder_hidden_states is not None:
            attn_out = self.attn(
                normed, encoder_hidden_states, encoder_hidden_states,
                need_weights=False,
            )[0]
        else:
            attn_out = self.attn(normed, normed, normed, need_weights=False)[0]
        hidden_states = hidden_states + attn_out

        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


# ============================================================================
# Encoders
# ============================================================================


class ActionEncoder(nn.Module):
    """GR00T-style action encoder: fuses noisy actions with timestep
    through concatenated sinusoidal encoding and a two-layer MLP with swish."""

    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()
        self.W1 = nn.Linear(action_dim, hidden_dim)
        self.W2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions:   [B, T, action_dim] noisy trajectory
            timesteps: [B] discretised timestep indices
        Returns:
            [B, T, hidden_dim] action features with fused time information
        """
        B, T, _ = actions.shape
        t_expanded = timesteps.unsqueeze(1).expand(-1, T)     # [B, T]

        a_emb = self.W1(actions)                               # [B, T, D]
        tau_emb = self.pos_encoding(t_expanded).to(a_emb.dtype)

        x = torch.cat([a_emb, tau_emb], dim=-1)               # [B, T, 2D]
        x = self.W2(x)
        x = x * torch.sigmoid(x)                              # swish
        return self.W3(x)                                      # [B, T, D]


class StateEncoder(nn.Module):
    """Encode ego state + flattened history trajectory into a single state token."""

    def __init__(
        self,
        ego_state_dim: int,
        max_history_points: int,
        history_point_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        state_dim = ego_state_dim + max_history_points * history_point_dim
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, ego_state: torch.Tensor, history_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ego_state:          [B, ego_state_dim]
            history_trajectory: [B, max_points, 3]
        Returns:
            [B, 1, hidden_dim] single state token
        """
        hist_flat = history_trajectory.reshape(ego_state.shape[0], -1)
        return self.mlp(torch.cat([ego_state, hist_flat], dim=-1)).unsqueeze(1)


# ============================================================================
# DiT (GR00T-style)
# ============================================================================


class GR00TDiT(nn.Module):
    """Diffusion Transformer following the GR00T architecture.

    Interleaves cross-attention blocks (attending to VLM backbone features)
    and self-attention blocks.  All blocks are conditioned on the flow-matching
    timestep through AdaLN.  The output block applies an additional AdaLN-style
    scale/shift before the final projection.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        vlm_feature_dim: int,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        interleave_self_attention: bool = True,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.interleave_self_attention = interleave_self_attention

        self.timestep_encoder = TimestepEncoder(hidden_dim)
        self.vlm_layer_norm = nn.LayerNorm(vlm_feature_dim)

        blocks = []
        for idx in range(num_layers):
            is_self = (idx % 2 == 1) and interleave_self_attention
            blocks.append(
                TransformerBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    cross_attention_dim=None if is_self else vlm_feature_dim,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    max_seq_len=max_seq_len,
                )
            )
        self.transformer_blocks = nn.ModuleList(blocks)

        self.norm_out = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.proj_out_2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:         [B, S, hidden_dim]  (state + action tokens)
            encoder_hidden_states: [B, L, vlm_feature_dim]  VLM features
            timestep:              [B]  discretised timestep indices
        Returns:
            [B, S, hidden_dim]
        """
        temb = self.timestep_encoder(timestep)
        encoder_hidden_states = self.vlm_layer_norm(encoder_hidden_states)

        for idx, block in enumerate(self.transformer_blocks):
            is_self = (idx % 2 == 1) and self.interleave_self_attention
            if is_self:
                hidden_states = block(hidden_states, temb)
            else:
                hidden_states = block(hidden_states, temb, encoder_hidden_states)

        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        )
        return self.proj_out_2(hidden_states)


# ============================================================================
# Action Expert
# ============================================================================


class FlowMatchingDiTActionExpert(nn.Module):
    """GR00T-style flow-matching DiT action expert for ego-frame trajectory
    prediction.

    Pipeline (training):
      1. StateEncoder  →  state token   [B, 1, D]
      2. ActionEncoder →  action tokens  [B, T, D]  (noisy traj + time fused)
      3. Concat [state ; action] → DiT → cross-attend to VLM features
      4. ActionDecoder on the action-token portion of DiT output

    Pipeline (inference):
      Euler integration from pure noise over ``num_inference_steps``.
    """

    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.config = config

        self.state_encoder = StateEncoder(
            ego_state_dim=config.ego_state_dim,
            max_history_points=config.max_history_traj_points,
            history_point_dim=config.history_traj_dim,
            hidden_dim=config.hidden_dim,
        )
        self.action_encoder = ActionEncoder(
            action_dim=config.trajectory_dim,
            hidden_dim=config.hidden_dim,
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.dit = GR00TDiT(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_dit_layers,
            vlm_feature_dim=config.vlm_feature_dim,
            dropout=config.dropout,
            mlp_ratio=config.mlp_ratio,
            interleave_self_attention=config.interleave_self_attention,
            max_seq_len=config.max_seq_len,
        )

        self.action_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.trajectory_dim),
        )

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.action_decoder[-1].weight)
        nn.init.zeros_(self.action_decoder[-1].bias)

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _norm_scale_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(self.config.trajectory_norm_scale, device=device, dtype=dtype)

    def normalize_trajectory(self, waypoints: torch.Tensor) -> torch.Tensor:
        return waypoints / self._norm_scale_tensor(waypoints.device, waypoints.dtype)

    def denormalize_trajectory(self, waypoints: torch.Tensor) -> torch.Tensor:
        return waypoints * self._norm_scale_tensor(waypoints.device, waypoints.dtype)

    # ------------------------------------------------------------------
    # Time sampling
    # ------------------------------------------------------------------

    def _sample_time(
        self, batch_size: int, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (1 - sample) * self.config.noise_s

    # ------------------------------------------------------------------
    # Core DiT forward
    # ------------------------------------------------------------------

    def _forward_dit(
        self,
        noisy_trajectory: torch.Tensor,
        timestep_discrete: torch.Tensor,
        vlm_features: torch.Tensor,
        ego_state: torch.Tensor,
        history_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """Run the full encoder → DiT → decoder pipeline and return the
        predicted velocity for the action tokens only."""
        state_tokens = self.state_encoder(ego_state, history_trajectory)
        action_tokens = self.action_encoder(noisy_trajectory, timestep_discrete)

        if self.config.add_pos_embed:
            T = action_tokens.shape[1]
            pos_ids = torch.arange(T, dtype=torch.long, device=action_tokens.device)
            action_tokens = action_tokens + self.position_embedding(pos_ids).unsqueeze(0)

        sa_tokens = torch.cat([state_tokens, action_tokens], dim=1)
        dit_out = self.dit(sa_tokens, vlm_features, timestep_discrete)

        action_out = dit_out[:, -noisy_trajectory.shape[1] :]
        return self.action_decoder(action_out)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        x_1: torch.Tensor,
        vlm_features: torch.Tensor,
        ego_state: torch.Tensor,
        history_trajectory: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        x_1 = self.normalize_trajectory(x_1)
        B = x_1.shape[0]
        device = x_1.device

        t = self._sample_time(B, device, x_1.dtype)
        t_expand = t[:, None, None]

        noise = torch.randn_like(x_1)
        x_t = (1 - t_expand) * noise + t_expand * x_1
        velocity_target = x_1 - noise

        t_discrete = (t * self.config.num_timestep_buckets).long()
        v_pred = self._forward_dit(x_t, t_discrete, vlm_features, ego_state, history_trajectory)

        loss = F.mse_loss(v_pred, velocity_target)

        with torch.no_grad():
            clean_pred = x_t + (1.0 - t_expand) * v_pred
            clean_pred = self.denormalize_trajectory(clean_pred.float())
            clean_target = self.denormalize_trajectory(x_1.float())

            x_err = clean_pred[..., 0] - clean_target[..., 0]
            y_err = clean_pred[..., 1] - clean_target[..., 1]
            theta_err = clean_pred[..., 2] - clean_target[..., 2]
            theta_err = torch.atan2(torch.sin(theta_err), torch.cos(theta_err))

            per_dim_mse = torch.stack([
                (x_err ** 2).mean(),
                (y_err ** 2).mean(),
                (theta_err ** 2).mean(),
            ])

        return {
            "loss": loss,
            "mse_x": per_dim_mse[0],
            "mse_y": per_dim_mse[1],
            "mse_theta": per_dim_mse[2],
        }

    def forward(
        self,
        x_1: torch.Tensor,
        vlm_features: torch.Tensor,
        ego_state: torch.Tensor,
        history_trajectory: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self._compute_loss(
            x_1=x_1,
            vlm_features=vlm_features,
            ego_state=ego_state,
            history_trajectory=history_trajectory,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        vlm_features: torch.Tensor,
        ego_state: torch.Tensor,
        history_trajectory: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate trajectory via Euler integration of the learned velocity field."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        num_steps = max(int(num_steps), 1)

        B = vlm_features.shape[0]
        device = vlm_features.device
        dtype = vlm_features.dtype

        actions = torch.randn(
            B, self.config.num_waypoints, self.config.trajectory_dim,
            device=device, dtype=dtype,
        )

        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_discrete = int(t_cont * self.config.num_timestep_buckets)
            t_tensor = torch.full((B,), t_discrete, device=device, dtype=torch.long)

            v_pred = self._forward_dit(
                actions, t_tensor, vlm_features, ego_state, history_trajectory,
            )
            actions = actions + dt * v_pred

        actions = self.denormalize_trajectory(actions)
        actions[..., 2] = torch.atan2(torch.sin(actions[..., 2]), torch.cos(actions[..., 2]))
        return actions


# ============================================================================
# Trajectory metrics
# ============================================================================


def compute_trajectory_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Compute trajectory evaluation metrics in raw ego-frame coordinates.

    Both inputs are **raw cumulative waypoints** (positions relative to ego).

    Args:
        pred:   [B, T, 3] raw predicted waypoints
        target: [B, T, 3] raw ground truth waypoints

    Returns:
        dict with ADE, FDE, heading error
    """
    pos_error = torch.sqrt(
        (pred[..., 0] - target[..., 0]) ** 2
        + (pred[..., 1] - target[..., 1]) ** 2
    )

    ade = pos_error.mean().item()
    fde = pos_error[:, -1].mean().item()

    heading_error = torch.abs(pred[..., 2] - target[..., 2])
    heading_error = torch.min(heading_error, 2 * math.pi - heading_error)
    mean_heading_error = heading_error.mean().item()

    return {
        "ADE_m": ade,
        "FDE_m": fde,
        "heading_error_rad": mean_heading_error,
        "heading_error_deg": math.degrees(mean_heading_error),
    }
