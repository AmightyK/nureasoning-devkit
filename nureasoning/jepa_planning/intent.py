"""Observation-conditioned future-intent predictor.

The predictor API intentionally contains only observation-derived tokens.  A
ground-truth future or target latent cannot be passed to ``forward`` and is used
only by the separate JEPA loss.
"""

from __future__ import annotations

import torch
from torch import nn


class IntentPredictor(nn.Module):
    """Predict ten time-aligned intent tokens from scene/state/command context."""

    def __init__(
        self,
        context_dim: int = 512,
        num_waypoints: int = 10,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if context_dim <= 0 or num_waypoints <= 0 or num_layers <= 0:
            raise ValueError(
                "context_dim, num_waypoints, and num_layers must be positive"
            )
        if num_heads <= 0 or context_dim % num_heads:
            raise ValueError("context_dim must be divisible by num_heads")
        if mlp_ratio <= 0.0:
            raise ValueError("mlp_ratio must be positive")

        self.context_dim = int(context_dim)
        self.num_waypoints = int(num_waypoints)
        self.intent_queries = nn.Parameter(
            torch.empty(1, num_waypoints, context_dim)
        )
        self.scene_type_embedding = nn.Parameter(torch.empty(1, 1, context_dim))
        self.state_type_embedding = nn.Parameter(torch.empty(1, 1, context_dim))
        self.command_type_embedding = nn.Parameter(torch.empty(1, 1, context_dim))
        self.memory_norm = nn.LayerNorm(context_dim)

        layer = nn.TransformerDecoderLayer(
            d_model=context_dim,
            nhead=num_heads,
            dim_feedforward=int(context_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_predictor = nn.TransformerDecoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(context_dim),
        )
        for parameter in (
            self.intent_queries,
            self.scene_type_embedding,
            self.state_type_embedding,
            self.command_type_embedding,
        ):
            nn.init.normal_(parameter, mean=0.0, std=0.02)

    def _validate_token_tensor(
        self,
        value: torch.Tensor,
        *,
        name: str,
        batch_size: int | None = None,
        token_count: int | None = None,
    ) -> int:
        if value.ndim != 3 or value.shape[-1] != self.context_dim:
            raise ValueError(
                f"{name} must have shape [B,T,{self.context_dim}], got "
                f"{tuple(value.shape)}"
            )
        if value.shape[1] <= 0:
            raise ValueError(f"{name} must contain at least one token")
        if token_count is not None and value.shape[1] != token_count:
            raise ValueError(f"{name} must contain exactly {token_count} token")
        if batch_size is not None and value.shape[0] != batch_size:
            raise ValueError(f"{name} batch dimension does not match scene_tokens")
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
        return int(value.shape[0])

    def forward(
        self,
        scene_tokens: torch.Tensor,
        state_token: torch.Tensor,
        command_token: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted intent ``[B, 10, D]`` from observations only."""

        batch_size = self._validate_token_tensor(scene_tokens, name="scene_tokens")
        self._validate_token_tensor(
            state_token,
            name="state_token",
            batch_size=batch_size,
            token_count=1,
        )
        self._validate_token_tensor(
            command_token,
            name="command_token",
            batch_size=batch_size,
            token_count=1,
        )
        if state_token.device != scene_tokens.device or command_token.device != scene_tokens.device:
            raise ValueError("all intent-predictor inputs must be on the same device")

        scene = scene_tokens + self.scene_type_embedding.to(dtype=scene_tokens.dtype)
        state = state_token + self.state_type_embedding.to(dtype=state_token.dtype)
        command = command_token + self.command_type_embedding.to(dtype=command_token.dtype)
        memory = self.memory_norm(torch.cat((scene, state, command), dim=1))
        queries = self.intent_queries.to(dtype=scene_tokens.dtype).expand(
            batch_size, -1, -1
        )
        return self.temporal_predictor(tgt=queries, memory=memory)
