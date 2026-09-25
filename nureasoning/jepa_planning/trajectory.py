"""Trajectory representation used by the future-intent JEPA.

The encoder consumes *already normalized* cumulative ego-frame poses.  Keeping
normalization outside the module makes the representation boundary explicit and
prevents the planner from normalizing a trajectory twice.  Stage A trains the
encoder and decoder together; later stages must load those weights, mark the
encoder as pretrained, and freeze it before using it as a JEPA target.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


DEFAULT_TRAJECTORY_SCALE = (50.0, 20.0, math.pi)


def _scale_tensor(
    trajectory: torch.Tensor,
    scale: Sequence[float],
) -> torch.Tensor:
    if trajectory.shape[-1] != 3:
        raise ValueError(
            "trajectory must end in (x, y, heading), got "
            f"shape {tuple(trajectory.shape)}"
        )
    if len(scale) != 3 or any(float(value) <= 0.0 for value in scale):
        raise ValueError("scale must contain three positive values")
    return trajectory.new_tensor(tuple(float(value) for value in scale))


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to ``[-pi, pi]`` with a differentiable periodic mapping."""

    return torch.atan2(torch.sin(angle), torch.cos(angle))


def normalize_trajectory(
    trajectory: torch.Tensor,
    scale: Sequence[float] = DEFAULT_TRAJECTORY_SCALE,
) -> torch.Tensor:
    """Normalize raw ego-frame poses once using the nuVLA scale convention."""

    return trajectory / _scale_tensor(trajectory, scale)


def denormalize_trajectory(
    trajectory: torch.Tensor,
    scale: Sequence[float] = DEFAULT_TRAJECTORY_SCALE,
) -> torch.Tensor:
    """Convert normalized ego-frame poses back to physical units.

    This function is the exact inverse of :func:`normalize_trajectory`; it does
    not wrap heading as that would make the transform non-invertible.  Inference
    code should explicitly call :func:`wrap_angle` on final headings.
    """

    return trajectory * _scale_tensor(trajectory, scale)


def _validate_trajectory(
    trajectory: torch.Tensor,
    *,
    name: str,
    num_waypoints: int,
) -> None:
    if trajectory.ndim != 3 or trajectory.shape[1:] != (num_waypoints, 3):
        raise ValueError(
            f"{name} must have shape [B,{num_waypoints},3], got "
            f"{tuple(trajectory.shape)}"
        )
    if not trajectory.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(trajectory).all():
        raise ValueError(f"{name} contains non-finite values")


def _validate_transformer_dimensions(hidden_dim: int, num_heads: int) -> None:
    if hidden_dim <= 0 or num_heads <= 0:
        raise ValueError("hidden_dim and num_heads must be positive")
    if hidden_dim % num_heads:
        raise ValueError("hidden_dim must be divisible by num_heads")


class TargetTrajectoryEncoder(nn.Module):
    """Encode normalized ten-pose trajectories as time-aligned intent tokens.

    Newly initialized weights are deliberately marked as not pretrained.
    ``freeze()`` refuses to freeze such weights by default so a random encoder
    cannot quietly become the fixed JEPA target.
    """

    def __init__(
        self,
        num_waypoints: int = 10,
        trajectory_dim: int = 3,
        latent_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_waypoints <= 0 or trajectory_dim != 3 or latent_dim <= 0:
            raise ValueError(
                "num_waypoints and latent_dim must be positive and "
                "trajectory_dim must be 3"
            )
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        _validate_transformer_dimensions(hidden_dim, num_heads)

        self.num_waypoints = int(num_waypoints)
        self.trajectory_dim = int(trajectory_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)

        self.input_projection = nn.Linear(trajectory_dim, hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.empty(1, num_waypoints, hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.latent_projection = nn.Linear(hidden_dim, latent_dim)
        self.register_buffer(
            "_pretrained_ready",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self._frozen = False
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    @property
    def is_pretrained(self) -> bool:
        """Whether Stage A/checkpoint loading explicitly marked these weights."""

        return bool(self._pretrained_ready.item())

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def mark_pretrained(self) -> "TargetTrajectoryEncoder":
        """Mark a successfully trained or compatibly loaded representation."""

        self._pretrained_ready.fill_(True)
        return self

    def require_pretrained(self) -> None:
        if not self.is_pretrained:
            raise RuntimeError(
                "trajectory target encoder is not pretrained; complete Stage A "
                "or load a compatible trajectory-autoencoder checkpoint first"
            )

    def freeze(self) -> "TargetTrajectoryEncoder":
        """Freeze a pretrained target and keep it in evaluation mode."""

        self.require_pretrained()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self._frozen = True
        super().train(False)
        return self

    def unfreeze_for_pretraining(self) -> "TargetTrajectoryEncoder":
        """Make the encoder trainable for Stage A autoencoder training."""

        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self._frozen = False
        super().train(True)
        return self

    def train(self, mode: bool = True) -> "TargetTrajectoryEncoder":
        # A parent ``model.train()`` must not reactivate dropout in a frozen
        # target encoder.
        return super().train(False if self._frozen else mode)

    def forward(self, normalized_future: torch.Tensor) -> torch.Tensor:
        _validate_trajectory(
            normalized_future,
            name="normalized_future",
            num_waypoints=self.num_waypoints,
        )
        hidden = self.input_projection(normalized_future)
        hidden = hidden + self.position_embedding.to(dtype=hidden.dtype)
        hidden = self.temporal_encoder(hidden)
        return self.latent_projection(hidden)


class TrajectoryDecoder(nn.Module):
    """Decode time-aligned intent tokens to normalized cumulative poses."""

    def __init__(
        self,
        num_waypoints: int = 10,
        trajectory_dim: int = 3,
        latent_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_waypoints <= 0 or trajectory_dim != 3 or latent_dim <= 0:
            raise ValueError(
                "num_waypoints and latent_dim must be positive and "
                "trajectory_dim must be 3"
            )
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        _validate_transformer_dimensions(hidden_dim, num_heads)

        self.num_waypoints = int(num_waypoints)
        self.latent_dim = int(latent_dim)
        self.latent_projection = nn.Linear(latent_dim, hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.empty(1, num_waypoints, hidden_dim)
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.output_projection = nn.Linear(hidden_dim, trajectory_dim)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        expected = (self.num_waypoints, self.latent_dim)
        if latent.ndim != 3 or latent.shape[1:] != expected:
            raise ValueError(
                f"latent must have shape [B,{expected[0]},{expected[1]}], got "
                f"{tuple(latent.shape)}"
            )
        if not latent.is_floating_point():
            raise TypeError("latent must be floating point")
        if not torch.isfinite(latent).all():
            raise ValueError("latent contains non-finite values")
        hidden = self.latent_projection(latent)
        hidden = hidden + self.position_embedding.to(dtype=hidden.dtype)
        hidden = self.temporal_decoder(hidden)
        return self.output_projection(hidden)


class TrajectoryAutoencoder(nn.Module):
    """Stage-A composition of the trajectory target encoder and decoder."""

    def __init__(
        self,
        encoder: TargetTrajectoryEncoder | None = None,
        decoder: TrajectoryDecoder | None = None,
        **architecture: object,
    ) -> None:
        super().__init__()
        if encoder is None:
            encoder = TargetTrajectoryEncoder(**architecture)
        if decoder is None:
            decoder = TrajectoryDecoder(**architecture)
        if encoder.num_waypoints != decoder.num_waypoints:
            raise ValueError("encoder and decoder waypoint counts must match")
        if encoder.latent_dim != decoder.latent_dim:
            raise ValueError("encoder and decoder latent dimensions must match")
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, normalized_future: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(normalized_future))
