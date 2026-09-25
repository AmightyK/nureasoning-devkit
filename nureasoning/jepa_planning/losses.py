"""Losses and diagnostics for trajectory pretraining and intent JEPA.

The contrastive objective uses only negatives in the current process-local
minibatch.  Gradient accumulation does not enlarge this set and cross-rank
gathering is intentionally deferred.  With a one-sample minibatch InfoNCE is
reported as inactive and contributes an exact differentiable zero.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from .trajectory import (
    DEFAULT_TRAJECTORY_SCALE,
    denormalize_trajectory,
    wrap_angle,
)


def _validate_pair(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"{name} tensors must have identical shapes, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.ndim != 3:
        raise ValueError(f"{name} tensors must have shape [B,T,D]")
    if prediction.shape[0] <= 0 or prediction.shape[1] <= 0:
        raise ValueError(f"{name} tensors may not have empty batch/token axes")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError(f"{name} tensors must be floating point")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError(f"{name} tensors contain non-finite values")


def trajectory_reconstruction_loss(
    reconstructed_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    *,
    scale: Sequence[float] = DEFAULT_TRAJECTORY_SCALE,
    dt_s: float = 0.5,
    position_weight: float = 1.0,
    heading_weight: float = 1.0,
    motion_weight: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """Reconstruction losses for normalized cumulative trajectories.

    Position, heading, and motion errors are computed after denormalizing to
    metres/radians.  Motion uses finite differences at the actual ``dt_s``;
    angular increments and heading reconstruction are wrap-aware.
    """

    _validate_pair(
        reconstructed_normalized,
        target_normalized,
        name="trajectory reconstruction",
    )
    if reconstructed_normalized.shape[-1] != 3:
        raise ValueError("trajectory reconstruction tensors must end in 3 values")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if min(position_weight, heading_weight, motion_weight) < 0.0:
        raise ValueError("reconstruction loss weights must be non-negative")

    reconstructed = denormalize_trajectory(reconstructed_normalized, scale)
    target = denormalize_trajectory(target_normalized, scale)

    position_error = reconstructed[..., :2] - target[..., :2]
    heading_error = wrap_angle(reconstructed[..., 2] - target[..., 2])
    position_loss = position_error.square().mean()
    heading_loss = heading_error.square().mean()

    if reconstructed.shape[1] > 1:
        position_velocity_error = (
            torch.diff(reconstructed[..., :2], dim=1)
            - torch.diff(target[..., :2], dim=1)
        ) / dt_s
        reconstructed_heading_delta = wrap_angle(
            torch.diff(reconstructed[..., 2], dim=1)
        )
        target_heading_delta = wrap_angle(torch.diff(target[..., 2], dim=1))
        angular_velocity_error = wrap_angle(
            reconstructed_heading_delta - target_heading_delta
        ) / dt_s
        motion_loss = torch.cat(
            (position_velocity_error, angular_velocity_error.unsqueeze(-1)),
            dim=-1,
        ).square().mean()
    else:
        motion_loss = reconstructed.sum() * 0.0

    total = (
        float(position_weight) * position_loss
        + float(heading_weight) * heading_loss
        + float(motion_weight) * motion_loss
    )
    return {
        "loss": total,
        "reconstruction_loss": total,
        "position_loss": position_loss,
        "heading_loss": heading_loss,
        "motion_loss": motion_loss,
        "position_rmse_m": position_error.square().mean().sqrt(),
        "heading_mae_rad": heading_error.abs().mean(),
    }


def _local_batch_info_nce(
    predicted: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric InfoNCE with each full temporal sequence as one example."""

    batch_size = predicted.shape[0]
    if batch_size == 1:
        return predicted.sum() * 0.0
    predicted_tokens = F.normalize(predicted, dim=-1)
    target_tokens = F.normalize(target, dim=-1)
    # Time-aligned token similarities, averaged into one logit per trajectory
    # pair. Off-diagonal entries are process-local minibatch negatives.
    logits = torch.einsum("btd,ctd->bc", predicted_tokens, target_tokens)
    logits = logits / (predicted.shape[1] * temperature)
    labels = torch.arange(batch_size, device=predicted.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )


def jepa_loss(
    predicted_intent: torch.Tensor,
    target_intent: torch.Tensor,
    *,
    feature_weight: float = 1.0,
    token_cosine_weight: float = 1.0,
    info_nce_weight: float = 0.1,
    temperature: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Compute JEPA alignment losses and collapse/alignment diagnostics.

    ``target_intent`` is always detached here as a defense-in-depth guarantee;
    the planner must additionally keep its pretrained target encoder frozen.
    """

    _validate_pair(predicted_intent, target_intent, name="JEPA latent")
    if min(feature_weight, token_cosine_weight, info_nce_weight) < 0.0:
        raise ValueError("JEPA loss weights must be non-negative")
    if temperature <= 0.0:
        raise ValueError("InfoNCE temperature must be positive")

    target = target_intent.detach()
    feature_loss = F.mse_loss(predicted_intent, target)
    token_alignment = F.cosine_similarity(predicted_intent, target, dim=-1)
    token_cosine_loss = (1.0 - token_alignment).mean()
    info_nce_loss = _local_batch_info_nce(
        predicted_intent,
        target,
        temperature,
    )
    total = (
        float(feature_weight) * feature_loss
        + float(token_cosine_weight) * token_cosine_loss
        + float(info_nce_weight) * info_nce_loss
    )

    batch_size = predicted_intent.shape[0]
    predicted_flat = predicted_intent.reshape(-1, predicted_intent.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])
    return {
        "loss": total,
        "jepa_loss": total,
        "feature_loss": feature_loss,
        "token_cosine_loss": token_cosine_loss,
        "info_nce_loss": info_nce_loss,
        "latent_alignment": token_alignment.mean().detach(),
        "predicted_latent_variance": predicted_flat.var(
            dim=0, unbiased=False
        ).mean().detach(),
        "target_latent_variance": target_flat.var(
            dim=0, unbiased=False
        ).mean().detach(),
        "info_nce_active": predicted_intent.new_tensor(float(batch_size > 1)),
        "num_local_negatives": predicted_intent.new_tensor(float(batch_size - 1)),
    }
