"""Frozen LeVJEPA feature extraction and fixed-budget scene adaptation.

The real checkpoint loader is intentionally conservative: it accepts only an
audited local snapshot with recorded model and code revisions, and is imported
only when a real backbone is requested.  Unit tests and CPU smoke runs use the
built-in deterministic stub and therefore need neither transformers nor model
weights.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from nureasoning.common.pretrained import from_pretrained
from nureasoning.jepa_planning.config import BackboneConfig


@dataclass(frozen=True)
class BackboneIdentity:
    """Serializable identity recorded alongside planning checkpoints."""

    model_id: str
    revision: str | None
    code_revision: str | None
    resolved_path: str
    model_class: str
    code_identity: str
    is_stub: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PatchLayout:
    """Audited frame-major layout of the returned patch sequence."""

    num_frames: int
    patches_per_frame: int
    patch_grid_height: int
    patch_grid_width: int

    @property
    def num_patch_tokens(self) -> int:
        return self.num_frames * self.patches_per_frame


class _StubLeVJEPAModel(nn.Module):
    """Small deterministic local model with the candidate checkpoint API."""

    def __init__(self, feature_dim: int, output_tokens: int) -> None:
        super().__init__()
        if output_tokens < 1:
            raise ValueError("stub_patch_tokens must be positive")
        self.projection = nn.Linear(1, feature_dim, bias=True)
        self.register_buffer(
            "token_offsets",
            torch.linspace(-1.0, 1.0, output_tokens).reshape(1, output_tokens, 1),
            persistent=False,
        )
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.linspace(0.25, 1.25, feature_dim).reshape(feature_dim, 1)
            )
            self.projection.bias.zero_()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        value = pixel_values.mean(dim=(1, 2, 3, 4), keepdim=False).reshape(-1, 1, 1)
        inputs = value + self.token_offsets.to(dtype=value.dtype, device=value.device)
        return self.projection(inputs)


def _model_code_identity(model: nn.Module) -> str:
    model_type = type(model)
    name = f"{model_type.__module__}.{model_type.__qualname__}"
    try:
        source_path = inspect.getsourcefile(model_type)
        if source_path is None:
            return f"{name}:source-unavailable"
        digest = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
        return f"{name}:sha256:{digest}"
    except (OSError, TypeError):
        return f"{name}:source-unavailable"


def _advertised_prefix_tokens(model: nn.Module) -> int | None:
    """Read common immutable model metadata without guessing from token values."""

    config = getattr(model, "config", None)
    if config is None:
        return None
    for name in ("num_prefix_tokens", "num_cls_tokens"):
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    for name in ("has_cls_token", "add_cls_token", "use_cls_token"):
        value = getattr(config, name, None)
        if isinstance(value, bool):
            return int(value)
    return None


def _load_real_model(config: BackboneConfig) -> tuple[nn.Module, str]:
    """Load the explicitly audited custom architecture from a local snapshot."""

    if not config.local_files_only:
        raise ValueError(
            "LeVJEPA loading must set local_files_only=True; automatic weight downloads "
            "are prohibited"
        )
    if config.trust_remote_code:
        raise ValueError(
            "set allow_audited_local_code instead of trust_remote_code; arbitrary "
            "repository code is prohibited"
        )
    if not config.allow_audited_local_code:
        raise ValueError(
            "real LeVJEPA construction is disabled until allow_audited_local_code=True; "
            "inject an already constructed local model or use the stub"
        )
    if not config.local_path or not config.revision or not config.code_revision:
        raise ValueError(
            "real LeVJEPA loading requires local_path plus immutable model and code "
            "revisions"
        )

    local_path = Path(config.local_path).expanduser().resolve()
    if not local_path.is_dir():
        raise FileNotFoundError(f"LeVJEPA local_path is not a directory: {local_path}")
    resolved = str(local_path)

    try:
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover - environment-dependent error path
        raise ImportError(
            "loading a real LeVJEPA checkpoint requires transformers; use the local "
            "stub for unit tests"
        ) from exc

    kwargs: dict[str, Any] = {
        "local_files_only": True,
        # This flag executes only code present in the explicitly selected local
        # snapshot.  The two recorded revisions are mandatory above so an M3
        # run cannot silently switch either weights or implementation.
        "trust_remote_code": True,
    }
    if config.revision is not None:
        kwargs["revision"] = config.revision
    try:
        model = from_pretrained(AutoModel.from_pretrained, resolved, **kwargs)
    except Exception as exc:  # pragma: no cover - requires a real audited checkpoint
        raise RuntimeError(
            "failed to load the audited LeVJEPA checkpoint using an installed model "
            "class. The loader never falls back to a Hub ID or network download"
        ) from exc
    if not isinstance(model, nn.Module):
        raise TypeError("LeVJEPA loader returned a non-module object")
    return model, resolved


def _extract_token_tensor(output: Any) -> torch.Tensor:
    """Extract a token sequence while keeping accepted output forms explicit."""

    if torch.is_tensor(output):
        return output
    if isinstance(output, Mapping):
        for key in ("last_hidden_state", "patch_tokens"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
    for name in ("last_hidden_state", "patch_tokens"):
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            return value
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        if output and torch.is_tensor(output[0]):
            return output[0]
    raise TypeError(
        "LeVJEPA must return a token tensor or expose last_hidden_state/patch_tokens"
    )


def _invoke_model(model: nn.Module, video: torch.Tensor) -> Any:
    """Use the standard HF name when advertised, otherwise a positional input."""

    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "pixel_values" in parameters:
        return model(pixel_values=video)
    return model(video)


def _validate_layout(num_tokens: int, num_frames: int) -> PatchLayout:
    if num_tokens <= 0:
        raise ValueError("LeVJEPA returned no patch tokens after CLS removal")
    if num_frames <= 0 or num_tokens % num_frames:
        raise ValueError(
            f"patch-token count {num_tokens} is incompatible with {num_frames} frames; "
            "expected a frame-major layout with equal spatial tokens per frame"
        )
    patches_per_frame = num_tokens // num_frames
    grid_size = math.isqrt(patches_per_frame)
    if grid_size * grid_size != patches_per_frame:
        raise ValueError(
            f"{patches_per_frame} patches per frame do not form a square spatial grid; "
            "checkpoint layout must be audited before use"
        )
    return PatchLayout(
        num_frames=num_frames,
        patches_per_frame=patches_per_frame,
        patch_grid_height=grid_size,
        patch_grid_width=grid_size,
    )


class LeVJEPABackbone(nn.Module):
    """Shared, frozen LeVJEPA encoder for chronological camera clips.

    Cameras are flattened into the batch dimension and processed with the same
    model weights.  ``camera_chunk_size`` bounds the number of camera clips in a
    single backbone call; it does not change output ordering.
    """

    def __init__(self, config: BackboneConfig, model: nn.Module | None = None) -> None:
        super().__init__()
        if not config.frozen:
            raise ValueError("the M0-M2 LeVJEPA backbone must remain frozen")
        if config.camera_chunk_size < 1:
            raise ValueError("camera_chunk_size must be at least one")
        if config.expected_feature_dim < 1:
            raise ValueError("expected_feature_dim must be positive")
        if config.trust_remote_code:
            raise ValueError("trust_remote_code must remain false for LeVJEPA loading")

        self.config = config
        if model is not None:
            resolved = "injected-local-model"
            self.model = model
            is_stub = isinstance(model, _StubLeVJEPAModel)
        elif config.use_stub:
            resolved = "builtin://levjepa-stub-v1"
            self.model = _StubLeVJEPAModel(
                config.expected_feature_dim,
                config.stub_patch_tokens,
            )
            is_stub = True
        else:
            self.model, resolved = _load_real_model(config)
            is_stub = False

        expected_prefix = int(config.expected_has_cls_token)
        advertised_prefix = _advertised_prefix_tokens(self.model)
        if advertised_prefix is not None and advertised_prefix != expected_prefix:
            raise ValueError(
                "checkpoint prefix-token metadata disagrees with "
                f"expected_has_cls_token={config.expected_has_cls_token}: "
                f"advertised {advertised_prefix} prefix tokens"
            )

        self.model.requires_grad_(False)
        self.model.eval()
        model_name = f"{type(self.model).__module__}.{type(self.model).__qualname__}"
        self.identity = BackboneIdentity(
            model_id=config.model_id,
            revision=config.revision,
            code_revision=config.code_revision,
            resolved_path=resolved,
            model_class=model_name,
            code_identity=_model_code_identity(self.model),
            is_stub=is_stub,
        )
        self.last_layout: PatchLayout | None = None

    def train(self, mode: bool = True) -> "LeVJEPABackbone":
        super().train(mode)
        # The adapter belongs outside this wrapper and remains trainable.  The
        # feature model itself must never inherit a parent module's train mode.
        self.model.eval()
        return self

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 6:
            raise ValueError(
                "video must have shape [B,V,3,F,H,W], "
                f"got {tuple(video.shape)}"
            )
        batch, cameras, channels, frames, height, width = video.shape
        if channels != 3:
            raise ValueError(f"video must contain 3 color channels, got {channels}")
        if batch < 1 or cameras < 1 or frames < 1 or height < 1 or width < 1:
            raise ValueError("video dimensions must all be positive")

        flat_video = video.reshape(batch * cameras, channels, frames, height, width)
        chunks: list[torch.Tensor] = []
        # no_grad creates ordinary tensors that the trainable adapter can save
        # during backward; inference_mode tensors would break that use case.
        with torch.no_grad():
            for chunk in flat_video.split(self.config.camera_chunk_size, dim=0):
                chunks.append(_extract_token_tensor(_invoke_model(self.model, chunk)))
        if not chunks:
            raise RuntimeError("no camera chunks were encoded")
        tokens = torch.cat(chunks, dim=0)
        if tokens.ndim != 3:
            raise ValueError(
                "LeVJEPA output must have shape [B*V,T,C], "
                f"got {tuple(tokens.shape)}"
            )
        if tokens.shape[0] != batch * cameras:
            raise ValueError(
                f"LeVJEPA output batch is {tokens.shape[0]}; expected {batch * cameras}"
            )
        if tokens.shape[-1] != self.config.expected_feature_dim:
            raise ValueError(
                f"LeVJEPA feature width is {tokens.shape[-1]}; expected "
                f"{self.config.expected_feature_dim}"
            )
        if self.config.expected_has_cls_token:
            if tokens.shape[1] < 2:
                raise ValueError("LeVJEPA output does not contain CLS plus patch tokens")
            tokens = tokens[:, 1:, :]

        self.last_layout = _validate_layout(tokens.shape[1], frames)
        return tokens.reshape(batch, cameras, tokens.shape[1], tokens.shape[2])


def _sinusoidal_positions(length: int, width: int, device: torch.device) -> torch.Tensor:
    """Return deterministic positions without imposing a maximum patch count."""

    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    half_width = (width + 1) // 2
    frequencies = torch.exp(
        torch.arange(half_width, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(half_width - 1, 1))
    ).unsqueeze(0)
    angles = positions * frequencies
    encoding = torch.cat((angles.sin(), angles.cos()), dim=1)[:, :width]
    return encoding


class SceneAdapter(nn.Module):
    """Fuse multiview patch tokens into a bounded trainable scene sequence."""

    def __init__(
        self,
        feature_dim: int,
        context_dim: int,
        num_scene_tokens: int,
        num_cameras: int,
        num_frames: int = 16,
        *,
        num_heads: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("feature_dim", feature_dim),
            ("context_dim", context_dim),
            ("num_scene_tokens", num_scene_tokens),
            ("num_cameras", num_cameras),
            ("num_frames", num_frames),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if num_heads is None:
            num_heads = next(
                candidate for candidate in (8, 4, 2, 1) if context_dim % candidate == 0
            )
        if num_heads < 1 or context_dim % num_heads:
            raise ValueError("context_dim must be divisible by num_heads")

        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.num_scene_tokens = num_scene_tokens
        self.num_cameras = num_cameras
        self.num_frames = num_frames

        self.input_projection = nn.Linear(feature_dim, context_dim)
        self.camera_embedding = nn.Embedding(num_cameras, context_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(1, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )
        self.input_norm = nn.LayerNorm(context_dim)
        self.scene_queries = nn.Parameter(torch.empty(num_scene_tokens, context_dim))
        self.cross_attention = nn.MultiheadAttention(
            context_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(context_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(context_dim, 4 * context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * context_dim, context_dim),
        )
        nn.init.normal_(self.scene_queries, std=0.02)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        camera_ids: torch.Tensor,
        frame_times_s: torch.Tensor,
    ) -> torch.Tensor:
        if patch_tokens.ndim != 4:
            raise ValueError(
                "patch_tokens must have shape [B,V,P,C], "
                f"got {tuple(patch_tokens.shape)}"
            )
        batch, cameras, patches, width = patch_tokens.shape
        if width != self.feature_dim:
            raise ValueError(f"patch feature width is {width}; expected {self.feature_dim}")
        if camera_ids.shape != (batch, cameras):
            raise ValueError(
                f"camera_ids must have shape {(batch, cameras)}, got "
                f"{tuple(camera_ids.shape)}"
            )
        if camera_ids.dtype != torch.long:
            raise TypeError("camera_ids must use torch.long")
        if frame_times_s.shape != (batch, cameras, self.num_frames):
            raise ValueError(
                f"frame_times_s must have shape {(batch, cameras, self.num_frames)}, "
                f"got {tuple(frame_times_s.shape)}"
            )
        if not torch.isfinite(patch_tokens).all() or not torch.isfinite(frame_times_s).all():
            raise ValueError("patch tokens and frame times must be finite")
        if torch.any(frame_times_s > 1e-6):
            raise ValueError("future-data leakage: frame_times_s must not exceed the anchor")
        if torch.any(frame_times_s[..., 1:] < frame_times_s[..., :-1]):
            raise ValueError("frame_times_s must be chronological within every camera")
        if torch.any(camera_ids < 0) or torch.any(camera_ids >= self.num_cameras):
            raise ValueError("camera_ids contain an out-of-vocabulary value")

        layout = _validate_layout(patches, self.num_frames)
        spatial = _sinusoidal_positions(
            layout.patches_per_frame,
            self.context_dim,
            patch_tokens.device,
        ).to(dtype=patch_tokens.dtype)

        memory = self.input_projection(patch_tokens)
        memory = memory.reshape(
            batch,
            cameras,
            self.num_frames,
            layout.patches_per_frame,
            self.context_dim,
        )
        camera_features = self.camera_embedding(camera_ids).reshape(
            batch, cameras, 1, 1, self.context_dim
        )
        time_features = self.time_projection(
            frame_times_s.to(dtype=memory.dtype).unsqueeze(-1)
        ).unsqueeze(3)
        memory = memory + camera_features + time_features
        memory = memory + spatial.reshape(1, 1, 1, layout.patches_per_frame, -1)
        memory = self.input_norm(memory)
        memory = memory.reshape(batch, cameras * patches, self.context_dim)

        queries = self.scene_queries.unsqueeze(0).expand(batch, -1, -1)
        attended, _ = self.cross_attention(
            queries,
            memory,
            memory,
            need_weights=False,
        )
        scene = self.output_norm(queries + attended)
        return scene + self.output_mlp(scene)


def build_backbone(
    config: BackboneConfig,
    *,
    model: nn.Module | None = None,
) -> LeVJEPABackbone:
    """Construct a frozen wrapper without importing real-model code eagerly."""

    return LeVJEPABackbone(config, model=model)


__all__ = [
    "BackboneIdentity",
    "LeVJEPABackbone",
    "PatchLayout",
    "SceneAdapter",
    "build_backbone",
]
