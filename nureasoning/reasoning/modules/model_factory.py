"""Resolve the right ``transformers`` model class for a checkpoint.

The reasoning pipeline is backbone-agnostic: Qwen3.5, Qwen3-VL and other
image-text-to-text checkpoints all load through the same code path. The class is
taken from the checkpoint's own ``architectures`` field, so no per-model wrapper
is needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import transformers
from transformers import AutoConfig


def resolve_model_class(model_path: str | Path, override: str | None = None) -> Any:
    """Return the model class to instantiate for *model_path*.

    *override* is the class name from the training config (``model_class:``) and
    wins when given, which is the escape hatch for checkpoints whose config
    advertises an architecture that is not exported by the installed
    ``transformers``.
    """
    if override:
        cls = getattr(transformers, override, None)
        if cls is None:
            raise SystemExit(
                f"model_class '{override}' is not available in transformers "
                f"{transformers.__version__}."
            )
        return cls

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    for arch in getattr(config, "architectures", None) or []:
        cls = getattr(transformers, arch, None)
        if cls is not None:
            return cls

    auto_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if auto_cls is not None:
        return auto_cls
    raise SystemExit(
        f"Could not resolve a model class for {model_path}. Set 'model_class' "
        "in the training config to a class exported by transformers "
        f"{transformers.__version__}."
    )
