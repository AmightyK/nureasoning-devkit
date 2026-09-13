"""Resolve Hugging Face checkpoints to a local directory or project cache."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def models_dir() -> str:
    override = os.environ.get("NUREASONING_MODELS_DIR")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(_REPO_ROOT, "models")


def default_hf_home() -> str:
    return os.path.join(models_dir(), ".hf")


def configure_hf_home() -> str:
    """Keep Hub downloads under ``<repo>/models/.hf`` unless ``HF_HOME`` is set."""
    os.environ.setdefault("HF_HOME", default_hf_home())
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    return os.environ["HF_HOME"]


def resolve_pretrained_path(model_id: str) -> str:
    """Return a local snapshot directory when one exists, else the original id."""
    expanded = os.path.expanduser(str(model_id).strip())
    if os.path.isdir(expanded) and os.path.isfile(os.path.join(expanded, "config.json")):
        return os.path.abspath(expanded)

    root = models_dir()
    name = expanded.split("/")[-1]
    for candidate in (
        os.path.join(root, expanded),
        os.path.join(root, name),
    ):
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "config.json")):
            return os.path.abspath(candidate)
    return expanded


def pretrained_load_args(model_id: str, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
    """Path + kwargs for ``from_pretrained``, preferring local files."""
    configure_hf_home()
    path = resolve_pretrained_path(model_id)
    extra = dict(kwargs)
    extra.setdefault("cache_dir", os.path.join(os.environ["HF_HOME"], "hub"))
    if os.path.isdir(path):
        extra.setdefault("local_files_only", True)
    return path, extra


def from_pretrained(loader: Callable[..., Any], model_id: str, **kwargs: Any) -> Any:
    path, extra = pretrained_load_args(model_id, **kwargs)
    return loader(path, **extra)
