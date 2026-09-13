"""Pickle helpers that remain compatible with legacy ``data_schema`` modules."""

from __future__ import annotations

import importlib
import pickle
import sys
from typing import Any


def install_schema_aliases() -> None:
    """Register ``data_schema`` / ``data_schema_v0`` aliases for stored pickles."""
    try:
        schema_module = importlib.import_module("nureasoning.common.schema")
    except Exception:
        return
    for legacy_name in ("data_schema", "data_schema_v0"):
        sys.modules.setdefault(legacy_name, schema_module)


def load_pickle(path: str) -> Any:
    """Load a pickle file with legacy schema-module aliases installed."""
    install_schema_aliases()
    with open(path, "rb") as f:
        return pickle.load(f)
