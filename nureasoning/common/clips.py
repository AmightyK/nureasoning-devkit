"""Helpers for discovering clip directories on disk."""

from __future__ import annotations

import os
from typing import List


def discover_clips(root: str, max_clips: int = 0) -> List[str]:
    """
    Recursively find clip directories under *root*.

    A directory is treated as a clip when it contains ``metadata.json``.
    This supports both a flat clip folder and the released layout with
    ``part_*`` subdirectories under a split root, e.g.
    ``dataset/data/train/part_1/<clip>``.

    If *max_clips* is > 0, stop after that many clips (order is walk order,
    then sorted).
    """
    clips: List[str] = []
    if not os.path.isdir(root):
        return clips
    for dirpath, dirnames, filenames in os.walk(root):
        if "metadata.json" in filenames:
            clips.append(dirpath)
            dirnames[:] = []  # clips are self-contained; do not descend further
            if max_clips > 0 and len(clips) >= max_clips:
                break
    return sorted(clips)
