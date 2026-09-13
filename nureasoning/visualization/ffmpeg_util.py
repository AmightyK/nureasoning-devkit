"""Locate an ffmpeg binary for matplotlib MP4 writers."""

from __future__ import annotations

import os
import shutil

import matplotlib


def require_ffmpeg() -> str:
    """Return an ffmpeg executable path and point matplotlib at it.

    Prefers a system ``ffmpeg``, then the binary shipped by ``imageio-ffmpeg``.
    """
    path = shutil.which("ffmpeg")
    if not path:
        try:
            import imageio_ffmpeg

            candidate = imageio_ffmpeg.get_ffmpeg_exe()
            if candidate and os.path.isfile(candidate):
                path = candidate
        except Exception:
            path = None
    if not path:
        raise RuntimeError(
            "ffmpeg is required for --video. Install a binary with "
            "`conda install -c conda-forge ffmpeg` or `pip install imageio-ffmpeg`."
        )
    matplotlib.rcParams["animation.ffmpeg_path"] = path
    return path
