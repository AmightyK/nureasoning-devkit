"""GPU detection for tuning concurrent eval clients."""
from __future__ import annotations

import shutil
import subprocess


def count_gpus() -> int:
    """Return visible CUDA GPU count via nvidia-smi; 0 if unavailable."""
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return 0
        return len([ln for ln in out.stdout.strip().splitlines() if ln.strip() != ""])
    except (OSError, subprocess.TimeoutExpired):
        return 0


def parse_api_urls(spec: str | None) -> list[str]:
    """Comma-separated OpenAI-compatible base URLs (each may be a vLLM replica on one GPU)."""
    if not spec or not spec.strip():
        return []
    return [u.strip().rstrip("/") for u in spec.split(",") if u.strip()]
