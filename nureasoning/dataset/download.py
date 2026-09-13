"""Download and extract the nuReasoning dataset.

    python -m nureasoning.dataset.download --local-dir ./dataset
    python -m nureasoning.dataset.download --local-dir ./dataset --splits train --parts part_1
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tarfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "nureasoning/nuReasoning"
SPLITS = ("train", "validation", "test")
TAR_SUFFIX = ".tar"


def _part_name(part: str) -> str:
    part = part.strip()
    return part if part.startswith("part_") else f"part_{part}"


def _allow_patterns(splits: list[str] | None, parts: list[str] | None) -> list[str] | None:
    if not splits and not parts:
        return None
    splits = splits or list(SPLITS)
    if parts:
        parts = [_part_name(p) for p in parts]
        patterns = [
            pattern
            for split in splits
            for part in parts
            for pattern in (f"data/{split}/{part}/*", f"data/{split}/{part}/**")
        ]
    else:
        patterns = [
            pattern
            for split in splits
            for pattern in (f"data/{split}/*", f"data/{split}/**")
        ]
    return patterns + ["*.md", "*.ipynb", "*.json"]


def _flatten(clip_dir: str) -> None:
    """Clip archives wrap contents in ``<clip_name>/``; strip that extra level."""
    if os.path.isfile(os.path.join(clip_dir, "metadata.json")):
        return
    nested = os.path.join(clip_dir, os.path.basename(clip_dir))
    if not os.path.isdir(nested):
        return
    for name in os.listdir(nested):
        dst = os.path.join(clip_dir, name)
        if not os.path.exists(dst):
            shutil.move(os.path.join(nested, name), dst)
    if not os.listdir(nested):
        os.rmdir(nested)


def _extract_tar(archive_path: str, out_dir: str) -> None:
    with tarfile.open(archive_path) as tf:
        tf.extractall(out_dir)


def _hub_token() -> str | None:
    """Return the Hub token, including the default user cache if HF_HOME was redirected."""
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    token = get_token()
    if token:
        return token
    default_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.isfile(default_path):
        with open(default_path, encoding="utf-8") as handle:
            return handle.read().strip() or None
    return None


def download(
    local_dir: str,
    splits: list[str] | None = None,
    parts: list[str] | None = None,
    max_workers: int = 16,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Install huggingface_hub: pip install -U huggingface_hub") from exc

    logger.info("Downloading %s to %s (splits=%s, parts=%s)",
                REPO_ID, local_dir, splits or "all", parts or "all")
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=local_dir,
            max_workers=max_workers,
            allow_patterns=_allow_patterns(splits, parts),
            token=_hub_token(),
        )
    except Exception as exc:
        try:
            from huggingface_hub.errors import (
                GatedRepoError,
                LocalTokenNotFoundError,
                RepositoryNotFoundError,
            )
        except ImportError:
            raise
        if isinstance(exc, (GatedRepoError, LocalTokenNotFoundError, RepositoryNotFoundError)):
            raise SystemExit(
                f"Cannot access dataset {REPO_ID}. {exc}\n"
                "Run `hf auth login` with a token that has access, then retry."
            ) from exc
        raise


def extract(
    local_dir: str,
    remove_tars: bool = True,
    overwrite: bool = False,
    splits: list[str] | None = None,
    parts: list[str] | None = None,
) -> None:
    data_dir = os.path.join(local_dir, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"{data_dir} does not exist")

    part_filter = {_part_name(p) for p in parts} if parts else None
    extracted = 0
    for dirpath, _, filenames in os.walk(data_dir):
        for name in sorted(filenames):
            if not name.endswith(TAR_SUFFIX):
                continue
            archive_path = os.path.join(dirpath, name)
            rel = os.path.relpath(archive_path, data_dir).replace("\\", "/").split("/")
            if splits and rel[0] not in splits:
                continue
            if part_filter and (len(rel) < 2 or rel[1] not in part_filter):
                continue
            out_dir = os.path.join(dirpath, os.path.splitext(name)[0])
            if os.path.isfile(os.path.join(out_dir, "metadata.json")) and not overwrite:
                if remove_tars:
                    os.remove(archive_path)
                continue
            os.makedirs(out_dir, exist_ok=True)
            try:
                _extract_tar(archive_path, out_dir)
            except (tarfile.TarError, OSError):
                logger.error("Corrupted archive, skipping: %s", archive_path)
                continue
            _flatten(out_dir)
            extracted += 1
            if remove_tars:
                os.remove(archive_path)

    for dirpath, dirnames, filenames in os.walk(data_dir):
        if "metadata.json" in filenames:
            dirnames.clear()
            continue
        _flatten(dirpath)
    logger.info("Extracted %d archives into %s", extracted, data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the nuReasoning dataset")
    parser.add_argument("--local-dir", default="./dataset")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=None)
    parser.add_argument("--parts", nargs="+", default=None,
                        help="e.g. part_1 or 1 (default: all parts of the selected splits)")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--extract-only", action="store_true",
                        help="Extract already-downloaded .tar files without downloading")
    parser.add_argument("--keep-tars", action="store_true",
                        help="Keep clip .tar files after a successful extract")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.extract_only:
        download(args.local_dir, args.splits, args.parts, args.max_workers)
    extract(args.local_dir, remove_tars=not args.keep_tars,
            overwrite=args.overwrite, splits=args.splits, parts=args.parts)


if __name__ == "__main__":
    main()
