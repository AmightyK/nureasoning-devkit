"""Frame and question selection shared by SFT-data building and evaluation.

Training and evaluation must see the same frames and the same question subset,
otherwise the reported numbers are not comparable. Both pipelines therefore go
through the helpers in this module.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _question_categories(path: Path) -> set[str]:
    """Lower-cased ``category`` values of the questions in one VQA file."""
    try:
        with path.open(encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()
    return {
        str(q.get("category") or "").lower()
        for q in (doc.get("questions") or [])
    }


def keyframe_paths(vqa_dir: Path) -> list[Path]:
    """One VQA file per clip, nearest the middle of the clip.

    ``nureasoning.vqa.generate`` emits one ``<timestamp>_vqa.json`` per reasoning
    frame. Keyframe-only mode keeps a single frame per clip so that a clip does
    not dominate the dataset with near-duplicate observations.

    Spatial reasoning is annotated on every reasoning frame (~1 Hz) but Decision
    and Counterfactual text only on key frames (every 5 s, including t ≈ 10 s), so
    the frame nearest the middle that carries non-Spatial questions is preferred.
    If no frame does, the nearest frame with any questions is used.
    """
    clips: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(vqa_dir.rglob("*_vqa.json")):
        clips[p.parent.name].append(p)

    out: list[Path] = []
    for clip_name in sorted(clips):
        frames = sorted(clips[clip_name])
        mid = len(frames) // 2
        order = sorted(range(len(frames)), key=lambda i: (abs(i - mid), i))
        fallback: Path | None = None
        for i in order:
            categories = _question_categories(frames[i])
            if not categories:
                continue
            if categories - {"spatial"}:
                out.append(frames[i])
                break
            if fallback is None:
                fallback = frames[i]
        else:
            if fallback is not None:
                out.append(fallback)
    return out


def vqa_paths(vqa_dir: Path, *, keyframe_only: bool) -> list[Path]:
    if keyframe_only:
        return keyframe_paths(vqa_dir)
    return sorted(vqa_dir.rglob("*_vqa.json"))


def frame_rng(qa_seed: int, clip_dir: str, frame_timestamp: str) -> random.Random:
    """Deterministic RNG keyed on (seed, clip, frame).

    Keying on the frame rather than on iteration order means that re-running
    evaluation, swapping models, or scanning the clips in a different order all
    select the exact same question subset for a given frame.
    """
    digest = hashlib.sha256(
        f"{qa_seed}|{clip_dir}|{frame_timestamp}".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def stratified_sample(
    questions: list[dict[str, Any]], max_n: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Sample up to *max_n* questions while keeping every question_type present."""
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        by_type[str(q.get("question_type") or "unknown")].append(q)
    types = sorted(by_type)
    if not types:
        return []

    selected: list[dict[str, Any]] = []
    budget = max_n
    per_type = max(1, budget // len(types))
    for t in types:
        pool = by_type[t]
        take = min(len(pool), per_type)
        selected.extend(rng.sample(pool, take))
        budget -= take
    if budget > 0:
        chosen = {id(q) for q in selected}
        leftover = [q for q in questions if id(q) not in chosen]
        if leftover:
            selected.extend(rng.sample(leftover, min(len(leftover), budget)))
    return selected[:max_n]


def select_questions(
    doc: dict[str, Any],
    *,
    clip_dir: str,
    frame_timestamp: str,
    max_spatial_per_frame: int | None = None,
    max_qa_per_frame: int | None = None,
    qa_seed: int = 42,
) -> list[dict[str, Any]]:
    """Apply the per-frame question cap.

    ``max_spatial_per_frame`` caps only the Spatial category, which is by far the
    most numerous, and keeps Decision / Counterfactual questions in full. This is
    the recommended setting: it rebalances the mix without discarding the rarer
    reasoning types. ``max_qa_per_frame`` is the blunter alternative and caps the
    frame as a whole, stratified across question types.
    """
    questions = list(doc.get("questions") or [])
    if not questions:
        return []

    rng = frame_rng(qa_seed, clip_dir, frame_timestamp)

    if max_spatial_per_frame is not None:
        spatial = [q for q in questions
                   if str(q.get("category") or "").lower() == "spatial"]
        other = [q for q in questions
                 if str(q.get("category") or "").lower() != "spatial"]
        if len(spatial) > max_spatial_per_frame:
            spatial = rng.sample(
                sorted(spatial, key=lambda q: str(q.get("question_id") or "")),
                max_spatial_per_frame,
            )
        return other + spatial

    if max_qa_per_frame is not None and len(questions) > max_qa_per_frame:
        return stratified_sample(questions, max_qa_per_frame, rng)

    return questions


def frame_timestamp_of(doc: dict[str, Any], path: Path) -> str:
    return str(doc.get("frame_timestamp") or path.stem.replace("_vqa", ""))
