"""Build the evaluation manifest from ``*_vqa.json`` files.

One manifest line = one question, together with the ordered image list that the
model is shown and the ground truth used for scoring.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nureasoning.reasoning.modules.image_selection import (
    collect_image_slots,
    resolve_image_slots,
)
from nureasoning.reasoning.modules.sampling import (
    frame_timestamp_of,
    select_questions,
    vqa_paths,
)


def build_manifest(
    vqa_dir: Path,
    out_manifest: Path,
    *,
    workspace: Path,
    max_images: int = 16,
    num_forward_frames: int | None = 2,
    history_first: bool = False,
    keyframe_only: bool = False,
    max_spatial_per_frame: int | None = None,
    qa_seed: int = 42,
) -> int:
    """Write *out_manifest* and return the number of questions in it.

    Image paths written by ``nureasoning.vqa.generate`` are absolute, so
    *workspace* only matters for VQA files that carry workspace-relative paths.
    """
    if not vqa_dir.is_dir():
        raise FileNotFoundError(f"VQA directory not found: {vqa_dir}")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    paths = vqa_paths(vqa_dir, keyframe_only=keyframe_only)
    kind = "keyframes" if keyframe_only else "VQA files"
    print(f"  Scanning {vqa_dir}: {len(paths)} {kind}")
    if max_spatial_per_frame is not None:
        print(
            f"  Spatial cap: {max_spatial_per_frame} questions/frame "
            f"(deterministic per-frame RNG, qa_seed={qa_seed})"
        )

    n = 0
    n_no_images = 0
    with out_manifest.open("w", encoding="utf-8") as out:
        for path in paths:
            with path.open(encoding="utf-8") as f:
                doc = json.load(f)
            clip_dir = path.parent.name
            frame_ts = frame_timestamp_of(doc, path)

            image_paths, image_contexts = resolve_image_slots(
                collect_image_slots(
                    doc,
                    max_images=max_images,
                    num_forward_frames=num_forward_frames,
                    history_first=history_first,
                ),
                workspace,
            )
            if not image_paths:
                n_no_images += 1
                continue

            questions = select_questions(
                doc,
                clip_dir=clip_dir,
                frame_timestamp=frame_ts,
                max_spatial_per_frame=max_spatial_per_frame,
                qa_seed=qa_seed,
            )
            for q in questions:
                record: dict[str, Any] = {
                    "sample_id": f"{clip_dir}/{frame_ts}/{q.get('question_id')}",
                    "vqa_path": str(path),
                    "clip_dir": clip_dir,
                    "frame_timestamp": frame_ts,
                    "question_id": q.get("question_id"),
                    "question_type": q.get("question_type"),
                    "category": q.get("category"),
                    "subcategory": q.get("subcategory"),
                    "input_paradigm": q.get("input_paradigm"),
                    "view_requirement": q.get("view_requirement"),
                    "question": q.get("question"),
                    "choices": q.get("choices"),
                    "gold": {
                        "answer": q.get("answer"),
                        "answer_text": q.get("answer_text"),
                        "tolerance": q.get("tolerance"),
                        "answer_format": q.get("answer_format"),
                    },
                    "image_paths": image_paths,
                    "image_contexts": image_contexts,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1

    if n_no_images:
        print(f"  Skipped {n_no_images} frames whose images could not be resolved")
    return n


def load_manifest(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows
