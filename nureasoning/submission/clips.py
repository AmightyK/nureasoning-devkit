"""Load challenge inference clips: cameras, ego state, and questions."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from nureasoning.common.pickle_io import load_pickle

CAMERAS = [
    "front", "front_left", "front_right", "left",
    "right", "back", "back_left", "back_right",
]

ANSWER_KEYS = ("answer", "answer_text", "correct_answer", "ground_truth", "gold")
QUESTION_FILENAME = "reasoning_questions.json"
KEY_FRAME_INDEX = 100


def load_clip_metadata(clip_path: str) -> Dict[str, Any]:
    with open(os.path.join(clip_path, "metadata.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _clip_image_path(clip_path: str, rel: str) -> str:
    if not rel:
        return ""
    rel = str(rel).replace("\\", "/")
    if os.path.isabs(rel):
        marker = "/cameras/"
        if marker in rel:
            rel = "cameras/" + rel.split(marker, 1)[1]
        else:
            rel = os.path.basename(rel)
    return os.path.normpath(os.path.join(clip_path, rel))


def cameras_for_frame(clip_path: str, frame: Dict[str, Any]) -> Dict[str, str]:
    cameras = ((frame.get("sensors") or {}).get("cameras") or {})
    resolved: Dict[str, str] = {}
    for cam in CAMERAS:
        rel = cameras.get(cam) or ""
        if rel:
            resolved[cam] = _clip_image_path(clip_path, rel)
    return resolved


def select_key_frame_idx(
    frames: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    frame_timestamp: Any = None,
    key_frame_index: Any = None,
) -> int:
    """Prefer the explicit question/metadata key-frame index over timestamps."""
    if not frames:
        return 0
    for idx in (key_frame_index, (metadata or {}).get("key_frame_index")):
        if isinstance(idx, int) and 0 <= idx < len(frames):
            return idx
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx_int < len(frames):
            return idx_int

    if frame_timestamp is not None:
        try:
            ts = int(frame_timestamp)
        except (TypeError, ValueError):
            ts = None
        if ts is not None:
            for i, frame in enumerate(frames):
                try:
                    if int(frame.get("timestamp_us") or 0) == ts:
                        return i
                except (TypeError, ValueError):
                    continue
    return min(KEY_FRAME_INDEX, len(frames) - 1)


def build_temporal_context(
    clip_path: str,
    metadata: Dict[str, Any],
    target_frame_index: int,
    history_frames: int,
    stride_frames: int,
) -> Dict[str, Any]:
    """Rebuild multi-view history from the inference clip metadata."""
    frames = metadata.get("frames") or []
    camera_sequences: Dict[str, List[Dict[str, Any]]] = {cam: [] for cam in CAMERAS}
    window: List[Dict[str, Any]] = []
    if not frames or target_frame_index < 0 or target_frame_index >= len(frames):
        return {
            "input_paradigm": "multi_view_multi_frame",
            "target_frame_index": None,
            "camera_sequences": camera_sequences,
            "current_frame_image_paths": {},
        }

    stride_frames = max(1, int(stride_frames))
    selected: List[int] = []
    for relative_second in range(-max(0, int(history_frames)), 1):
        frame_idx = target_frame_index + relative_second * stride_frames
        if 0 <= frame_idx < len(frames):
            selected.append(frame_idx)

    for frame_idx in selected:
        frame = frames[frame_idx]
        ts = int(frame.get("timestamp_us") or 0)
        relative_index = int(round((frame_idx - target_frame_index) / stride_frames))
        record = {
            "frame_index": frame_idx,
            "timestamp_us": ts,
            "relative_index": relative_index,
            "relative_time_s": relative_index,
        }
        window.append(record)
        paths = cameras_for_frame(clip_path, frame)
        for cam in CAMERAS:
            camera_sequences[cam].append({
                **record,
                "image_path": paths.get(cam, ""),
            })

    target_frame = frames[target_frame_index]
    return {
        "input_paradigm": "multi_view_multi_frame",
        "target_frame_index": target_frame_index,
        "target_timestamp_us": int(target_frame.get("timestamp_us") or 0),
        "frame_rate_hz": metadata.get("frame_rate_hz"),
        "history_frames": history_frames,
        "stride_frames": stride_frames,
        "history_layout": [item["relative_index"] for item in window],
        "window_frame_indices": window,
        "camera_sequences": camera_sequences,
        "current_frame_image_paths": cameras_for_frame(clip_path, target_frame),
    }


def _sanitize_questions(questions: Any) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for question in questions or []:
        if not isinstance(question, dict):
            continue
        sanitized.append({k: v for k, v in question.items() if k not in ANSWER_KEYS})
    return sanitized


def load_clip_questions(clip_path: str) -> List[Dict[str, Any]]:
    path = os.path.join(clip_path, QUESTION_FILENAME)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        payload = {"questions": payload}
    return _sanitize_questions(payload.get("questions"))


def load_clip_sample(clip_path: str) -> Dict[str, Any]:
    """Load one inference clip: metadata, key-frame cameras, and questions."""
    clip_name = os.path.basename(os.path.normpath(clip_path))
    question_path = os.path.join(clip_path, QUESTION_FILENAME)
    payload: Dict[str, Any] = {}
    if os.path.isfile(question_path):
        with open(question_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        payload = loaded if isinstance(loaded, dict) else {"questions": loaded}

    questions = _sanitize_questions(payload.get("questions"))
    sample: Dict[str, Any] = {
        "file_path": question_path if os.path.isfile(question_path) else None,
        "clip": clip_name,
        "clip_path": clip_path,
        "frame_timestamp": payload.get("frame_timestamp"),
        "image_paths": {},
        "temporal_multiview_context": {},
        "questions": questions,
        "num_questions": len(questions),
        "key_frame_index": payload.get("key_frame_index"),
        "clip_token": None,
        "log_name": None,
    }

    if not os.path.isfile(os.path.join(clip_path, "metadata.json")):
        return sample

    metadata = load_clip_metadata(clip_path)
    frames = metadata.get("frames") or []
    target_idx = select_key_frame_idx(
        frames,
        metadata,
        frame_timestamp=payload.get("frame_timestamp"),
        key_frame_index=payload.get("key_frame_index"),
    )
    vqa_ctx = payload.get("temporal_multiview_context") or {}
    history_frames = int(vqa_ctx.get("history_frames") or 1)
    stride_frames = int(
        vqa_ctx.get("stride_frames")
        or max(1, int(round(float(metadata.get("frame_rate_hz") or 10))))
    )
    temporal = build_temporal_context(
        clip_path, metadata, target_idx, history_frames, stride_frames
    )
    sample["clip_token"] = metadata.get("clip_token")
    sample["log_name"] = metadata.get("log_name")
    sample["frame_timestamp"] = temporal.get(
        "target_timestamp_us", payload.get("frame_timestamp")
    )
    sample["image_paths"] = temporal.get("current_frame_image_paths") or {}
    sample["temporal_multiview_context"] = temporal
    sample["key_frame_index"] = target_idx
    sample["metadata"] = metadata
    sample["frames"] = frames
    return sample


def load_key_frame_ego_state(clip_path: str, frame: Dict[str, Any]) -> Any:
    ego_rel = frame.get("ego_state")
    if not ego_rel:
        raise FileNotFoundError(f"frame has no ego_state entry in {clip_path}")
    return load_pickle(os.path.join(clip_path, ego_rel))


