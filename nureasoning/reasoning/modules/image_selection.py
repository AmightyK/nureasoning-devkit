"""Select image paths from a NuReasoning VQA JSON record."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _sorted_cameras(cam_seq: dict[str, Any]) -> list[str]:
    return sorted(cam_seq.keys())


def nuvla_image_context(camera: str, relative_index: Any = 0) -> str:
    """Caption used by nuVLA training: ``t=-1, left camera.`` / ``t=0 (current), front camera.``"""
    try:
        rel = int(relative_index)
    except (TypeError, ValueError):
        rel = 0
    t_label = f"t={rel}" if rel < 0 else "t=0 (current)"
    cam = str(camera or "unknown").strip() or "unknown"
    return f"{t_label}, {cam} camera."


def _slot(path: Any, camera: str, relative_index: Any) -> dict[str, Any] | None:
    if not path:
        return None
    return {
        "path": str(path),
        "camera": camera,
        "relative_index": relative_index,
    }


def collect_image_slots(
    vqa_doc: dict[str, Any],
    *,
    max_images: int | None = 32,
    num_forward_frames: int | None = None,
    history_first: bool = False,
) -> list[dict[str, Any]]:
    """Ordered ``{path, camera, relative_index}`` slots from ``camera_sequences``.

    Always multi-view multi-frame: every camera, each camera's last
    ``num_forward_frames`` timestamps. Captions from ``nuvla_image_context``
    match nuVLA ``load_vlm_observation`` / ``VLATrainer``.
    """
    ctx = vqa_doc.get("temporal_multiview_context") or {}
    seqs = ctx.get("camera_sequences") or {}
    if not isinstance(seqs, dict):
        return []

    cameras = _sorted_cameras(seqs)
    cam_to_frames: dict[str, list[dict[str, Any]]] = {}
    for cam in cameras:
        frames = seqs.get(cam) or []
        if not isinstance(frames, list):
            continue
        ordered = sorted(frames, key=lambda x: x.get("relative_index", 0))
        if num_forward_frames is not None and len(ordered) > num_forward_frames:
            ordered = ordered[-num_forward_frames:]
        cam_to_frames[cam] = ordered

    raw: list[dict[str, Any]] = []
    if history_first and any(len(v) > 1 for v in cam_to_frames.values()):
        max_len = max(len(v) for v in cam_to_frames.values()) if cam_to_frames else 0
        for slot_from_end in range(max_len, 0, -1):
            for cam in cameras:
                frames = cam_to_frames.get(cam) or []
                if len(frames) >= slot_from_end:
                    fr = frames[-slot_from_end]
                    slot = _slot(
                        fr.get("image_path"),
                        cam,
                        fr.get("relative_index", 0),
                    )
                    if slot:
                        raw.append(slot)
    else:
        for cam in cameras:
            for fr in cam_to_frames.get(cam, []):
                slot = _slot(
                    fr.get("image_path"),
                    cam,
                    fr.get("relative_index", 0),
                )
                if slot:
                    raw.append(slot)

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for slot in raw:
        path = slot["path"]
        if path not in seen:
            seen.add(path)
            out.append(slot)

    if max_images is not None and len(out) > max_images:
        out = out[:max_images]
    return out


def resolve_workspace_path(path: str, workspace_root: Path) -> Path:
    """Resolve VQA paths like ./nureasoning_training_data/... against workspace root."""
    p = path.strip()
    if p.startswith("./"):
        p = p[2:]
    return (workspace_root / p).resolve()


def resolve_image_slots(
    slots: list[dict[str, Any]],
    workspace_root: Path,
) -> tuple[list[str], list[str]]:
    """Resolve slot paths and return aligned ``(images, nuvla_image_contexts)``."""
    images: list[str] = []
    contexts: list[str] = []
    for slot in slots:
        rp = resolve_workspace_path(str(slot.get("path") or ""), workspace_root)
        if not rp.is_file():
            continue
        images.append(str(rp))
        contexts.append(
            nuvla_image_context(str(slot.get("camera") or ""), slot.get("relative_index", 0))
        )
    return images, contexts
