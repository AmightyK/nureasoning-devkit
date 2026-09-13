"""
Generate VQA (Visual Question Answering) data from structured nureasoning JSON.

Produces both multiple-choice (A/B/C/D) and numerical questions grounded in
the Spatial, Driving, and Counterfactual sections of per-frame reasoning files.

Usage:
    python -m nureasoning.vqa.generate                              # process all clips
    python -m nureasoning.vqa.generate --clip <clip_dir>            # single clip
    python -m nureasoning.vqa.generate --key-frame 100              # only the keyframe index (e.g. 100 ≈ 10s at 10Hz)
    python -m nureasoning.vqa.generate --gemini-rephrase            # use Gemini to diversify question phrasing
"""

import argparse
import json
import math
import os
import random
import sys
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAMERAS = [
    "front", "front_left", "front_right", "left",
    "right", "back", "back_left", "back_right",
]

CATEGORY_DISPLAY = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus": "bus",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian": "pedestrian",
    "vehicle.construction": "construction vehicle",
    "vehicle.emergency": "emergency vehicle",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic cone",
    "static_object.bicycle_rack": "bicycle rack",
}

DIRECTION_DISPLAY = {
    "front": "in front of",
    "front_aligned": "directly in front of",
    "front_left": "to the front-left of",
    "front_right": "to the front-right of",
    "rear": "behind",
    "rear_aligned": "directly behind",
    "rear_left": "to the rear-left of",
    "rear_right": "to the rear-right of",
    "behind": "behind",
    "behind_aligned": "directly behind",
    "back_left": "to the rear-left of",
    "back_right": "to the rear-right of",
    "left_side": "to the left of",
    "right_side": "to the right of",
}

LONGITUDINAL_OPTIONS = [
    "Remain stopped",
    "Quickly come to a stop",
    "Gently come to a stop",
    "Slow down quickly",
    "Slow down gently",
    "Quickly accelerate",
    "Gently accelerate",
    "Maintain speed",
    "Reverse",
]

LATERAL_OPTIONS = [
    "Slightly move left in the lane",
    "Slightly move right in the lane",
    "Left lane change",
    "Right lane change",
    "Turn left",
    "Turn right",
    "No lateral action",
]

LONGITUDINAL_DEFINITIONS = {
    "Remain stopped": (
        "Maintain a complete stop when already stationary and stopping is still required. "
        "Used when a control point remains active (red light, stop sign, yield condition, "
        "school bus, rail crossing) or when a hazard or obstacle is still present ahead "
        "within the stopping zone. Continue holding position until the control condition "
        "clears, the hazard is resolved, or it is safe and lawful to proceed."
    ),
    "Quickly come to a stop": (
        "Apply rapid deceleration to a complete stop and remain stopped at control points "
        "(stop or yield lines, red lights, school bus, rail crossing) or when an immediate "
        "hazard requires it. Use when the need to stop is urgent—because the constraint is "
        "close or the available stopping distance is short."
    ),
    "Gently come to a stop": (
        "Use gradual deceleration to come to a complete stop at control points "
        "(stop or yield lines, red lights, school bus, rail crossing) or when a hazard "
        "ahead requires a stop but is not yet urgent. Choose this when the constraint or "
        "control point is visible and there is enough time and distance for a smooth, "
        "comfortable stop."
    ),
    "Slow down quickly": (
        "Apply stronger deceleration to reduce speed soon when a roadway feature "
        "(curve, grade, bump, ramp, roundabout, turn), limited visibility, occlusion, "
        "work zone, or other uncertainty is close ahead. Used when the need to slow is "
        "immediate and the available distance is short. Do not intend to fully stop; "
        "reduce speed enough to pass through or handle the condition safely."
    ),
    "Slow down gently": (
        "Apply gradual deceleration when a roadway feature (curve, grade, bump, ramp, "
        "roundabout, turn), limited visibility, occlusion, work zone, or other uncertainty "
        "is visible but still at a distance. Use when there is enough time and space to "
        "reduce speed smoothly. Do not intend to fully stop; lower speed for comfort and "
        "safety."
    ),
    "Quickly accelerate": (
        "Increase speed promptly when timing matters, such as clearing an intersection on "
        "green before it changes, merging into fast-moving traffic, or moving into a brief "
        "gap. Use when the window to act is short and a stronger acceleration is needed for "
        "safety or to finish the maneuver. Often used with a lateral action "
        "(e.g., lane change or merge)."
    ),
    "Gently accelerate": (
        "Increase speed gradually for normal progress, such as cruising in lane, finishing "
        "a lane change, or resuming speed after yielding. Use when there is no tight time "
        "pressure. Keeps the ride smooth and avoids sudden acceleration."
    ),
    "Maintain speed": (
        "Maintain the current speed when unconstrained by traffic, obstacles, geometry, or "
        "control rules; allows only minor adjustments for smoothness."
    ),
    "Reverse": (
        "Low-speed backward motion used only when the ego vehicle is already stopped, and "
        "only for short recovery maneuvers such as parking correction, dead-end recovery, "
        "or unblocking."
    ),
}

LATERAL_DEFINITIONS = {
    "Slightly move left in the lane": (
        "Temporarily shift toward the left side of the lane without crossing the left lane "
        "line. Use to increase clearance from a blockage or hazard on the right "
        "(e.g., parked car, cyclist, debris, narrow shoulder). Remain inside the lane; "
        "this is not a lane change."
    ),
    "Slightly move right in the lane": (
        "Temporarily shift toward the right side of the lane without crossing the right "
        "lane line. Use to increase clearance from a blockage or hazard on the left "
        "(e.g., oncoming traffic on an undivided road, obstruction near the centerline). "
        "Remain inside the lane; this is not a lane change."
    ),
    "Left lane change": (
        "Shift fully into the left adjacent lane by crossing the left lane line. Use when "
        "passing a slower vehicle, preparing for a left turn, or avoiding a blockage that "
        "needs a full lane change. Check the left lane, confirm a safe gap, then move into "
        "the target lane. This is a full lane change, not an in-lane nudge."
    ),
    "Right lane change": (
        "Shift fully into the right adjacent lane by crossing the right lane line. Use when "
        "moving to a slower lane, preparing for a right turn or exit, or avoiding a "
        "blockage that needs a full lane change. Check the right lane, confirm a safe gap, "
        "then move into the target lane. This is a full lane change, not an in-lane nudge."
    ),
    "Turn left": (
        "Execute a left turn onto another road segment, with a large heading change "
        "(e.g., ~90° at an intersection). Use at intersections, T-junctions, driveways, or "
        "entrances when the route requires a left turn. Involves moving into and following "
        "the target road’s path; distinct from a left lane change on the same road."
    ),
    "Turn right": (
        "Execute a right turn onto another road segment, with a large heading change "
        "(e.g., ~90° at an intersection). Use at intersections, T-junctions, driveways, or "
        "exits when the route requires a right turn. Involves moving into and following "
        "the target road’s path; distinct from a right lane change on the same road."
    ),
    "No lateral action": (
        "Do not change lateral position. Typically used when the vehicle is stopped or when "
        "only a longitudinal adjustment is required."
    ),
}

# Legacy / informal labels remapped onto the canonical action schema.
LONGITUDINAL_ALIASES = {
    "gently decelerate": "Slow down gently",
    "quickly decelerate": "Slow down quickly",
    "decelerate gently": "Slow down gently",
    "decelerate quickly": "Slow down quickly",
    "maintain current speed": "Maintain speed",
    "keep speed": "Maintain speed",
    "come to a stop quickly": "Quickly come to a stop",
    "come to a stop gently": "Gently come to a stop",
    "hard brake": "Quickly come to a stop",
    "soft brake": "Gently come to a stop",
}

LATERAL_ALIASES = {
    "stay centered in the lane": "No lateral action",
    "stay centered": "No lateral action",
    "keep lane": "No lateral action",
    "keep centered": "No lateral action",
    "lane keep": "No lateral action",
    "change lane left": "Left lane change",
    "change lane right": "Right lane change",
    "left lane-change": "Left lane change",
    "right lane-change": "Right lane change",
    "nudge left": "Slightly move left in the lane",
    "nudge right": "Slightly move right in the lane",
}

TEMPORAL_WINDOW_SUBCATEGORIES = {
    "object_speed",
    "motion_relation",
    "conflict_prediction",
    "driving_decision_joint",
    "driving_decision_longitudinal",
    "driving_decision_lateral",
    "driving_reasoning_trace",
    "critical_vehicle_behavior",
    "scene_description",
    "unsafe_action_identification",
    "action_risk_assessment",
}

FUTURE_PREDICTION_SUBCATEGORIES = {
    "future_motion_label",
    "future_path_intersection",
    "future_xy_trajectory",
    "future_time_to_conflict_s",
}

MULTIVIEW_REQUIRED_SUBCATEGORIES = TEMPORAL_WINDOW_SUBCATEGORIES | FUTURE_PREDICTION_SUBCATEGORIES | {
    "camera_view_for_3d_object",
    "multiview_camera_pair",
    "multiview_center_coordinates_1000",
}

SAFE_ACTION_PAIR_FALLBACKS = [
    "Maintain speed + No lateral action",
    "Slow down gently + No lateral action",
    "Gently come to a stop + No lateral action",
    "Gently accelerate + No lateral action",
    "Quickly come to a stop + No lateral action",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _display_category(cat: str) -> str:
    return CATEGORY_DISPLAY.get(cat, cat.split(".")[-1] if "." in cat else cat)


def _round(val: float, n: int = 1) -> float:
    return round(val, n)


def _speed_mps(vel: Dict[str, float]) -> float:
    vx = vel.get("x", vel.get("vx", 0.0))
    vy = vel.get("y", vel.get("vy", 0.0))
    return math.sqrt(vx ** 2 + vy ** 2)


def _make_id() -> str:
    return uuid.uuid4().hex[:12]


def _coerce_reasoning_section(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _dedupe_preserve(items: List[str], *, exclude: Optional[str] = None) -> List[str]:
    seen = {exclude} if exclude is not None else set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _make_question(
    *,
    question_type: str,
    category: str,
    subcategory: str,
    question: str,
    answer_text: str,
    choices: Any = None,
    answer: Any = None,
    **extra: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "question_id": _make_id(),
        "question_type": question_type,
        "category": category,
        "subcategory": subcategory,
        "question": question,
        "choices": choices,
        "answer": answer,
        "answer_text": answer_text,
    }
    payload.update(extra)
    return payload


def _shuffle_choices(correct: str, distractors: List[str], num_choices: int = 4) -> Tuple[Dict[str, str], str]:
    """Build A/B/C/D choices with shuffled order, return (choices_dict, correct_letter)."""
    pool = _dedupe_preserve(distractors, exclude=correct)
    random.shuffle(pool)
    options = [correct] + pool[: num_choices - 1]
    while len(options) < num_choices:
        options.append("None of the above")
    random.shuffle(options)
    labels = "ABCDEFGH"
    choices = {labels[i]: options[i] for i in range(len(options))}
    answer_letter = [k for k, v in choices.items() if v == correct][0]
    return choices, answer_letter


def _bbox_to_xyxy(bbox: Any) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]

    return x1, y1, x2, y2


def _bbox_center(bbox: Any) -> Optional[Tuple[float, float]]:
    coords = _bbox_to_xyxy(bbox)
    x1, y1, x2, y2 = coords
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_size(bbox: Any) -> Optional[Tuple[float, float]]:
    coords = _bbox_to_xyxy(bbox)
    x1, y1, x2, y2 = coords
    return x2 - x1, y2 - y1


def _camera_display(cam: str) -> str:
    return cam.replace("_", " ")


@lru_cache(maxsize=512)
def _image_size(image_path: str) -> Optional[Tuple[int, int]]:
    if not image_path:
        return None
    with Image.open(image_path) as img:
        return img.size

def _normalize_to_1000(value: float, size: int) -> int:
    if size <= 1:
        return 0

    normalized = int(round((float(value) / float(size - 1)) * 1000.0))
    return max(0, min(1000, normalized))


def _normalized_bbox_coordinates(bbox: Any, image_path: str) -> Optional[Dict[str, int]]:
    coords = _bbox_to_xyxy(bbox)
    size = _image_size(image_path)
    if size is None:
        return None
    x1, y1, x2, y2 = coords
    width, height = size
    cx, cy = _bbox_center(bbox)
    return {
        "x1": _normalize_to_1000(x1, width),
        "y1": _normalize_to_1000(y1, height),
        "x2": _normalize_to_1000(x2, width),
        "y2": _normalize_to_1000(y2, height),
        "cx": _normalize_to_1000(cx, width),
        "cy": _normalize_to_1000(cy, height),
    }


def _format_bbox_1000(bbox: Any, image_path: str) -> str:
    normalized = _normalized_bbox_coordinates(bbox, image_path)
    if normalized is None:
        return "[unavailable]"
    return (
        f"[{int(normalized['x1'])}, {int(normalized['y1'])}, "
        f"{int(normalized['x2'])}, {int(normalized['y2'])}]"
    )


def _image_region_name(bbox: Any, image_path: str) -> Optional[str]:
    normalized = _normalized_bbox_coordinates(bbox, image_path)
    if normalized is None:
        return None
    cx, cy = normalized["cx"], normalized["cy"]
    horizontal = "left" if cx < 333 else "center" if cx < 667 else "right"
    vertical = "upper" if cy < 333 else "middle" if cy < 667 else "lower"
    return f"{vertical} {horizontal}"


def _collect_grounded_observations(spatial: Dict, max_relations: int = 6) -> List[Dict[str, Any]]:
    """Pair each object's 3D relation with one or more 2D camera observations."""
    observations: List[Dict[str, Any]] = []
    relations = spatial.get("object_relations", [])
    per_camera_results = spatial.get("per_camera_results", {}) or {}
    sorted_rels = sorted(
        relations,
        key=lambda r: r.get("geometric_relations", {}).get("euclidean_distance_m", 1e9),
    )
    for rel in sorted_rels[:max_relations]:
        camera_obs = rel.get("camera_observations", {}) or {}
        for cam, obs in camera_obs.items():
            bbox = obs.get("detection_bbox_2d")
            if _bbox_to_xyxy(bbox) is None:
                continue
            observations.append(
                {
                    "track_token": rel.get("track_token"),
                    "camera": cam,
                    "category": rel.get("category", obs.get("detection_label", "unknown")),
                    "bbox_2d": bbox,
                    "image_path": per_camera_results.get(cam, {}).get("image_path", ""),
                    "detection_label": obs.get("detection_label"),
                    "position_3d_ego": rel.get("position_3d_ego", {}),
                    "geometric_relations": rel.get("geometric_relations", {}),
                    "semantic_relations": rel.get("semantic_relations", {}),
                    "future_conflict": rel.get("future_conflict", {}),
                    "object_role": _object_role_from_relation(rel),
                }
            )
    return observations


def _relation_index_by_track_token(spatial: Dict) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for rel in spatial.get("object_relations", []) or []:
        token = rel.get("track_token")
        if token:
            index[str(token)] = rel
    return index


def _object_role_from_relation(rel: Optional[Dict[str, Any]]) -> str:
    if not isinstance(rel, dict):
        return "context"
    fc = rel.get("future_conflict", {}) or {}

    if fc.get("conflict_with_ego"):
        return "critical"

    return "context"


def _relation_question_metadata(rel: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target_track_token": rel.get("track_token"),
        "object_role": _object_role_from_relation(rel),
    }


def _observation_question_metadata(obs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target_track_token": obs.get("track_token"),
        "target_camera": obs.get("camera"),
        "object_role": obs.get("object_role", "context"),
    }


def _image_path_lookup(spatial: Dict[str, Any]) -> Dict[str, str]:
    per_camera_results = spatial.get("per_camera_results", {}) or {}
    return {
        cam: (cam_data or {}).get("image_path", "")
        for cam, cam_data in per_camera_results.items()
    }


def _inject_resolved_image_paths(
    spatial: Dict[str, Any],
    image_paths: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    if not image_paths:
        return spatial

    merged_spatial = dict(spatial)
    per_camera_results = _coerce_reasoning_section(merged_spatial.get("per_camera_results"))
    merged_per_camera_results: Dict[str, Any] = {
        cam: dict(cam_data) if isinstance(cam_data, dict) else {}
        for cam, cam_data in per_camera_results.items()
    }

    for cam, image_path in image_paths.items():
        cam_data = dict(merged_per_camera_results.get(cam, {}))
        cam_data["image_path"] = image_path
        merged_per_camera_results[cam] = cam_data

    merged_spatial["per_camera_results"] = merged_per_camera_results
    return merged_spatial


def _pick_primary_camera_observation(
    rel: Dict[str, Any],
    image_path_lookup: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    camera_obs = rel.get("camera_observations", {}) or {}
    best_cam = None
    best_obs = None
    best_area = -1.0
    for cam, obs in camera_obs.items():
        bbox = obs.get("detection_bbox_2d")
        size = _bbox_size(bbox)
        if size is None:
            continue
        width, height = size
        area = width * height
        if area > best_area:
            best_area = area
            best_cam = cam
            best_obs = obs
    if best_cam is None or best_obs is None:
        return None
    return {
        "camera": best_cam,
        "bbox_2d": best_obs.get("detection_bbox_2d"),
        "detection_label": best_obs.get("detection_label"),
        "image_path": (image_path_lookup or {}).get(best_cam, ""),
    }


def _relation_anchor(
    rel: Dict[str, Any],
    include_category: bool = True,
    include_3d_context: bool = False,
    image_path_lookup: Optional[Dict[str, str]] = None,
) -> str:
    pos = rel.get("position_3d_ego", {}) or {}
    primary = _pick_primary_camera_observation(rel, image_path_lookup=image_path_lookup)
    subject = "the object"
    if include_category:
        subject = f"the {_display_category(rel.get('category', 'unknown'))}"
    if primary is not None:
        subject += (
            " in the "
            f"{_camera_display(primary['camera'])} camera with normalized 2D box "
            f"{_format_bbox_1000(primary['bbox_2d'], primary.get('image_path', ''))}"
        )
    parts = [subject]
    if include_3d_context:
        xy_text = _ego_xy_position_text(pos)
        if xy_text:
            parts.append(xy_text)
    return ", ".join(parts)


def _observation_anchor(
    obs: Dict[str, Any],
    include_category: bool = True,
    include_3d_context: bool = False,
) -> str:
    pos = obs.get("position_3d_ego", {}) or {}
    subject = "the object"
    if include_category:
        subject = f"the {_display_category(obs.get('category', 'unknown'))}"
    subject += (
        " in the "
        f"{_camera_display(obs['camera'])} camera with normalized 2D box "
        f"{_format_bbox_1000(obs['bbox_2d'], obs.get('image_path', ''))}"
    )
    parts = [subject]
    if include_3d_context:
        xy_text = _ego_xy_position_text(pos)
        if xy_text:
            parts.append(xy_text)
    return ", ".join(parts)


def _semantic_position_display(semantic_relations: Dict[str, Any]) -> Optional[str]:
    front_rear = (semantic_relations or {}).get("front_rear", "")
    left_right = (semantic_relations or {}).get("left_right", "")
    if front_rear and left_right:
        key = f"{front_rear}_{left_right}"
        return DIRECTION_DISPLAY.get(key, f"{front_rear}/{left_right}")
    if front_rear:
        return DIRECTION_DISPLAY.get(front_rear, front_rear)
    if left_right:
        return DIRECTION_DISPLAY.get(left_right, left_right)
    return None


def _ego_xy_position_text(position_3d_ego: Dict[str, Any]) -> Optional[str]:
    pos = position_3d_ego or {}
    x = pos.get("x")
    y = pos.get("y")
    if x is None or y is None:
        return None
    return f"ego-frame 3D position [x, y] = [{_round(x, 1)}, {_round(y, 1)}] m"


def _append_spatial_context(
    parts: List[str],
    semantic_relations: Dict[str, Any],
    position_3d_ego: Optional[Dict[str, Any]] = None,
    include_3d_context: bool = False,
) -> None:
    position_text = _semantic_position_display(semantic_relations)
    distance_bucket = semantic_relations.get("distance_bucket")
    if position_text:
        parts.append(f"that is {position_text} the ego vehicle")
    if distance_bucket:
        parts.append(f"at {distance_bucket} range")
    if include_3d_context:
        xy_text = _ego_xy_position_text(position_3d_ego or {})
        if xy_text:
            parts.append(f"with {xy_text}")


def _observation_reference(
    obs: Dict[str, Any],
    *,
    include_camera: bool = True,
    include_3d_context: bool = False,
) -> str:
    cat = _display_category(obs.get("category", "unknown"))
    if include_camera:
        head = f"the {cat} in the {_camera_display(obs['camera'])} camera"
    else:
        head = f"the {cat} object"
    parts = [head]
    _append_spatial_context(
        parts,
        obs.get("semantic_relations", {}) or {},
        obs.get("position_3d_ego", {}) or {},
        include_3d_context=include_3d_context,
    )
    return " ".join(parts)


def _observation_reference_for_2d_target(
    obs: Dict[str, Any],
    include_3d_context: bool = False,
) -> str:
    return _observation_reference(obs, include_camera=True, include_3d_context=include_3d_context)


def _observation_reference_for_camera_view_target(
    obs: Dict[str, Any],
    include_3d_context: bool = True,
) -> str:
    return _observation_reference(obs, include_camera=False, include_3d_context=include_3d_context)


def _multiview_reference(
    correspondence_value: Dict[str, Any],
    relation_index: Dict[str, Dict[str, Any]],
    max_views: int = 2,
    include_3d_context: bool = False,
    include_view_text: bool = True,
) -> str:
    track_token = str(correspondence_value.get("track_token", ""))
    rel = relation_index.get(track_token)
    role = _object_role_from_relation(rel)
    cat = _display_category(correspondence_value.get("category", "unknown"))
    parts = [f"the same {role} {cat}"]
    if include_view_text:
        views = sorted(correspondence_value.get("views", []))[:max_views]
        view_text = " and ".join(f"{_camera_display(v)} camera" for v in views)
        parts[0] += f" observed in the {view_text}"
    if rel is not None:
        _append_spatial_context(
            parts,
            rel.get("semantic_relations", {}) or {},
            rel.get("position_3d_ego", {}) or {},
            include_3d_context=include_3d_context,
        )
    return " ".join(parts)


def _export_path(path: str, clip_dir: Optional[str] = None) -> str:
    if not path:
        return ""
    resolved = path
    if not os.path.isabs(resolved):
        if clip_dir is not None:
            resolved = os.path.join(clip_dir, resolved)
        else:
            resolved = os.path.abspath(resolved)
    return os.path.abspath(os.path.normpath(resolved))


def _resolve_frame_camera_paths(frame: Dict[str, Any], clip_dir: str) -> Dict[str, str]:
    cameras = ((frame.get("sensors") or {}).get("cameras") or {})
    resolved: Dict[str, str] = {}
    for cam in CAMERAS:
        rel_path = cameras.get(cam, "")
        if rel_path:
            resolved[cam] = _export_path(rel_path, clip_dir)
    return resolved


def _build_temporal_multiview_context(
    metadata: Dict[str, Any],
    target_frame_index: int,
    clip_dir: str,
    history_frames: int,
    stride_frames: int,
) -> Dict[str, Any]:
    frames = metadata.get("frames", []) or []
    if not frames:
        return {
            "input_paradigm": "multi_view_multi_frame",
            "target_frame_index": None,
            "camera_sequences": {},
        }

    camera_sequences: Dict[str, List[Dict[str, Any]]] = {cam: [] for cam in CAMERAS}
    window_index_records: List[Dict[str, Any]] = []
    if target_frame_index < 0 or target_frame_index >= len(frames):
        return {
            "input_paradigm": "multi_view_multi_frame",
            "target_frame_index": None,
            "target_timestamp_us": None,
            "history_frames": history_frames,
            "history_layout": [],
            "camera_sequences": camera_sequences,
            "current_frame_image_paths": {},
        }

    stride_frames = max(1, int(stride_frames))
    selected_frame_indices: List[int] = []
    for relative_second in range(-history_frames, 1):
        frame_idx = target_frame_index + relative_second * stride_frames
        if 0 <= frame_idx < len(frames):
            selected_frame_indices.append(frame_idx)

    for frame_idx in selected_frame_indices:
        frame = frames[frame_idx]
        ts = int(frame.get("timestamp_us", 0))
        relative_index = int(round((frame_idx - target_frame_index) / stride_frames))
        window_index_records.append(
            {
                "frame_index": frame_idx,
                "timestamp_us": ts,
                "relative_index": relative_index,
                "relative_time_s": relative_index,
            }
        )
        current_paths = _resolve_frame_camera_paths(frame, clip_dir)
        for cam in CAMERAS:
            image_path = current_paths.get(cam, "")
            camera_sequences[cam].append(
                {
                    "frame_index": frame_idx,
                    "timestamp_us": ts,
                    "relative_index": relative_index,
                    "relative_time_s": relative_index,
                    "image_path": image_path,
                }
            )

    target_frame = frames[target_frame_index]
    frame_rate_hz = metadata.get("frame_rate_hz")
    return {
        "input_paradigm": "multi_view_multi_frame",
        "target_frame_index": target_frame_index,
        "target_timestamp_us": int(target_frame.get("timestamp_us", 0)),
        "frame_rate_hz": frame_rate_hz,
        "history_frames": history_frames,
        "stride_frames": stride_frames,
        "history_layout": [item["relative_index"] for item in window_index_records],
        "window_frame_indices": window_index_records,
        "camera_sequences": camera_sequences,
        "current_frame_image_paths": _resolve_frame_camera_paths(target_frame, clip_dir),
    }


def _question_temporal_target(question: Dict[str, Any]) -> str:
    if question.get("subcategory") in FUTURE_PREDICTION_SUBCATEGORIES:
        return "future_prediction"
    if question.get("subcategory") in TEMPORAL_WINDOW_SUBCATEGORIES:
        return "temporal_window"
    return "current_frame"


def _question_view_requirement(question: Dict[str, Any]) -> str:
    if question.get("subcategory") in MULTIVIEW_REQUIRED_SUBCATEGORIES:
        return "all_views"
    return "anchored_view_with_multiview_context"


def _redesign_questions_for_multiview_multiframe(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    redesigned: List[Dict[str, Any]] = []
    for q in questions:
        question = dict(q)
        question["input_paradigm"] = "multi_view_multi_frame"
        question["temporal_target"] = _question_temporal_target(question)
        question["view_requirement"] = _question_view_requirement(question)
        redesigned.append(question)
    return redesigned


# ---------------------------------------------------------------------------
# Question generators – Spatial
# ---------------------------------------------------------------------------

def gen_object_type_choice(spatial: Dict) -> List[Dict]:
    """What type of object is closest / at position X?"""
    questions = []
    relations = spatial.get("object_relations", [])
    if not relations:
        return questions
    image_paths = _image_path_lookup(spatial)

    sorted_rels = sorted(relations, key=lambda r: r.get("geometric_relations", {}).get("euclidean_distance_m", 1e9))

    if sorted_rels:
        closest = sorted_rels[0]
        correct = _display_category(closest.get("category", "unknown"))
        all_cats = list({_display_category(r.get("category", "unknown")) for r in relations})
        extra_distractors = ["pedestrian", "bicycle", "bus", "motorcycle", "traffic cone"]
        distractors = list(set(all_cats + extra_distractors) - {correct})
        choices, answer = _shuffle_choices(correct, distractors)
        anchor = _relation_anchor(
            closest,
            include_category=False,
            include_3d_context=True,
            image_path_lookup=image_paths,
        )
        questions.append({
            "question_id": _make_id(),
            "question_type": "choice",
            "category": "spatial",
            "subcategory": "closest_object_type",
            "question": f"{anchor}. What type of object is this closest fused object to the ego vehicle?",
            "choices": choices,
            "answer": answer,
            "answer_text": correct,
            **_relation_question_metadata(closest),
        })

    return questions


def gen_object_position_choice(spatial: Dict) -> List[Dict]:
    """Where is <object> relative to the ego vehicle?"""
    questions = []
    relations = spatial.get("object_relations", [])
    if not relations:
        return questions
    image_paths = _image_path_lookup(spatial)

    for rel in relations[:5]:
        sem = rel.get("semantic_relations", {})
        front_rear = sem.get("front_rear", "")
        left_right = sem.get("left_right", "")

        if front_rear and left_right:
            correct_pos = f"{front_rear}/{left_right}"
        elif front_rear:
            correct_pos = front_rear
        else:
            continue

        correct_display = DIRECTION_DISPLAY.get(correct_pos.replace("/", "_"), correct_pos)

        all_positions = [
            "in front of", "to the front-left of", "to the front-right of",
            "behind", "to the rear-left of", "to the rear-right of",
            "to the left of", "to the right of",
        ]
        distractors = [p for p in all_positions if p != correct_display]

        choices, answer = _shuffle_choices(correct_display, distractors)
        anchor = _relation_anchor(
            rel,
            include_category=True,
            include_3d_context=False,
            image_path_lookup=image_paths,
        )
        questions.append({
            "question_id": _make_id(),
            "question_type": "choice",
            "category": "spatial",
            "subcategory": "object_relative_position",
            "question": f"For {anchor}, what is its 3D position relative to the ego vehicle?",
            "choices": choices,
            "answer": answer,
            "answer_text": correct_display,
            **_relation_question_metadata(rel),
        })

    return questions


def gen_object_distance_numerical(spatial: Dict) -> List[Dict]:
    """What is the distance to <object>?"""
    questions = []
    relations = spatial.get("object_relations", [])
    if not relations:
        return questions
    image_paths = _image_path_lookup(spatial)

    sorted_rels = sorted(relations, key=lambda r: r.get("geometric_relations", {}).get("euclidean_distance_m", 1e9))

    for rank, rel in enumerate(sorted_rels[:3]):
        geo = rel.get("geometric_relations", {})
        dist = geo.get("euclidean_distance_m")
        if dist is None:
            continue
        ordinal = ["closest", "second closest", "third closest"][rank]
        anchor = _relation_anchor(
            rel,
            include_category=True,
            include_3d_context=False,
            image_path_lookup=image_paths,
        )
        questions.append({
            "question_id": _make_id(),
            "question_type": "numerical",
            "category": "spatial",
            "subcategory": "object_distance",
            "question": f"For the {ordinal} fused object, {anchor}, what is its Euclidean distance from the ego vehicle in meters?",
            "answer": _round(dist, 1),
            "answer_text": f"{_round(dist, 1)} m",
            "tolerance": 1.0,
            **_relation_question_metadata(rel),
        })

    return questions


def gen_object_speed_numerical(spatial: Dict) -> List[Dict]:
    """What is the speed of <object>?"""
    questions = []
    relations = spatial.get("object_relations", [])
    if not relations:
        return questions
    image_paths = _image_path_lookup(spatial)

    for rel in relations:
        vel = rel.get("velocity_3d_ego", {})
        speed = _speed_mps(vel)
        if speed < 0.5:
            continue
        anchor = _relation_anchor(
            rel,
            include_category=True,
            include_3d_context=True,
            image_path_lookup=image_paths,
        )
        questions.append({
            "question_id": _make_id(),
            "question_type": "numerical",
            "category": "spatial",
            "subcategory": "object_speed",
            "question": f"For {anchor}, what is this object's speed in m/s?",
            "answer": _round(speed, 1),
            "answer_text": f"{_round(speed, 1)} m/s",
            "tolerance": 1.0,
            **_relation_question_metadata(rel),
        })
        if len(questions) >= 3:
            break

    return questions


def gen_motion_relation_choice(spatial: Dict) -> List[Dict]:
    """What is the motion relationship between ego and <object>?"""
    questions = []
    relations = spatial.get("object_relations", [])
    if not relations:
        return questions
    image_paths = _image_path_lookup(spatial)

    motion_options = [
        "stable_relative_distance", "closing", "separating",
        "overtaking_ego", "ego_overtaking",
    ]

    for rel in relations[:4]:
        sem = rel.get("semantic_relations", {})
        motion = sem.get("motion_relation")
        if not motion:
            continue
        correct = motion.replace("_", " ")
        distractors = [m.replace("_", " ") for m in motion_options if m != motion]
        choices, answer = _shuffle_choices(correct, distractors)
        anchor = _relation_anchor(
            rel,
            include_category=True,
            include_3d_context=True,
            image_path_lookup=image_paths,
        )
        questions.append({
            "question_id": _make_id(),
            "question_type": "choice",
            "category": "spatial",
            "subcategory": "motion_relation",
            "question": f"For {anchor}, what is the motion relationship between this object and the ego vehicle?",
            "choices": choices,
            "answer": answer,
            "answer_text": correct,
            **_relation_question_metadata(rel),
        })

    return questions


def gen_conflict_prediction_choice(spatial: Dict) -> List[Dict]:
    """Is there a predicted future conflict with <object>?"""
    questions = []
    relations = spatial.get("object_relations", [])
    image_paths = _image_path_lookup(spatial)

    conflict_objects = [
        r for r in relations
        if (r.get("future_conflict", {}) or {}).get("conflict_with_ego")
    ]
    no_conflict = [
        r for r in relations
        if not (r.get("future_conflict", {}) or {}).get("conflict_with_ego")
    ]
    conflict_objects = sorted(
        conflict_objects,
        key=lambda r: (r.get("geometric_relations", {}) or {}).get("euclidean_distance_m", 1e9),
    )
    no_conflict = sorted(
        no_conflict,
        key=lambda r: (r.get("geometric_relations", {}) or {}).get("euclidean_distance_m", 1e9),
    )

    # Keep a stronger representation of non-conflict objects.
    for rel in (conflict_objects[:2] + no_conflict[:6]):
        fc = rel.get("future_conflict", {})
        has_conflict = fc.get("conflict_with_ego", False)

        correct = "Yes, conflict predicted" if has_conflict else "No conflict predicted"
        distractors = ["No conflict predicted", "Yes, conflict predicted"]
        if has_conflict:
            ctype = fc.get("conflict_type", "path crossing")
            distractors += [f"Yes, {ctype}"]
        choices, answer = _shuffle_choices(correct, distractors, num_choices=2)
        anchor = _relation_anchor(
            rel,
            include_category=True,
            include_3d_context=True,
            image_path_lookup=image_paths,
        )
        questions.append({
            "question_id": _make_id(),
            "question_type": "choice",
            "category": "spatial",
            "subcategory": "conflict_prediction",
            "question": f"For {anchor}, is there a predicted future conflict with the ego vehicle?",
            "choices": choices,
            "answer": answer,
            "answer_text": correct,
            **_relation_question_metadata(rel),
        })

    return questions


def _future_horizon_seconds(traj: List[List[float]], frame_rate_hz: Optional[float]) -> Optional[float]:
    if not traj or not frame_rate_hz or frame_rate_hz <= 0:
        return None
    return round(len(traj) / float(frame_rate_hz), 1)


def _future_interval_samples(
    traj: List[List[float]],
    frame_rate_hz: Optional[float],
    interval_s: float = 0.5,
) -> List[Tuple[float, List[float]]]:
    if not traj or not frame_rate_hz or frame_rate_hz <= 0 or interval_s <= 0:
        return []
    step_frames = max(1, int(round(float(frame_rate_hz) * interval_s)))
    samples: List[Tuple[float, List[float]]] = []
    sample_idx = step_frames - 1
    while sample_idx < len(traj):
        time_s = round((sample_idx + 1) / float(frame_rate_hz), 1)
        samples.append((time_s, traj[sample_idx]))
        sample_idx += step_frames
    if samples:
        last_time = round(len(traj) / float(frame_rate_hz), 1)
        if abs(samples[-1][0] - last_time) > 1e-6:
            samples.append((last_time, traj[-1]))
    return samples


def _future_motion_label(rel: Dict[str, Any]) -> Optional[str]:
    desc = str(rel.get("future_movement_description", "") or "").strip().lower()
    if desc:
        first_clause = desc.split(";")[0].strip()
        if "approaching ego" in first_clause:
            return "approaching ego"
        if "moving away from ego" in first_clause:
            return "moving away from ego"
        if "maintaining distance" in first_clause:
            return "maintaining relative distance"

    pos = rel.get("position_3d_ego", {}) or {}
    traj = rel.get("future_trajectory_3d_ego") or []
    if not traj:
        return None

    current_dist = math.sqrt(float(pos.get("x", 0.0)) ** 2 + float(pos.get("y", 0.0)) ** 2)
    last = traj[-1]
    future_dist = math.sqrt(float(last[0]) ** 2 + float(last[1]) ** 2)
    delta = future_dist - current_dist
    if delta <= -2.0:
        return "approaching ego"
    if delta >= 2.0:
        return "moving away from ego"
    return "maintaining relative distance"


def gen_future_motion_prediction_questions(
    spatial: Dict,
    frame_rate_hz: Optional[float] = None,
) -> List[Dict]:
    """Ask future motion prediction questions from structured future trajectories."""
    questions = []
    relations = spatial.get("object_relations", [])
    if not relations:
        return questions
    image_paths = _image_path_lookup(spatial)

    candidate_relations = []
    for rel in relations:
        traj = rel.get("future_trajectory_3d_ego") or []
        if not traj:
            continue
        speed = _speed_mps(rel.get("velocity_3d_ego", {}) or {})
        if speed < 0.5:
            continue
        candidate_relations.append(rel)

    candidate_relations = sorted(
        candidate_relations,
        key=lambda r: r.get("geometric_relations", {}).get("euclidean_distance_m", 1e9),
    )

    primary_relations = candidate_relations[:4]
    path_yes_relations = [
        r for r in candidate_relations
        if bool((r.get("future_conflict", {}) or {}).get("path_intersection_bev", False))
    ]
    path_no_relations = [
        r for r in candidate_relations
        if not bool((r.get("future_conflict", {}) or {}).get("path_intersection_bev", False))
    ]
    path_intersection_relations = path_yes_relations[:2] + path_no_relations[:6]

    for rel in primary_relations:
        traj = rel.get("future_trajectory_3d_ego") or []
        if not traj:
            continue
        anchor_with_3d_context = _relation_anchor(
            rel,
            include_category=True,
            include_3d_context=True,
            image_path_lookup=image_paths,
        )
        anchor_without_3d_context = _relation_anchor(
            rel,
            include_category=True,
            include_3d_context=False,
            image_path_lookup=image_paths,
        )
        horizon_s = _future_horizon_seconds(traj, frame_rate_hz)
        horizon_text = f" at the end of the future horizon (~{horizon_s}s)" if horizon_s is not None else " at the end of the future horizon"

        motion_label = _future_motion_label(rel)
        if motion_label:
            choices, answer = _shuffle_choices(
                motion_label,
                [
                    "approaching ego",
                    "moving away from ego",
                    "maintaining relative distance",
                    "stopped",
                ],
            )
            questions.append({
                "question_id": _make_id(),
                "question_type": "choice",
                "category": "spatial",
                "subcategory": "future_motion_label",
                "question": f"For {anchor_with_3d_context}, what is the most likely future motion trend relative to the ego vehicle?",
                "choices": choices,
                "answer": answer,
                "answer_text": motion_label,
                "prediction_horizon_s": horizon_s,
                **_relation_question_metadata(rel),
            })

        fc = rel.get("future_conflict", {}) or {}

        trajectory_samples = []
        for prediction_time_s, point in _future_interval_samples(traj, frame_rate_hz, interval_s=0.5):
            if len(point) < 2:
                continue
            trajectory_samples.append(
                [
                    round(float(prediction_time_s), 1),
                    _round(float(point[0]), 1),
                    _round(float(point[1]), 1),
                ]
            )
        if trajectory_samples:
            questions.append({
                "question_id": _make_id(),
                "question_type": "numerical",
                "category": "spatial",
                "subcategory": "future_xy_trajectory",
                "question": (
                    f"For {anchor_without_3d_context}, what is the predicted future ego-frame trajectory sampled every 0.5s "
                    f"from t=+0.5s to t=+{trajectory_samples[-1][0]:.1f}s? "
                    f"Output the sequence as [[t, x, y], ...] in meters."
                ),
                "answer": trajectory_samples,
                "answer_text": json.dumps(trajectory_samples),
                "answer_format": "txy_sequence_m",
                "tolerance": 1.0,
                "prediction_interval_s": 0.5,
                "prediction_horizon_s": trajectory_samples[-1][0],
                **_relation_question_metadata(rel),
            })

        ttc = fc.get("ttc_s")
        if ttc is None:
            ttc = fc.get("time_to_conflict_s")
        if ttc is not None:
            try:
                ttc_val = round(float(ttc), 1)
            except (TypeError, ValueError):
                ttc_val = None
            if ttc_val is not None:
                questions.append({
                    "question_id": _make_id(),
                    "question_type": "numerical",
                    "category": "spatial",
                    "subcategory": "future_time_to_conflict_s",
                    "question": f"For {anchor_with_3d_context}, what is the predicted time-to-conflict with the ego vehicle in seconds?",
                    "answer": ttc_val,
                    "answer_text": f"{ttc_val} s",
                    "tolerance": 0.5,
                    **_relation_question_metadata(rel),
                })

    for rel in path_intersection_relations:
        traj = rel.get("future_trajectory_3d_ego") or []
        if not traj:
            continue
        fc = rel.get("future_conflict", {}) or {}
        path_intersection = bool(fc.get("path_intersection_bev", False))
        correct = "Yes, the paths will intersect" if path_intersection else "No, the paths will not intersect"
        choices, answer = _shuffle_choices(
            correct,
            ["Yes, the paths will intersect", "No, the paths will not intersect"],
            num_choices=2,
        )
        anchor_with_3d_context = _relation_anchor(
            rel,
            include_category=True,
            include_3d_context=True,
            image_path_lookup=image_paths,
        )
        horizon_s = _future_horizon_seconds(traj, frame_rate_hz)
        horizon_text = (
            f" at the end of the future horizon (~{horizon_s}s)"
            if horizon_s is not None
            else " at the end of the future horizon"
        )
        questions.append({
            "question_id": _make_id(),
            "question_type": "choice",
            "category": "spatial",
            "subcategory": "future_path_intersection",
            "question": f"For {anchor_with_3d_context}, will this object's future path intersect the ego vehicle's path{horizon_text}?",
            "choices": choices,
            "answer": answer,
            "answer_text": correct,
            "prediction_horizon_s": horizon_s,
            **_relation_question_metadata(rel),
        })

    return questions


def gen_object_position_numerical(spatial: Dict) -> List[Dict]:
    """Ask for the ego-frame 3D coordinate pair [x, y] of an object."""
    questions = []
    relations = spatial.get("object_relations", [])
    if not relations:
        return questions
    image_paths = _image_path_lookup(spatial)

    sorted_rels = sorted(relations, key=lambda r: r.get("geometric_relations", {}).get("euclidean_distance_m", 1e9))

    for rel in sorted_rels[:3]:
        pos = rel.get("position_3d_ego", {})
        x, y = pos.get("x"), pos.get("y")
        if x is None or y is None:
            continue
        anchor = _relation_anchor(rel, include_category=True, image_path_lookup=image_paths)
        questions.append({
            "question_id": _make_id(),
            "question_type": "numerical",
            "category": "spatial",
            "subcategory": "object_xy_position",
            "question": f"For {anchor}, what are the ego-frame 3D coordinates [x, y] in meters?",
            "answer": [_round(x, 1), _round(y, 1)],
            "answer_text": f"[{_round(x, 1)}, {_round(y, 1)}] m",
            "answer_format": "xy_pair_m",
            "tolerance": 1.0,
            **_relation_question_metadata(rel),
        })

    return questions


def gen_bbox_center_numerical(spatial: Dict) -> List[Dict]:
    """Ask for normalized 2D center coordinates on a 0-1000 scale."""
    questions = []
    observations = _collect_grounded_observations(spatial)
    for obs in observations[:4]:
        normalized = _normalized_bbox_coordinates(obs["bbox_2d"], obs.get("image_path", ""))
        if normalized is None:
            continue
        anchor = _observation_reference_for_2d_target(obs, include_3d_context=True)
        questions.append({
            "question_id": _make_id(),
            "question_type": "numerical",
            "category": "spatial",
            "subcategory": "bbox_center_coordinates_1000",
            "question": (
                f"For {anchor}, what are the normalized 2D bounding-box center coordinates [cx, cy] "
                "on a 0-1000 integer scale?"
            ),
            "answer": [int(normalized["cx"]), int(normalized["cy"])],
            "answer_text": f"[{int(normalized['cx'])}, {int(normalized['cy'])}]",
            "answer_format": "xy_pair_1000",
            "tolerance": 8,
            **_observation_question_metadata(obs),
        })
    return questions


def gen_normalized_bbox_coordinate_numerical(spatial: Dict) -> List[Dict]:
    """Ask for normalized 2D coordinate tuples on a 0-1000 integer scale."""
    questions = []
    observations = _collect_grounded_observations(spatial)
    for obs in observations[:4]:
        normalized = _normalized_bbox_coordinates(obs["bbox_2d"], obs.get("image_path", ""))
        if normalized is None:
            continue
        anchor = _observation_reference_for_2d_target(obs, include_3d_context=True)
        questions.append({
            "question_id": _make_id(),
            "question_type": "numerical",
            "category": "spatial",
            "subcategory": "normalized_bbox_coordinates_1000",
            "question": f"For {anchor}, what are the normalized 2D bounding-box coordinates [x1, y1, x2, y2] on a 0-1000 integer scale?",
            "answer": [
                int(normalized["x1"]),
                int(normalized["y1"]),
                int(normalized["x2"]),
                int(normalized["y2"]),
            ],
            "answer_text": (
                f"[{int(normalized['x1'])}, {int(normalized['y1'])}, "
                f"{int(normalized['x2'])}, {int(normalized['y2'])}]"
            ),
            "answer_format": "xyxy_1000",
            "tolerance": 8,
            **_observation_question_metadata(obs),
        })
    return questions


def gen_2d_to_3d_distance_choice(spatial: Dict) -> List[Dict]:
    """Ask for approximate 3D distance given a 2D box in a named camera."""
    questions = []
    observations = _collect_grounded_observations(spatial)
    all_distances = [
        _round(obs.get("geometric_relations", {}).get("euclidean_distance_m", 0.0))
        for obs in observations
        if obs.get("geometric_relations", {}).get("euclidean_distance_m") is not None
    ]
    for obs in observations[:5]:
        dist = obs.get("geometric_relations", {}).get("euclidean_distance_m")
        if dist is None:
            continue
        correct = f"{_round(dist)} m"
        candidate_distances = {f"{d} m" for d in all_distances if d != _round(dist)}
        if len(candidate_distances) < 3:
            for delta in (8.0, 12.0, -6.0, -10.0):
                alt = max(0.0, _round(dist + delta))
                candidate_distances.add(f"{alt} m")
        choices, answer = _shuffle_choices(correct, list(candidate_distances))
        anchor = _observation_anchor(obs, include_category=True)
        questions.append({
            "question_id": _make_id(),
            "question_type": "choice",
            "category": "spatial",
            "subcategory": "distance_from_2d_bbox",
            "question": f"For {anchor}, what is the object's approximate Euclidean distance from the ego vehicle?",
            "choices": choices,
            "answer": answer,
            "answer_text": correct,
            **_observation_question_metadata(obs),
        })
    return questions


def gen_2d_to_3d_position_choice(spatial: Dict) -> List[Dict]:
    """Ask for ego-frame position from a specific 2D camera observation."""
    questions = []
    observations = _collect_grounded_observations(spatial)
    position_options = [
        "in front of",
        "to the front-left of",
        "to the front-right of",
        "behind",
        "to the rear-left of",
        "to the rear-right of",
        "to the left of",
        "to the right of",
    ]
    for obs in observations[:5]:
        sem = obs.get("semantic_relations", {})
        front_rear = sem.get("front_rear", "")
        left_right = sem.get("left_right", "")
        if front_rear and left_right:
            key = f"{front_rear}_{left_right}"
        else:
            key = front_rear
        correct = DIRECTION_DISPLAY.get(key)
        if not correct:
            continue
        choices, answer = _shuffle_choices(correct, [p for p in position_options if p != correct])
        anchor = _observation_anchor(obs, include_category=True)
        questions.append({
            "question_id": _make_id(),
            "question_type": "choice",
            "category": "spatial",
            "subcategory": "ego_position_from_2d_bbox",
            "question": f"For {anchor}, what is the object's 3D position relative to the ego vehicle?",
            "choices": choices,
            "answer": answer,
            "answer_text": correct,
            **_observation_question_metadata(obs),
        })
    return questions


def gen_image_region_choice(spatial: Dict) -> List[Dict]:
    """Ask where a 3D-grounded object lands in the image plane."""
    questions = []
    observations = _collect_grounded_observations(spatial)
    region_options = [
        "upper left", "upper center", "upper right",
        "middle left", "middle center", "middle right",
        "lower left", "lower center", "lower right",
    ]
    for obs in observations[:5]:
        region = _image_region_name(obs["bbox_2d"], obs.get("image_path", ""))
        if region is None:
            continue
        choices, answer = _shuffle_choices(region, [r for r in region_options if r != region])
        anchor = _observation_reference_for_2d_target(obs, include_3d_context=True)
        questions.append({
            "question_id": _make_id(),
            "question_type": "choice",
            "category": "spatial",
            "subcategory": "image_region_from_3d_object",
            "question": f"For {anchor}, which image region does this object occupy?",
            "choices": choices,
            "answer": answer,
            "answer_text": region,
            **_observation_question_metadata(obs),
        })
    return questions


def gen_camera_view_for_object_choice(spatial: Dict) -> List[Dict]:
    """Ask which camera contains a specific 3D-grounded object."""
    questions = []
    observations = _collect_grounded_observations(spatial)
    used_tokens = set()
    for obs in observations:
        token = obs.get("track_token")
        if token in used_tokens:
            continue
        used_tokens.add(token)
        correct = obs["camera"]
        choices, answer = _shuffle_choices(correct, [cam for cam in CAMERAS if cam != correct])
        anchor = _observation_reference_for_camera_view_target(obs, include_3d_context=True)
        questions.append({
            "question_id": _make_id(),
            "question_type": "choice",
            "category": "spatial",
            "subcategory": "camera_view_for_3d_object",
            "question": f"For {anchor}, which camera view contains this object?",
            "choices": choices,
            "answer": answer,
            "answer_text": correct,
            **_observation_question_metadata(obs),
        })
        if len(questions) >= 4:
            break
    return questions


def gen_multiview_camera_pair_choice(spatial: Dict) -> List[Dict]:
    """Ask which pair of cameras jointly observes the same 3D object."""
    questions = []
    correspondence = spatial.get("cross_view_correspondence", {})
    relation_index = _relation_index_by_track_token(spatial)
    pair_options = set()
    for value in correspondence.values():
        views = sorted(value.get("views", []))
        if len(views) >= 2:
            pair_options.add(" + ".join(views[:2]))
    for value in correspondence.values():
        views = sorted(value.get("views", []))
        if len(views) < 2:
            continue
        correct = " + ".join(views[:2])
        distractors = [p for p in pair_options if p != correct]
        if len(distractors) < 3:
            distractors.extend([
                "front + front_left",
                "front + front_right",
                "back + back_left",
                "back + back_right",
            ])
        choices, answer = _shuffle_choices(correct, distractors)
        reference = _multiview_reference(
            value,
            relation_index,
            include_3d_context=True,
            include_view_text=False,
        )
        rel = relation_index.get(str(value.get("track_token", "")))
        questions.append({
            "question_id": _make_id(),
            "question_type": "choice",
            "category": "spatial",
            "subcategory": "multiview_camera_pair",
            "question": f"For {reference}, which camera pair observes this same object?",
            "choices": choices,
            "answer": answer,
            "answer_text": correct,
            "target_track_token": value.get("track_token"),
            "target_cameras": views[:2],
            "object_role": _object_role_from_relation(rel),
        })
        if len(questions) >= 3:
            break
    return questions


def gen_multiview_coordinate_consistency_numerical(spatial: Dict) -> List[Dict]:
    """Ask for the same object's 2D coordinates across multiple views."""
    questions = []
    correspondence = spatial.get("cross_view_correspondence", {}) or {}
    per_camera_results = spatial.get("per_camera_results", {}) or {}
    relation_index = _relation_index_by_track_token(spatial)

    for value in correspondence.values():
        views = sorted(value.get("views", []))
        observations = value.get("per_view_observations", []) or []
        if len(views) < 2 or len(observations) < 2:
            continue

        selected_views = views[:2]
        selected_obs = []
        for view_name in selected_views:
            match = next((obs for obs in observations if obs.get("camera") == view_name), None)
            if match is None:
                selected_obs = []
                break
            image_path = per_camera_results.get(view_name, {}).get("image_path", "")
            normalized = _normalized_bbox_coordinates(match.get("detection_bbox_2d"), image_path)
            if normalized is None:
                selected_obs = []
                break
            selected_obs.append(
                [
                    view_name,
                    int(normalized["cx"]),
                    int(normalized["cy"]),
                ]
            )
        if len(selected_obs) != 2:
            continue

        rel = relation_index.get(str(value.get("track_token", "")))
        reference = _multiview_reference(value, relation_index, include_3d_context=True)
        questions.append({
            "question_id": _make_id(),
            "question_type": "numerical",
            "category": "spatial",
            "subcategory": "multiview_center_coordinates_1000",
            "question": (
                f"For {reference}, what are the normalized 2D center coordinates [cx, cy] on each view? "
                f"Output as [[camera, cx, cy], ...] in the order [{selected_views[0]}, {selected_views[1]}]."
            ),
            "answer": selected_obs,
            "answer_text": json.dumps(selected_obs),
            "answer_format": "camera_cxy_sequence_1000",
            "tolerance": 8,
            "target_track_token": value.get("track_token"),
            "target_cameras": selected_views,
            "object_role": _object_role_from_relation(rel),
        })
        if len(questions) >= 4:
            break

    return questions


# ---------------------------------------------------------------------------
# Question generators – Driving
# ---------------------------------------------------------------------------

def _canonicalize_action_label(label: Any, options: List[str], aliases: Dict[str, str]) -> Optional[str]:
    """Map a free-form action label onto the canonical schema when possible."""
    if not isinstance(label, str):
        return None
    text = " ".join(label.strip().split())
    if not text:
        return None
    lowered = text.lower()
    for opt in options:
        if opt.lower() == lowered:
            return opt
    return aliases.get(lowered, text)


def _normalize_longitudinal(label: Any) -> Optional[str]:
    return _canonicalize_action_label(label, LONGITUDINAL_OPTIONS, LONGITUDINAL_ALIASES)


def _normalize_lateral(label: Any) -> Optional[str]:
    return _canonicalize_action_label(label, LATERAL_OPTIONS, LATERAL_ALIASES)


def _apply_action_pairing_constraint(
    longitudinal: Optional[str],
    lateral: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Remain stopped must always be paired with No lateral action."""
    if longitudinal == "Remain stopped":
        return longitudinal, "No lateral action"
    return longitudinal, lateral


def _format_action_pair(longitudinal: str, lateral: str) -> str:
    return f"{longitudinal} + {lateral}"


def _normalize_action_pair(longitudinal: Any, lateral: Any) -> Optional[Tuple[str, str]]:
    lon, lat = _apply_action_pairing_constraint(
        _normalize_longitudinal(longitudinal),
        _normalize_lateral(lateral),
    )
    if lon not in LONGITUDINAL_OPTIONS or lat not in LATERAL_OPTIONS:
        return None
    return lon, lat


@lru_cache(maxsize=1)
def _all_valid_action_pairs() -> Tuple[str, ...]:
    pairs: List[str] = []
    seen = set()
    for lon in LONGITUDINAL_OPTIONS:
        for lat in LATERAL_OPTIONS:
            normalized = _normalize_action_pair(lon, lat)
            if not normalized:
                continue
            text = _format_action_pair(*normalized)
            if text not in seen:
                seen.add(text)
                pairs.append(text)
    return tuple(pairs)


def _action_pair_distractors(longitudinal: str, lateral: str) -> List[str]:
    correct = _format_action_pair(longitudinal, lateral)
    return [pair for pair in _all_valid_action_pairs() if pair != correct]


def _action_pair_texts(actions: List[Any]) -> List[str]:
    texts: List[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        normalized = _normalize_action_pair(action.get("Longitudinal"), action.get("Lateral"))
        if normalized:
            texts.append(_format_action_pair(*normalized))
    return texts


def _driving_decision_fields(driving: Dict) -> Tuple[Optional[str], Optional[str]]:
    decision = _coerce_reasoning_section(driving.get("Driving decision"))
    normalized = _normalize_action_pair(decision.get("Longitudinal"), decision.get("Lateral"))
    if normalized:
        return normalized
    lon = _normalize_longitudinal(decision.get("Longitudinal"))
    lat = _normalize_lateral(decision.get("Lateral"))
    lon, lat = _apply_action_pairing_constraint(lon, lat)
    return (
        lon if lon in LONGITUDINAL_OPTIONS else None,
        lat if lat in LATERAL_OPTIONS else None,
    )


def gen_driving_decision_questions(driving: Dict) -> List[Dict]:
    """Joint / longitudinal / lateral driving-decision MCQs."""
    longitudinal, lateral = _driving_decision_fields(driving)
    questions: List[Dict] = []

    if longitudinal and lateral:
        correct = _format_action_pair(longitudinal, lateral)
        choices, answer = _shuffle_choices(correct, _action_pair_distractors(longitudinal, lateral))
        questions.append(_make_question(
            question_type="choice",
            category="driving",
            subcategory="driving_decision_joint",
            question=(
                "What is the best combined driving decision (longitudinal + lateral) "
                "for the ego vehicle in this scenario? "
            ),
            choices=choices,
            answer=answer,
            answer_text=correct,
            longitudinal=longitudinal,
            lateral=lateral,
            longitudinal_definition=LONGITUDINAL_DEFINITIONS.get(longitudinal, ""),
            lateral_definition=LATERAL_DEFINITIONS.get(lateral, ""),
        ))

    if longitudinal:
        choices, answer = _shuffle_choices(
            longitudinal,
            [opt for opt in LONGITUDINAL_OPTIONS if opt != longitudinal],
        )
        questions.append(_make_question(
            question_type="choice",
            category="driving",
            subcategory="driving_decision_longitudinal",
            question="What is the best longitudinal action for the ego vehicle in this scenario?",
            choices=choices,
            answer=answer,
            answer_text=longitudinal,
            longitudinal=longitudinal,
            longitudinal_definition=LONGITUDINAL_DEFINITIONS.get(longitudinal, ""),
        ))

    if lateral:
        choices, answer = _shuffle_choices(
            lateral,
            [opt for opt in LATERAL_OPTIONS if opt != lateral],
        )
        questions.append(_make_question(
            question_type="choice",
            category="driving",
            subcategory="driving_decision_lateral",
            question=(
                "What is the best lateral action for the ego vehicle in this scenario? "
                "If the longitudinal action is Remain stopped, choose No lateral action."
            ),
            choices=choices,
            answer=answer,
            answer_text=lateral,
            lateral=lateral,
            lateral_definition=LATERAL_DEFINITIONS.get(lateral, ""),
        ))

    return questions


def gen_driving_reasoning_trace(driving: Dict) -> List[Dict]:
    """Output the full driving reasoning trace."""
    trace = driving.get("Reasoning trace", "")
    if not isinstance(trace, str) or not trace:
        return []
    return [_make_question(
        question_type="text",
        category="driving",
        subcategory="driving_reasoning_trace",
        question="What is the complete driving reasoning trace for the ego decision in this frame?",
        answer=trace,
        answer_text=trace,
    )]


def gen_critical_vehicle_behavior(driving: Dict) -> List[Dict]:
    """What is the behavior of the critical vehicle?"""
    questions = []
    cc = _coerce_reasoning_section(driving.get("Critical components"))

    for comp_name, comp_data in cc.items():
        if not isinstance(comp_data, dict):
            continue
        if "vehicle" not in comp_name.lower():
            continue
        behavior = comp_data.get("Behavior", "")
        if not behavior:
            continue
        vtype = comp_data.get("Type", "vehicle")
        location = comp_data.get("Relative location", "")
        questions.append(_make_question(
            question_type="choice",
            category="driving",
            subcategory="critical_vehicle_behavior",
            question=f"What is the {vtype.lower()} ({location.lower()}) doing in this scene?",
            choices=None,
            answer=None,
            answer_text=behavior,
            _needs_gemini_choices=True,
        ))

    return questions


def gen_scene_description_choice(driving: Dict) -> List[Dict]:
    """Multiple-choice about the scene description."""
    desc = driving.get("Scene description", "")
    if not isinstance(desc, str) or not desc:
        return []
    return [_make_question(
        question_type="choice",
        category="driving",
        subcategory="scene_description",
        question="Which description best matches the current driving scene?",
        choices=None,
        answer=None,
        answer_text=desc,
        _needs_gemini_choices=True,
    )]


# ---------------------------------------------------------------------------
# Question generators – Counterfactual
# ---------------------------------------------------------------------------

def gen_unsafe_action_choice(counterfactual: Dict) -> List[Dict]:
    """Which action would be unsafe?"""
    unsafe = _action_pair_texts(_coerce_list(counterfactual.get("Top safety-critical actions")))
    if not unsafe:
        return []
    correct = unsafe[0]
    distractors = _dedupe_preserve(
        _action_pair_texts(_coerce_list(counterfactual.get("Alternative actions")))
        + unsafe[1:]
        + SAFE_ACTION_PAIR_FALLBACKS,
        exclude=correct,
    )
    choices, answer = _shuffle_choices(correct, distractors)
    return [_make_question(
        question_type="choice",
        category="counterfactual",
        subcategory="unsafe_action_identification",
        question="Which of the following actions would be UNSAFE in this scenario?",
        choices=choices,
        answer=answer,
        answer_text=correct,
    )]


def gen_risk_level_choice(counterfactual: Dict) -> List[Dict]:
    """What is the risk level of a specific action?"""
    questions = []
    all_actions = (
        _coerce_list(counterfactual.get("Alternative actions"))
        + _coerce_list(counterfactual.get("Top safety-critical actions"))
    )

    for action in all_actions[:3]:
        if not isinstance(action, dict):
            continue
        normalized = _normalize_action_pair(action.get("Longitudinal"), action.get("Lateral"))
        risk = action.get("Risk level", "")
        if not normalized or not risk:
            continue
        lon, lat = normalized
        choices, answer = _shuffle_choices(risk, ["Safe", "Unsafe", "Moderate risk", "Highly dangerous"])
        questions.append(_make_question(
            question_type="choice",
            category="counterfactual",
            subcategory="action_risk_assessment",
            question=f"What is the risk level of: '{lon}' (longitudinal) + '{lat}' (lateral)?",
            choices=choices,
            answer=answer,
            answer_text=risk,
            explanation=action.get("Reason", ""),
            longitudinal=lon,
            lateral=lat,
        ))

    return questions


# ---------------------------------------------------------------------------
# Gemini-based question enrichment (optional)
# ---------------------------------------------------------------------------

def enrich_with_gemini(
    questions: List[Dict],
    reasoning_data: Dict,
    api_key: Optional[str] = None,
) -> List[Dict]:
    """
    Use Gemini to generate plausible distractor choices for questions that
    were flagged with _needs_gemini_choices, and to rephrase questions for
    diversity.
    """
    needs_enrichment = [q for q in questions if q.get("_needs_gemini_choices")]
    if not needs_enrichment:
        return questions

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Warning: google-genai not installed. Skipping Gemini enrichment.")
        for q in needs_enrichment:
            q.pop("_needs_gemini_choices", None)
        return questions

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("Warning: GEMINI_API_KEY not set. Skipping Gemini enrichment.")
        for q in needs_enrichment:
            q.pop("_needs_gemini_choices", None)
        return questions

    client = genai.Client(api_key=key)

    driving = _coerce_reasoning_section(reasoning_data.get("Driving"))
    scene_desc = driving.get("Scene description", "")

    prompt_items = []
    for i, q in enumerate(needs_enrichment):
        prompt_items.append({
            "index": i,
            "question": q["question"],
            "correct_answer": q["answer_text"],
            "subcategory": q.get("subcategory", ""),
        })

    system_prompt = f"""You are generating VQA distractor choices for autonomous driving scenes.

Scene context: {scene_desc}

For each question below, generate 3 plausible but INCORRECT distractor answers.
Also provide a rephrased version of the question for diversity.

Return a JSON array where each element has:
- "index": the original index
- "rephrased_question": a natural rephrased version of the question
- "distractors": array of 3 plausible wrong answers (strings)

Questions:
{json.dumps(prompt_items, indent=2)}
"""

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=system_prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json",
            ),
        )
        raw_text = ""
        for cand in (response.candidates or []):
            for part in (cand.content.parts or []):
                if hasattr(part, "text") and part.text:
                    raw_text += part.text
        enrichments = json.loads(raw_text)
    except Exception as e:
        print(f"Warning: Gemini enrichment failed: {e}")
        for q in needs_enrichment:
            q.pop("_needs_gemini_choices", None)
        return questions

    enrich_map = {e["index"]: e for e in enrichments if isinstance(e, dict)}
    for i, q in enumerate(needs_enrichment):
        e = enrich_map.get(i, {})
        distractors = e.get("distractors", [])
        if distractors and len(distractors) >= 3:
            choices, answer = _shuffle_choices(q["answer_text"], distractors)
            q["choices"] = choices
            q["answer"] = answer
        rephrased = e.get("rephrased_question")
        if rephrased:
            q["question"] = rephrased
        q.pop("_needs_gemini_choices", None)

    return questions


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

SPATIAL_QUESTION_GENERATORS = (
    gen_object_type_choice,
    gen_object_position_choice,
    gen_object_distance_numerical,
    gen_object_speed_numerical,
    gen_motion_relation_choice,
    gen_conflict_prediction_choice,
    gen_object_position_numerical,
    gen_bbox_center_numerical,
    gen_normalized_bbox_coordinate_numerical,
    gen_2d_to_3d_distance_choice,
    gen_2d_to_3d_position_choice,
    gen_image_region_choice,
    gen_camera_view_for_object_choice,
    gen_multiview_camera_pair_choice,
    gen_multiview_coordinate_consistency_numerical,
)

DRIVING_QUESTION_GENERATORS = (
    gen_driving_decision_questions,
    gen_driving_reasoning_trace,
    gen_critical_vehicle_behavior,
    gen_scene_description_choice,
)

COUNTERFACTUAL_QUESTION_GENERATORS = (
    gen_unsafe_action_choice,
    gen_risk_level_choice,
)


def generate_vqa_for_frame(
    reasoning_data: Dict,
    frame_timestamp: str,
    image_paths: Optional[Dict[str, str]] = None,
    temporal_multiview_context: Optional[Dict[str, Any]] = None,
) -> Dict:
    """Generate all VQA questions for a single frame's reasoning data."""
    spatial = _inject_resolved_image_paths(
        _coerce_reasoning_section(reasoning_data.get("Spatial")),
        image_paths,
    )
    driving = _coerce_reasoning_section(reasoning_data.get("Driving"))
    counterfactual = _coerce_reasoning_section(reasoning_data.get("Counterfactual"))
    frame_rate_hz = None
    if isinstance(temporal_multiview_context, dict):
        frame_rate_hz = temporal_multiview_context.get("frame_rate_hz")

    all_questions: List[Dict] = []
    for generator in SPATIAL_QUESTION_GENERATORS:
        all_questions.extend(generator(spatial))
    all_questions.extend(gen_future_motion_prediction_questions(spatial, frame_rate_hz=frame_rate_hz))
    for generator in DRIVING_QUESTION_GENERATORS:
        all_questions.extend(generator(driving))
    for generator in COUNTERFACTUAL_QUESTION_GENERATORS:
        all_questions.extend(generator(counterfactual))

    cameras = dict(image_paths or {})
    if not cameras:
        pcr = _coerce_reasoning_section(spatial.get("per_camera_results"))
        for cam, cam_data in pcr.items():
            img = _coerce_reasoning_section(cam_data).get("image_path", "")
            if img:
                cameras[cam] = img

    return {
        "frame_timestamp": frame_timestamp,
        "image_paths": cameras,
        "temporal_multiview_context": temporal_multiview_context,
        "num_questions": len(all_questions),
        "questions": all_questions,
    }


def process_reasoning_file(
    reasoning_path: str,
    image_paths: Optional[Dict[str, str]] = None,
    temporal_multiview_context: Optional[Dict[str, Any]] = None,
    use_gemini: bool = False,
    api_key: Optional[str] = None,
) -> Optional[Dict]:
    """Process a single reasoning JSON file into VQA format."""
    if not os.path.isfile(reasoning_path):
        return None

    with open(reasoning_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    timestamp = os.path.splitext(os.path.basename(reasoning_path))[0]
    result = generate_vqa_for_frame(
        data,
        timestamp,
        image_paths=image_paths,
        temporal_multiview_context=temporal_multiview_context,
    )

    if use_gemini:
        result["questions"] = enrich_with_gemini(result["questions"], data, api_key)

    result["questions"] = [q for q in result["questions"] if not q.get("_needs_gemini_choices")]
    result["questions"] = _redesign_questions_for_multiview_multiframe(result["questions"])
    result["num_questions"] = len(result["questions"])

    return result


def _key_frame_timestamp(metadata: Dict[str, Any], key_frame_index: int) -> Optional[int]:
    frames = metadata.get("frames", []) or []
    if key_frame_index < 0 or key_frame_index >= len(frames):
        return None
    return int(frames[key_frame_index].get("timestamp_us", 0))


def process_clip(
    clip_dir: str,
    output_dir: str,
    key_frame_index: Optional[int] = None,
    history_frames: int = 3,
    use_gemini: bool = False,
    api_key: Optional[str] = None,
) -> str:
    """Process reasoning files in a clip directory.

    If ``key_frame_index`` is set, only that metadata frame is converted.
    If ``key_frame_index`` is None, all reasoning JSON frames are converted.
    """
    reasoning_dir = os.path.join(clip_dir, "reasoning")
    if not os.path.isdir(reasoning_dir):
        print(f"Warning: No reasoning directory in {clip_dir}")
        return ""

    clip_name = os.path.basename(clip_dir)
    clip_output_dir = os.path.join(output_dir, clip_name)
    os.makedirs(clip_output_dir, exist_ok=True)

    metadata_path = os.path.join(clip_dir, "metadata.json")
    metadata: Dict[str, Any] = {}
    frame_index_by_timestamp: Dict[int, int] = {}
    frame_stride = 1
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        for idx, frame in enumerate(metadata.get("frames", []) or []):
            ts = int(frame.get("timestamp_us", 0))
            frame_index_by_timestamp[ts] = idx
        frame_rate_hz = metadata.get("frame_rate_hz", 1)
        try:
            frame_stride = max(1, int(round(float(frame_rate_hz))))
        except (TypeError, ValueError):
            frame_stride = 1

    reasoning_files: List[str]
    if key_frame_index is None:
        reasoning_files = sorted(
            fname for fname in os.listdir(reasoning_dir) if fname.endswith(".json")
        )
        if not reasoning_files:
            print(f"Warning: No reasoning JSON files in {clip_name}")
            return ""
    else:
        if not metadata:
            print(f"Warning: metadata.json required to resolve keyframe for {clip_name}")
            return ""
        key_timestamp = _key_frame_timestamp(metadata, key_frame_index)
        if key_timestamp is None:
            print(
                f"Warning: Keyframe index {key_frame_index} out of range for {clip_name} "
                f"({len(metadata.get('frames', []) or [])} frames)"
            )
            return ""
        reasoning_name = f"{key_timestamp}.json"
        if not os.path.isfile(os.path.join(reasoning_dir, reasoning_name)):
            print(
                f"Warning: Missing keyframe reasoning file for {clip_name}: {reasoning_name} "
                f"(key_frame={key_frame_index})"
            )
            return ""
        reasoning_files = [reasoning_name]

    all_frames_vqa = []
    for fname in reasoning_files:
        rpath = os.path.join(reasoning_dir, fname)
        print(f"  Processing {fname}...")
        stem = os.path.splitext(fname)[0]
        image_paths = None
        temporal_context = None
        target_idx = None
        if metadata:
            try:
                timestamp = int(stem)
            except ValueError:
                timestamp = None
            if timestamp is not None:
                target_idx = frame_index_by_timestamp.get(timestamp)
            if target_idx is not None:
                temporal_context = _build_temporal_multiview_context(
                    metadata,
                    target_idx,
                    clip_dir,
                    history_frames=history_frames,
                    stride_frames=frame_stride,
                )
                image_paths = temporal_context.get("current_frame_image_paths")

        frame_vqa = process_reasoning_file(
            rpath,
            image_paths=image_paths,
            temporal_multiview_context=temporal_context,
            use_gemini=use_gemini,
            api_key=api_key,
        )
        if not frame_vqa:
            continue
        if target_idx is not None:
            frame_vqa["key_frame_index"] = target_idx
        all_frames_vqa.append(frame_vqa)

    for fvqa in all_frames_vqa:
        ts = fvqa["frame_timestamp"]
        out_path = os.path.join(clip_output_dir, f"{ts}_vqa.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fvqa, f, indent=2)

    print(
        f"  Clip {clip_name}: {len(all_frames_vqa)} frames, "
        f"{sum(f['num_questions'] for f in all_frames_vqa)} questions total"
    )
    print(f"  Output: {clip_output_dir}/")
    return clip_output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate VQA from nureasoning structured data")
    parser.add_argument("--data-root", default="./dataset/data/train",
                        help="Root directory containing clip subdirectories "
                             "(recurses into part_* folders)")
    parser.add_argument("--clip", default=None,
                        help="Process a single clip directory (relative to data-root or absolute)")
    parser.add_argument("--output", default="./vqa_output",
                        help="Output directory for VQA files")
    parser.add_argument(
        "--key-frame",
        type=int,
        default=None,
        help=(
            "Optional keyframe index to generate (e.g. 100 ≈ 10s at 10Hz). "
            "If omitted, generate VQA for all reasoning JSON frames."
        ),
    )
    parser.add_argument("--history-frames", type=int, default=3,
                        help="Number of 1Hz history steps before the current frame "
                             "(default: 3 for [-3,-2,-1,0])")
    parser.add_argument("--gemini-rephrase", action="store_true",
                        help="Use Gemini API to generate distractors and rephrase questions")
    parser.add_argument("--api-key", default=None,
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible choice shuffling")
    args = parser.parse_args()

    random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)

    process_kwargs = {
        "key_frame_index": args.key_frame,
        "history_frames": args.history_frames,
        "use_gemini": args.gemini_rephrase,
        "api_key": args.api_key,
    }

    if args.clip:
        clip_path = args.clip if os.path.isabs(args.clip) else os.path.join(args.data_root, args.clip)
        if not os.path.isdir(clip_path):
            print(f"Error: Clip directory not found: {clip_path}")
            sys.exit(1)
        print(f"Processing single clip: {clip_path}")
        process_clip(clip_path, args.output, **process_kwargs)
    else:
        from nureasoning.common.clips import discover_clips

        if not os.path.isdir(args.data_root):
            print(f"Error: Data root not found: {args.data_root}")
            sys.exit(1)
        clip_dirs = [
            clip_dir for clip_dir in discover_clips(args.data_root)
            if os.path.isdir(os.path.join(clip_dir, "reasoning"))
        ]
        for clip_dir in clip_dirs:
            print(f"\nProcessing clip: {os.path.basename(clip_dir)}")
            process_clip(clip_dir, args.output, **process_kwargs)

    print("\nDone.")


if __name__ == "__main__":
    main()
