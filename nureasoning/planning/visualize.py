from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.patches as mpatches
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

from nureasoning.planning.benchmark import (
    KEY_FRAME_INDEX,
    BenchmarkConfig,
    BenchmarkSummary,
    ClipResult,
    _load_pickle,
    discover_clips,
    evaluate_clip,
    load_future_object_states,
    select_key_frame_idx,
)
from nureasoning.common.schema import Annotations, EgoState, nuReasoningStaticMap
from nureasoning.visualization.plotting import (
    ACTOR_STYLES,
    TRAJECTORY_STYLES,
    _viz_add_trajectory,
    _viz_classify_actor,
    _viz_configure_bev_ax,
    _viz_draw_ego,
    _viz_draw_objects,
    _viz_draw_static_map,
)
from nureasoning.nuvla.trajectory_provider import (
    VLATrajectoryProvider,
    add_planning_prompt_arguments,
    load_vlm_observation,
)


logger = logging.getLogger(__name__)

CAMERA_ORDER = [
    "front",
    "front_left",
    "front_right",
    "left",
    "right",
    "back",
    "back_left",
    "back_right",
]

CAMERA_KEY_TO_NAME = {
    "front": "CAM_M_F",
    "front_left": "CAM_M_L0",
    "front_right": "CAM_M_R0",
    "left": "CAM_M_L1",
    "right": "CAM_M_R1",
    "back": "CAM_M_B",
    "back_left": "CAM_M_L2",
    "back_right": "CAM_M_R2",
}


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_metadata(clip_path: str) -> Dict[str, Any]:
    with open(os.path.join(clip_path, "metadata.json"), "r") as f:
        return json.load(f)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        return np.eye(3, dtype=np.float64)
    q /= norm
    qw, qx, qy, qz = q.tolist()
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )


def _yaw_to_rotmat(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _ego_pose_matrix(ego_state: EgoState) -> Tuple[np.ndarray, np.ndarray]:
    pose = ego_state.pose or {}
    t_ego = np.array(
        [
            _safe_float(pose.get("x")),
            _safe_float(pose.get("y")),
            _safe_float(pose.get("z")),
        ],
        dtype=np.float64,
    )
    if all(k in pose for k in ("qw", "qx", "qy", "qz")):
        R_ego = _quat_to_rotmat(
            _safe_float(pose.get("qw"), 1.0),
            _safe_float(pose.get("qx")),
            _safe_float(pose.get("qy")),
            _safe_float(pose.get("qz")),
        )
    else:
        R_ego = _yaw_to_rotmat(_safe_float(pose.get("yaw")))
    return R_ego, t_ego


def _get_camera_calibration(
    metadata: Dict[str, Any], camera_key: str
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]:
    calibs = metadata.get("camera_calibrations", {}) or {}
    calib = calibs.get(CAMERA_KEY_TO_NAME.get(camera_key, camera_key)) or calibs.get(camera_key)
    if not isinstance(calib, dict):
        return None
    try:
        intrinsic = np.asarray(calib["intrinsic"], dtype=np.float64)
        translation = np.asarray(calib["sensor2lidar_translation"], dtype=np.float64)
        rotation = calib["sensor2lidar_rotation"]
        R_cam_to_lidar = _quat_to_rotmat(
            _safe_float(rotation[0], 1.0),
            _safe_float(rotation[1]),
            _safe_float(rotation[2]),
            _safe_float(rotation[3]),
        )
        width = int(calib.get("width", 0))
        height = int(calib.get("height", 0))
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return intrinsic, R_cam_to_lidar, translation, width, height


def _project_points_global_to_image(
    points_global: np.ndarray,
    intrinsic: np.ndarray,
    R_cam_to_lidar: np.ndarray,
    t_cam_to_lidar: np.ndarray,
    R_lidar_to_global: np.ndarray,
    t_lidar_to_global: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if points_global.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)
    pts_lidar = (R_lidar_to_global.T @ (points_global - t_lidar_to_global.reshape(1, 3)).T).T
    pts_cam = (R_cam_to_lidar.T @ (pts_lidar - t_cam_to_lidar.reshape(1, 3)).T).T
    depth = pts_cam[:, 2]
    valid = depth > 0.5
    uv = np.full((pts_cam.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        uvw = (intrinsic @ pts_cam[valid].T).T
        uv[valid] = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)
    return uv, valid


def _coerce_spatial_reasoning(reasoning: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("Spatial", "spatial", "spatial_reasoning", "scene_spatial"):
        value = reasoning.get(key)
        if isinstance(value, dict):
            return value
    if isinstance(reasoning.get("per_camera_results"), dict):
        return reasoning
    return {}


def _load_spatial_reasoning(clip_path: str, frame: Dict[str, Any]) -> Dict[str, Any]:
    reasoning_rel = frame.get("reasoning")
    if not reasoning_rel:
        return {}
    reasoning_path = os.path.join(clip_path, reasoning_rel)
    if not os.path.isfile(reasoning_path):
        return {}
    return _coerce_spatial_reasoning(_load_json(reasoning_path))


def _vector_speed_mps(velocity: Any) -> Optional[float]:
    if not isinstance(velocity, dict):
        return None
    vx = velocity.get("x", velocity.get("vx"))
    vy = velocity.get("y", velocity.get("vy"))
    vz = velocity.get("z", velocity.get("vz", 0.0))
    if vx is None or vy is None:
        return None
    return float(math.sqrt(_safe_float(vx) ** 2 + _safe_float(vy) ** 2 + _safe_float(vz) ** 2))


def _draw_surrounding_object_future_trajectories(
    ax: Any,
    current_annotations: Annotations,
    future_annotations: Sequence[Annotations],
    ego_pose: Dict[str, float],
    vis_range_m: float,
    *,
    trajectory_dt_s: float = 0.1,
    plot_dt_s: float = 0.5,
) -> None:
    """Draw unlabeled actor future trails so the BEV legend stays trajectory-only."""
    future_by_token: Dict[str, List[Tuple[float, float]]] = {}
    category_by_token: Dict[str, str] = {}

    for obj in current_annotations.objects:
        category = str(obj.category or "").lower()
        if category.startswith("other.") or category == "other":
            continue
        token = str(obj.track_token or "")
        if not token:
            continue
        future_by_token[token] = [
            (
                _safe_float(obj.pose.get("x")),
                _safe_float(obj.pose.get("y")),
            )
        ]
        category_by_token[token] = category

    for ann in future_annotations:
        objects = getattr(ann, "objects", []) or []
        obj_by_token = {str(obj.track_token or ""): obj for obj in objects}
        for token, points in future_by_token.items():
            obj = obj_by_token.get(token)
            if obj is None:
                continue
            points.append(
                (
                    _safe_float(obj.pose.get("x")),
                    _safe_float(obj.pose.get("y")),
                )
            )

    ego_x = _safe_float(ego_pose.get("x"))
    ego_y = _safe_float(ego_pose.get("y"))
    padded_range = float(vis_range_m) * 1.10
    for token, points in future_by_token.items():
        if len(points) < 2:
            continue
        pts = np.asarray(points, dtype=np.float64)
        if not np.all(np.isfinite(pts)):
            continue
        stride = max(1, int(round(float(plot_dt_s) / max(float(trajectory_dt_s), 1e-6))))
        sampled_pts = pts[::stride]
        if not np.array_equal(sampled_pts[-1], pts[-1]):
            sampled_pts = np.vstack([sampled_pts, pts[-1]])
        in_view = (
            (sampled_pts[:, 0] >= ego_x - padded_range)
            & (sampled_pts[:, 0] <= ego_x + padded_range)
            & (sampled_pts[:, 1] >= ego_y - padded_range)
            & (sampled_pts[:, 1] <= ego_y + padded_range)
        )
        if not np.any(in_view):
            continue

        actor_key = _viz_classify_actor(category_by_token.get(token, ""))
        actor_style = ACTOR_STYLES.get(actor_key, ACTOR_STYLES["generic"])
        trajectory_style = TRAJECTORY_STYLES["gt"]
        color = actor_style.fill_color or actor_style.line_color or trajectory_style.line_color
        ax.plot(
            sampled_pts[:, 0],
            sampled_pts[:, 1],
            color=color,
            alpha=trajectory_style.line_alpha,
            linewidth=trajectory_style.line_width,
            linestyle=trajectory_style.line_style,
            marker=trajectory_style.marker,
            markersize=trajectory_style.marker_size,
            markeredgecolor=trajectory_style.marker_edge_color,
            zorder=trajectory_style.zorder - 0.2,
        )


def _relation_lookup(spatial: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for rel in spatial.get("object_relations", []) or []:
        if isinstance(rel, dict) and rel.get("track_token"):
            lookup[str(rel["track_token"])] = rel
    for token, value in (spatial.get("cross_view_correspondence", {}) or {}).items():
        if isinstance(value, dict):
            lookup.setdefault(str(token), value)
    return lookup


def _object_position_ego(obj: Dict[str, Any], rel: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    candidates: List[Any] = [
        obj.get("position_3d_ego"),
        obj.get("detection_bbox_3d", {}).get("center_3d_ego")
        if isinstance(obj.get("detection_bbox_3d"), dict)
        else None,
    ]
    if rel:
        candidates.extend(
            [
                rel.get("position_3d_ego"),
                rel.get("detection_bbox_3d", {}).get("center_3d_ego")
                if isinstance(rel.get("detection_bbox_3d"), dict)
                else None,
            ]
        )
    for candidate in candidates:
        if isinstance(candidate, dict):
            return {
                "x": _safe_float(candidate.get("x")),
                "y": _safe_float(candidate.get("y")),
                "z": _safe_float(candidate.get("z")),
            }
    return None


def _object_speed_mps(obj: Dict[str, Any], rel: Optional[Dict[str, Any]]) -> Optional[float]:
    for candidate in (
        obj.get("velocity_3d_ego"),
        rel.get("velocity_3d_ego") if rel else None,
    ):
        speed = _vector_speed_mps(candidate)
        if speed is not None:
            return speed
    for candidate in (
        obj.get("speed_mps"),
        obj.get("relative_speed_mps"),
        rel.get("speed_mps") if rel else None,
        rel.get("relative_speed_mps") if rel else None,
    ):
        if candidate is not None:
            return _safe_float(candidate)
    return None


def _draw_spatial_objects_on_image(
    ax: Any,
    objects: Sequence[Dict[str, Any]],
    relation_by_token: Dict[str, Dict[str, Any]],
    img_w: int,
    img_h: int,
    projected_centers: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    for obj in objects:
        bbox = obj.get("detection_bbox_2d")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = [_safe_float(v, np.nan) for v in bbox[:4]]
        if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1 = float(np.clip(x1, 0.0, max(img_w - 1, 0)))
        y1 = float(np.clip(y1, 0.0, max(img_h - 1, 0)))
        x2 = float(np.clip(x2, 0.0, img_w))
        y2 = float(np.clip(y2, 0.0, img_h))
        if x2 <= x1 or y2 <= y1:
            continue

        token = str(obj.get("track_token", ""))
        rel = relation_by_token.get(token)
        category = str(obj.get("category") or obj.get("detection_label") or "object")
        actor_key = _viz_classify_actor(category)
        color = ACTOR_STYLES.get(actor_key, ACTOR_STYLES["generic"]).fill_color or "#7CFC00"

        ax.add_patch(
            mpatches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=2.0,
                alpha=0.95,
                zorder=4,
            )
        )

        pos = _object_position_ego(obj, rel)
        speed = _object_speed_mps(obj, rel)
        label_parts = [category.split(".")[-1][:16]]
        if token:
            label_parts.append(token[:6])
        if pos is not None:
            label_parts.append(f"x={pos['x']:.1f} y={pos['y']:.1f} z={pos['z']:.1f}m")
        if speed is not None:
            label_parts.append(f"v={speed:.1f}m/s")
        ax.text(
            x1,
            max(4.0, y1 - 5.0),
            " | ".join(label_parts),
            fontsize=7,
            color="black",
            bbox=dict(facecolor=color, edgecolor="none", alpha=0.78, pad=1.4),
            ha="left",
            va="bottom",
            zorder=5,
        )

        if projected_centers and token in projected_centers:
            u, v = projected_centers[token]
            if np.isfinite(u) and np.isfinite(v) and 0 <= u <= img_w and 0 <= v <= img_h:
                ax.scatter([u], [v], s=34, c=color, edgecolors="black", linewidths=0.8, zorder=6)


def _project_reasoning_centers(
    metadata: Dict[str, Any],
    camera: str,
    ego_state: EgoState,
    objects: Sequence[Dict[str, Any]],
    relation_by_token: Dict[str, Dict[str, Any]],
    img_w: int,
    img_h: int,
) -> Dict[str, Tuple[float, float]]:
    calib = _get_camera_calibration(metadata, camera)
    if calib is None:
        return {}
    intrinsic, R_cam, t_cam, _, _ = calib
    R_ego, t_ego = _ego_pose_matrix(ego_state)
    projected: Dict[str, Tuple[float, float]] = {}
    for obj in objects:
        token = str(obj.get("track_token", ""))
        if not token:
            continue
        pos = _object_position_ego(obj, relation_by_token.get(token))
        if pos is None:
            continue
        point_ego = np.array([[pos["x"], pos["y"], pos["z"]]], dtype=np.float64)
        point_global = (R_ego @ point_ego.T).T + t_ego.reshape(1, 3)
        uv, valid = _project_points_global_to_image(point_global, intrinsic, R_cam, t_cam, R_ego, t_ego)
        if bool(valid[0]) and np.isfinite(uv[0]).all():
            u, v = float(uv[0, 0]), float(uv[0, 1])
            if -0.25 * img_w <= u <= 1.25 * img_w and -0.25 * img_h <= v <= 1.25 * img_h:
                projected[token] = (u, v)
    return projected


def _frame_image_path(clip_path: str, frame: Dict[str, Any], camera: str, cam_data: Dict[str, Any]) -> str:
    image_rel = cam_data.get("image_path") or (frame.get("sensors", {}).get("cameras", {}) or {}).get(camera, "")
    return os.path.join(clip_path, image_rel) if image_rel else ""


def _save_camera_axis(
    fig: Any,
    ax: Any,
    image_path: str,
    camera: str,
    objects: Sequence[Dict[str, Any]],
    relation_by_token: Dict[str, Dict[str, Any]],
    projected_centers: Dict[str, Tuple[float, float]],
) -> None:
    if image_path and os.path.isfile(image_path):
        image = Image.open(image_path).convert("RGB")
        ax.imshow(image)
        img_w, img_h = image.size
    else:
        img_w, img_h = 1600, 900
        ax.set_facecolor("#333333")
    _draw_spatial_objects_on_image(ax, objects, relation_by_token, img_w, img_h, projected_centers)
    ax.set_title(f"{camera} | objects={len(objects)}", fontsize=10)
    ax.axis("off")


def _clip_result_to_jsonable(result: ClipResult) -> Dict[str, Any]:
    payload = asdict(result) if is_dataclass(result) else dict(result)
    for key in ("planned_traj", "gt_traj"):
        value = payload.get(key)
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
    return payload


def _summary_to_jsonable(summary: Any) -> Dict[str, Any]:

    return {
        "total_clips": summary.total_clips,
        "evaluated_clips": summary.evaluated_clips,
        "failed_clips": summary.failed_clips,
        "mean_collision": summary.mean_collision,
        "mean_driveable": summary.mean_driveable,
        "mean_progress": summary.mean_progress,
        "mean_comfort": summary.mean_comfort,
        "mean_human": summary.mean_human,
        "mean_ADE_m": summary.mean_ade_m,
        "mean_FDE_m": summary.mean_fde_m,
        "mean_planning_score": summary.mean_planning_score,
        "per_clip": [_clip_result_to_jsonable(r) for r in summary.results],
    }


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _build_vla_multiview_inputs(
    provider: VLATrajectoryProvider,
    clip_path: str,
    key_frame_idx: int,
) -> Dict[str, Any]:
    cfg = provider.cfg
    num_hist = int(cfg.get("num_history_steps", 1) or 0)
    flat_images, image_contexts, mission_cmd = load_vlm_observation(
        clip_path, key_frame_idx, cfg, cam_names=provider._cam_names,
    )
    if not flat_images:
        raise IndexError(
            f"key_frame_idx={key_frame_idx} produced no VLM images for {clip_path}"
        )

    prompt = provider.build_planning_prompt(
        num_history_steps=num_hist,
        num_current_cameras=len(provider._cam_names),
        mission_command=mission_cmd,
    )
    inputs = provider.vlm.prepare_inputs(
        images=flat_images,
        text_prompt=prompt,
        image_contexts=image_contexts,
    )
    return {
        "inputs": inputs,
        "mission_command": mission_cmd,
        "prompt": prompt,
    }


def generate_vla_reasoning_for_clip(
    provider: VLATrajectoryProvider,
    clip_path: str,
    key_frame_idx: int,
    *,
    max_new_tokens: int = 4096,
) -> Dict[str, Any]:
    """
    Generate the VLM reasoning response after planning has been produced.

    This intentionally reuses the same multiview prompt/input construction as
    trajectory inference so the saved reasoning corresponds to the planned frame.
    """
    import torch

    prepared = _build_vla_multiview_inputs(provider, clip_path, key_frame_idx)
    inputs = prepared["inputs"]
    vlm_inputs = {
        k: v.to(provider.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
        if k not in {"labels", "prompt_length"}
    }

    generation_kwargs: Dict[str, Any] = {
        "input_ids": vlm_inputs["input_ids"],
        "attention_mask": vlm_inputs["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
        if key in vlm_inputs:
            generation_kwargs[key] = vlm_inputs[key]

    tokenizer = provider.vlm.processor.tokenizer
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with torch.no_grad():
        generated_ids = provider.vlm.model.generate(**generation_kwargs)

    prompt_len = int(vlm_inputs["input_ids"].shape[-1])
    new_tokens = generated_ids[:, prompt_len:]
    reasoning_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    return {
        "clip_path": clip_path,
        "clip_name": os.path.basename(os.path.normpath(clip_path)),
        "key_frame_idx": key_frame_idx,
        "mission_command": prepared["mission_command"],
        "max_new_tokens": max_new_tokens,
        "reasoning_text": reasoning_text,
    }


def _build_summary(results: List[ClipResult], total_clips: int) -> BenchmarkSummary:
    valid = [r for r in results if r.error is None]
    valid_ade = [
        float(r.details_human["ade"])
        for r in valid
        if isinstance(r.details_human.get("ade"), (int, float))
        and np.isfinite(float(r.details_human["ade"]))
    ]
    valid_fde = [
        float(r.details_human["fde"])
        for r in valid
        if isinstance(r.details_human.get("fde"), (int, float))
        and np.isfinite(float(r.details_human["fde"]))
    ]
    return BenchmarkSummary(
        total_clips=total_clips,
        evaluated_clips=len(valid),
        failed_clips=len(results) - len(valid),
        mean_collision=float(np.mean([r.s_collision for r in valid])) if valid else 0.0,
        mean_driveable=float(np.mean([r.s_driveable for r in valid])) if valid else 0.0,
        mean_progress=float(np.mean([r.s_progress for r in valid])) if valid else 0.0,
        mean_comfort=float(np.mean([r.s_comfort for r in valid])) if valid else 0.0,
        mean_human=float(np.mean([r.s_human for r in valid])) if valid else 0.0,
        mean_ade_m=float(np.mean(valid_ade)) if valid_ade else 0.0,
        mean_fde_m=float(np.mean(valid_fde)) if valid_fde else 0.0,
        mean_planning_score=float(np.mean([r.planning_score for r in valid])) if valid else 0.0,
        results=results,
    )


def visualize_vla_model_on_clips(
    data_root: str,
    checkpoint_dir: str,
    output_dir: str,
    *,
    max_clips: int = 0,
    num_inference_steps: int = 5,
    device: str = "cuda",
    seed: int = 42,
    config: Optional[BenchmarkConfig] = None,
    save_report: bool = True,
    clip_paths: Optional[Sequence[str]] = None,
    max_reasoning_tokens: int = 4096,
    planning_prompt: str = "auto",
    planning_reasoning_format: Optional[str] = None,
) -> Any:
    """
    Run VLA inference and save visualization artifacts for each selected clip.

    ``max_clips`` is the only cap: every selected clip gets a per-sample JSON,
    BEV planning image, generated reasoning JSON, and camera overlay image(s).
    """
    os.makedirs(output_dir, exist_ok=True)
    set_random_seed(seed)
    config = config or BenchmarkConfig()
    provider = VLATrajectoryProvider(
        checkpoint_dir=checkpoint_dir,
        num_inference_steps=num_inference_steps,
        device=device,
        planning_prompt=planning_prompt,
        reasoning_format=planning_reasoning_format,
    )

    clips = list(clip_paths) if clip_paths is not None else discover_clips(data_root)
    if max_clips > 0:
        clips = clips[:max_clips]
    logger.info("Visualizing %d clips under %s", len(clips), data_root)

    results: List[ClipResult] = []
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    for sample_idx, clip_path in enumerate(clips):
        clip_path = os.path.normpath(clip_path)
        clip_name = os.path.basename(clip_path)
        sample_dir = os.path.join(samples_dir, f"{sample_idx:04d}_{clip_name[:80]}")
        os.makedirs(sample_dir, exist_ok=True)
        try:
            metadata = _load_metadata(clip_path)
            frames = metadata.get("frames", [])
            key_idx = select_key_frame_idx(frames, config.key_frame_index)
            key_frame = frames[key_idx]
            ego_state: EgoState = _load_pickle(os.path.join(clip_path, key_frame["ego_state"]))
            planned = provider(clip_path, key_idx, ego_state)
            result = evaluate_clip(clip_path, config, planned_trajectory=planned)
        except Exception as exc:
            logger.warning("Failed to visualize clip %s: %s", clip_path, exc)
            result = ClipResult(
                clip_path=clip_path,
                clip_name=clip_name,
                key_frame_idx=-1,
                error=str(exc),
            )

        results.append(result)
        _save_json(os.path.join(sample_dir, "result.json"), _clip_result_to_jsonable(result))

        if result.error is None:
            plot_planning_results_on_bev(
                result,
                os.path.join(sample_dir, "planning_bev.png"),
                config=config,
                show=False,
            )
            reasoning_path = os.path.join(sample_dir, "reasoning.json")
            try:
                reasoning_payload = generate_vla_reasoning_for_clip(
                    provider,
                    clip_path,
                    result.key_frame_idx,
                    max_new_tokens=max_reasoning_tokens,
                )
            except Exception as exc:
                logger.warning("Failed to generate reasoning for clip %s: %s", clip_path, exc)
                reasoning_payload = {
                    "clip_path": clip_path,
                    "clip_name": clip_name,
                    "key_frame_idx": result.key_frame_idx,
                    "error": str(exc),
                }
            _save_json(reasoning_path, reasoning_payload)
            logger.info("Saved generated reasoning to %s", reasoning_path)
            plot_visualization_images_with_spatial_annotations(
                clip_path,
                os.path.join(sample_dir, "spatial_images"),
                key_frame_idx=result.key_frame_idx,
                show=False,
            )

    summary = _build_summary(results, total_clips=len(clips))
    if save_report:
        report_path = os.path.join(output_dir, "vla_visualization_report.json")
        with open(report_path, "w") as f:
            json.dump(_summary_to_jsonable(summary), f, indent=2)
        logger.info("Saved VLA visualization report to %s", report_path)
    return summary


def plot_visualization_images_with_spatial_annotations(
    clip_path: str,
    output_dir: str,
    *,
    key_frame_idx: Optional[int] = None,
    cameras: Sequence[str] = CAMERA_ORDER,
    save_individual: bool = True,
    show: bool = False,
) -> str:
    """
    Save visualization-time camera images with spatial reasoning annotations.

    Each camera view receives the 2-D detection box, the ego-frame 3-D
    location, and object speed when those fields exist in the reasoning JSON.
    The function returns the camera-grid image path.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata = _load_metadata(clip_path)
    frames = metadata.get("frames", [])
    if not frames:
        raise ValueError(f"No frames found in {clip_path}")
    key_idx = select_key_frame_idx(frames) if key_frame_idx is None else int(key_frame_idx)
    key_idx = max(0, min(key_idx, len(frames) - 1))
    frame = frames[key_idx]
    ego_state: EgoState = _load_pickle(os.path.join(clip_path, frame["ego_state"]))
    spatial = _load_spatial_reasoning(clip_path, frame)
    per_cam = spatial.get("per_camera_results", {}) or {}
    relation_by_token = _relation_lookup(spatial)

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    for idx, camera in enumerate(cameras):
        ax = axes[idx]
        cam_data = per_cam.get(camera, {}) if isinstance(per_cam.get(camera, {}), dict) else {}
        objects = cam_data.get("objects", []) or []
        image_path = _frame_image_path(clip_path, frame, camera, cam_data)

        if image_path and os.path.isfile(image_path):
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        else:
            img_w, img_h = 1600, 900
        projected_centers = _project_reasoning_centers(
            metadata, camera, ego_state, objects, relation_by_token, img_w, img_h
        )
        _save_camera_axis(fig, ax, image_path, camera, objects, relation_by_token, projected_centers)

        if save_individual:
            cam_fig, cam_ax = plt.subplots(1, 1, figsize=(12, 7))
            _save_camera_axis(
                cam_fig,
                cam_ax,
                image_path,
                camera,
                objects,
                relation_by_token,
                projected_centers,
            )
            cam_fig.tight_layout()
            cam_out = os.path.join(output_dir, f"{os.path.basename(clip_path)}_frame{key_idx:03d}_{camera}.png")
            cam_fig.savefig(cam_out, dpi=160, bbox_inches="tight", facecolor="white")
            if show:
                plt.show()
            plt.close(cam_fig)

    for ax in axes[len(cameras) :]:
        ax.axis("off")
    fig.suptitle(
        f"Spatial Reasoning Projection | {os.path.basename(clip_path)} | frame={key_idx}",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    grid_path = os.path.join(output_dir, f"{os.path.basename(clip_path)}_frame{key_idx:03d}_camera_grid.png")
    fig.savefig(grid_path, dpi=160, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved spatial camera grid to %s", grid_path)
    return grid_path


def _as_clip_result(result: Any, config: BenchmarkConfig) -> ClipResult:
    if isinstance(result, ClipResult):
        return result
    if not isinstance(result, dict):
        raise TypeError("result must be a ClipResult or a dict from vla_visualization_report.json")

    clip_path = result.get("clip_path")
    if not clip_path:
        raise ValueError("Result dict must contain clip_path")

    planned = result.get("planned_traj") or result.get("planned_trajectory")
    gt = result.get("gt_traj") or result.get("ground_truth_trajectory")
    if planned is not None and gt is not None:
        planned_arr = np.asarray(planned, dtype=np.float64)
        gt_arr = np.asarray(gt, dtype=np.float64)
        ego_pose = result.get("ego_pose") or {}
        key_idx = int(result.get("key_frame_idx", result.get("key_frame", 0)))
        return ClipResult(
            clip_path=clip_path,
            clip_name=result.get("clip_name") or result.get("clip") or os.path.basename(clip_path),
            key_frame_idx=key_idx,
            planned_traj=planned_arr,
            gt_traj=gt_arr,
            ego_pose=ego_pose,
            planning_score=float(result.get("planning_score", 0.0)),
        )

    planned_raw = result.get("trajectory") or result.get("predicted_trajectory")
    if planned_raw is not None:
        return evaluate_clip(clip_path, config, planned_trajectory=np.asarray(planned_raw, dtype=np.float64))
    return evaluate_clip(clip_path, config, planned_trajectory=None)


def plot_planning_results_on_bev(
    result: Any,
    output_path: str,
    *,
    config: Optional[BenchmarkConfig] = None,
    show: bool = False,
) -> str:
    """
    Plot ground-truth and planned trajectories on BEV.

    The legend intentionally contains only "Ground-truth" and "Planning", as
    requested, while map/actors remain unlabeled visual context.
    """
    config = config or BenchmarkConfig()
    clip_result = _as_clip_result(result, config)
    if clip_result.error is not None:
        raise ValueError(f"Cannot plot failed result {clip_result.clip_name}: {clip_result.error}")
    if clip_result.ego_pose is None:
        raise ValueError("ClipResult is missing ego_pose")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    metadata = _load_metadata(clip_result.clip_path)
    key_frame = metadata["frames"][clip_result.key_frame_idx]
    ann: Annotations = _load_pickle(os.path.join(clip_result.clip_path, key_frame["annotations"]))
    ego_state: EgoState = _load_pickle(os.path.join(clip_result.clip_path, key_frame["ego_state"]))

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ego_x = _safe_float(clip_result.ego_pose.get("x"))
    ego_y = _safe_float(clip_result.ego_pose.get("y"))
    _viz_configure_bev_ax(ax, ego_x, ego_y, config.vis_range_m)
    ax.set_title(
        f"{clip_result.clip_name}\nFrame {clip_result.key_frame_idx} | Planning={clip_result.planning_score:.3f}",
        fontsize=12,
        fontweight="bold",
    )

    map_path = os.path.join(clip_result.clip_path, metadata.get("map_annotation", "map.pkl"))
    if os.path.isfile(map_path):
        static_map: nuReasoningStaticMap = _load_pickle(map_path)
        _viz_draw_static_map(ax, static_map)

    future_anns = load_future_object_states(
        clip_result.clip_path,
        metadata.get("frames", []),
        clip_result.key_frame_idx,
        max(config.trajectory_steps - 1, 0),
        dt_s=config.trajectory_dt_s,
    )

    _draw_surrounding_object_future_trajectories(
        ax,
        ann,
        future_anns,
        clip_result.ego_pose,
        config.vis_range_m,
        trajectory_dt_s=config.trajectory_dt_s,
        plot_dt_s=0.5,
    )
    _viz_draw_objects(ax, ann)
    ego_dims = getattr(ego_state, "dimensions", None) or {"l": 5.176, "w": 2.297}
    _viz_draw_ego(ax, clip_result.ego_pose, ego_dims)
    _viz_add_trajectory(
        ax,
        clip_result.gt_traj,
        TRAJECTORY_STYLES["gt"],
        trajectory_dt_s=config.trajectory_dt_s,
    )
    _viz_add_trajectory(
        ax,
        clip_result.planned_traj,
        TRAJECTORY_STYLES["planned"],
        trajectory_dt_s=config.trajectory_dt_s,
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=TRAJECTORY_STYLES["gt"].line_color,
            marker=TRAJECTORY_STYLES["gt"].marker,
            linewidth=2.0,
            label="Ground-truth",
        ),
        Line2D(
            [0],
            [0],
            color=TRAJECTORY_STYLES["planned"].line_color,
            marker=TRAJECTORY_STYLES["planned"].marker,
            linewidth=2.0,
            label="Planning",
        ),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved BEV planning visualization to %s", output_path)
    return output_path


def _iter_valid_results(summary_or_report: Any) -> Iterable[Any]:
    if hasattr(summary_or_report, "results"):
        for result in summary_or_report.results:
            if getattr(result, "error", None) is None:
                yield result
        return
    per_clip = summary_or_report.get("per_clip", []) if isinstance(summary_or_report, dict) else []
    for result in per_clip:
        if result.get("error") is None:
            yield result


def _load_report(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLA inference and save per-clip visualization artifacts.")
    parser.add_argument("--data-root", type=str, default="", help="Root containing clip directories.")
    parser.add_argument("--checkpoint-dir", type=str, default="", help="VLA checkpoint directory.")
    parser.add_argument("--output-dir", type=str, default="nureasoning_test_visualizer_output")
    parser.add_argument("--max-clips", type=int, default=300, help="Number of clips to visualize; 0 means all.")
    parser.add_argument("--num-inference-steps", type=int, default=5)
    parser.add_argument("--max-reasoning-tokens", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-path", type=str, default="", help="Optional single clip path for image projection.")
    parser.add_argument(
        "--key-frame-index",
        type=int,
        default=KEY_FRAME_INDEX,
        help="Metadata frame index used for planning (default: 100; clamped on shorter clips)",
    )
    parser.add_argument("--report-json", type=str, default="", help="Existing report with saved trajectories.")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--skip-vla", action="store_true", help="Only plot from --clip-path or --report-json.")
    add_planning_prompt_arguments(parser, hyphen=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    summary_or_report: Any = None
    config = BenchmarkConfig(key_frame_index=args.key_frame_index)
    if args.report_json:
        summary_or_report = _load_report(args.report_json)
    elif not args.skip_vla:
        if not args.checkpoint_dir or (not args.data_root and not args.clip_path):
            raise SystemExit("--checkpoint-dir and either --data-root or --clip-path are required unless --skip-vla or --report-json is used")
        clip_paths = [args.clip_path] if args.clip_path else None
        data_root = args.data_root or os.path.dirname(os.path.normpath(args.clip_path))
        visualize_vla_model_on_clips(
            data_root=data_root,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            max_clips=0 if clip_paths else args.max_clips,
            num_inference_steps=args.num_inference_steps,
            device=args.device,
            seed=args.seed,
            config=config,
            clip_paths=clip_paths,
            max_reasoning_tokens=args.max_reasoning_tokens,
            planning_prompt=args.planning_prompt,
            planning_reasoning_format=args.planning_reasoning_format,
        )
    if args.report_json:
        max_to_plot = args.max_clips if args.max_clips > 0 else None
        for visualized, result in enumerate(_iter_valid_results(summary_or_report or {})):
            if max_to_plot is not None and visualized >= max_to_plot:
                break
            clip_path = result.clip_path if isinstance(result, ClipResult) else result.get("clip_path")
            if clip_path:
                if isinstance(result, ClipResult):
                    key_frame_idx = result.key_frame_idx
                else:
                    key_frame_idx = result.get("key_frame_idx")
                sample_dir = os.path.join(args.output_dir, "samples_from_report", f"{visualized:04d}_{os.path.basename(clip_path)[:80]}")
                os.makedirs(sample_dir, exist_ok=True)
                with open(os.path.join(sample_dir, "result.json"), "w") as f:
                    json.dump(
                        _clip_result_to_jsonable(result) if isinstance(result, ClipResult) else result,
                        f,
                        indent=2,
                    )
                plot_visualization_images_with_spatial_annotations(
                    clip_path,
                    os.path.join(sample_dir, "spatial_images"),
                    key_frame_idx=key_frame_idx,
                    show=args.show,
                )
            else:
                sample_dir = os.path.join(args.output_dir, "samples_from_report", f"{visualized:04d}_clip")
                os.makedirs(sample_dir, exist_ok=True)
            bev_path = os.path.join(sample_dir, "planning_bev.png")
            plot_planning_results_on_bev(result, bev_path, config=config, show=args.show)

    if args.clip_path and summary_or_report is None and args.skip_vla:
        plot_visualization_images_with_spatial_annotations(
            args.clip_path,
            os.path.join(args.output_dir, "samples", f"0000_{os.path.basename(args.clip_path)[:80]}", "spatial_images"),
            show=args.show,
        )


if __name__ == "__main__":
    main()
