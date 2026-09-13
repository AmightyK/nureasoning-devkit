"""
nuReasoning clip visualizer.

Default output is a composite overview (8 cameras, ego-state card, BEV with
history/future/route, optional lidar). Use ``--video`` for an MP4, or
``--save-panels`` for the older per-reasoning-frame PNG dumps.

Run:
  python -m nureasoning.visualization.view_data --input-root ./dataset/data/train/part_1 --frame-index 0
  python -m nureasoning.visualization.view_data --clip-path <clip> --video --max-frames 50
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt

from nureasoning.common.clips import discover_clips
from nureasoning.common.pickle_io import load_pickle
from nureasoning.common.schema import (
    Annotations,
    EgoState,
    ObjectAnnotation,
    nuReasoningStaticMap,
)
from nureasoning.visualization.ffmpeg_util import require_ffmpeg
from nureasoning.visualization.plotting import (
    ACTOR_STYLES,
    TRAJECTORY_STYLES,
    _metadata_ego_dimensions,
    _viz_add_trajectory,
    _viz_classify_actor,
    _viz_configure_bev_ax,
    _viz_draw_bev_scene,
    _viz_draw_camera_center_ego,
    _viz_draw_ego,
    _viz_draw_objects,
    _viz_draw_static_map,
    _viz_draw_traffic_light_states,
)

# ---------------------------------------------------------------------------
# Standalone PCD reader
# ---------------------------------------------------------------------------

def _lzf_decompress(data: bytes, uncompressed_size: int) -> bytes:
    """Minimal LZF decompressor for PCD binary_compressed."""
    out = bytearray(uncompressed_size)
    i = 0
    o = 0
    n = len(data)
    while i < n and o < uncompressed_size:
        ctrl = data[i]
        i += 1
        if ctrl < 32:
            length = ctrl + 1
            out[o:o + length] = data[i:i + length]
            i += length
            o += length
        else:
            length = (ctrl >> 5)
            if length == 7:
                length += data[i]
                i += 1
            length += 2
            offset = ((ctrl & 0x1F) << 8) + data[i] + 1
            i += 1
            ref = o - offset
            for _ in range(length):
                out[o] = out[ref]
                o += 1
                ref += 1
    return bytes(out)


def load_pcd_points(pcd_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read a PCD file (ascii / binary / binary_compressed) → (xyz[3,N], intensity[N]|None)."""
    import struct

    header: Dict[str, str] = {}
    header_size = 0
    header_lines = 0
    with open(pcd_path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            header_size += len(line)
            header_lines += 1
            text = line.decode("utf-8", errors="replace").strip()
            if text.startswith("#") or not text:
                continue
            parts = text.split(None, 1)
            key = parts[0].upper()
            val = parts[1] if len(parts) > 1 else ""
            header[key] = val
            if key == "DATA":
                break

    num_points = int(header.get("POINTS", header.get("WIDTH", "0")))
    if num_points <= 0:
        return np.zeros((3, 0), dtype=np.float32), None

    fields = header.get("FIELDS", "x y z").split()
    sizes = [int(s) for s in header.get("SIZE", "4 4 4").split()]
    types = header.get("TYPE", "F F F").split()
    counts = [int(c) for c in header.get("COUNT", " ".join(["1"] * len(fields))).split()]
    data_mode = header.get("DATA", "binary").lower()

    _dtype_map = {
        ("F", 4): np.float32, ("F", 8): np.float64,
        ("U", 1): np.uint8, ("U", 2): np.uint16, ("U", 4): np.uint32,
        ("I", 1): np.int8, ("I", 2): np.int16, ("I", 4): np.int32,
    }

    if data_mode == "ascii":
        data = np.loadtxt(pcd_path, skiprows=header_lines)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        xyz = data[:, :3].T.astype(np.float32)
        intensity = data[:, 3].astype(np.float32) if data.shape[1] > 3 else None
        return xyz, intensity

    with open(pcd_path, "rb") as f:
        f.seek(header_size)

        if data_mode == "binary_compressed":
            compressed_size = struct.unpack("<I", f.read(4))[0]
            uncompressed_size = struct.unpack("<I", f.read(4))[0]
            compressed_data = f.read(compressed_size)
            raw = _lzf_decompress(compressed_data, uncompressed_size)

            field_arrays: Dict[str, np.ndarray] = {}
            offset = 0
            for fi, field_name in enumerate(fields):
                s = sizes[fi]
                count = counts[fi]
                t = types[fi]
                dtype = _dtype_map.get((t, s), np.float32)
                total_elems = num_points * count
                arr = np.frombuffer(raw, dtype=dtype, count=total_elems, offset=offset)
                field_arrays[field_name.lower()] = arr.copy()
                offset += total_elems * s
        else:
            point_size = sum(s * c for s, c in zip(sizes, counts))
            raw_bytes = f.read(point_size * num_points)
            field_arrays = {}
            byte_offset = 0
            for fi, field_name in enumerate(fields):
                s = sizes[fi]
                count = counts[fi]
                t = types[fi]
                dtype = _dtype_map.get((t, s), np.float32)
                arr = np.zeros(num_points * count, dtype=dtype)
                for pi in range(num_points):
                    start = pi * point_size + byte_offset
                    for ci in range(count):
                        arr[pi * count + ci] = np.frombuffer(
                            raw_bytes[start + ci * s : start + ci * s + s], dtype=dtype
                        )[0]
                field_arrays[field_name.lower()] = arr
                byte_offset += s * count

    x = field_arrays.get("x", np.zeros(num_points, dtype=np.float32))
    y = field_arrays.get("y", np.zeros(num_points, dtype=np.float32))
    z = field_arrays.get("z", np.zeros(num_points, dtype=np.float32))
    xyz = np.vstack([x, y, z]).astype(np.float32)
    intensity = field_arrays.get("intensity")
    if intensity is not None:
        intensity = intensity.astype(np.float32)
    return xyz, intensity


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAMERA_LAYOUT: List[List[Optional[str]]] = [
    ["front_left", "front", "front_right"],
    ["left", None, "right"],
    ["back_left", "back", "back_right"],
]
CAMERA_ORDER = [cam for row in CAMERA_LAYOUT for cam in row if cam is not None]

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

BEV_RANGE_M = 80.0
LIDAR_RANGE_M = 80.0
LIDAR_MAX_POINTS = 200_000
LIDAR_SAMPLING_MIN_RADIUS_M = 5.0

BOX_3D_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClipContext:
    clip_path: str
    clip_name: str
    metadata: Dict[str, Any]
    frames: List[Dict[str, Any]]
    ts_to_index: Dict[int, int]


@dataclass
class ReasoningEntry:
    clip_path: str
    clip_name: str
    reasoning_path: str
    reasoning_name: str
    frame_index: int


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded if isinstance(loaded, dict) else {}


def _load_pickle(path: str) -> Any:
    return load_pickle(path)


def _resolve_path(path_value: str, clip_path: str) -> str:
    if not path_value:
        return ""
    expanded = os.path.expanduser(path_value)
    if os.path.isabs(expanded):
        return expanded
    direct = os.path.abspath(expanded)
    if os.path.exists(direct):
        return direct
    by_clip = os.path.abspath(os.path.join(clip_path, expanded))
    if os.path.exists(by_clip):
        return by_clip
    stripped = expanded[2:] if expanded.startswith("./") else expanded
    by_cwd = os.path.abspath(stripped)
    if os.path.exists(by_cwd):
        return by_cwd
    return by_clip


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_image(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.isfile(path):
        return None
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return np.asarray(img)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Geometry / projection helpers
# ---------------------------------------------------------------------------

def _quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.eye(3, dtype=np.float64)
    q /= norm
    qw, qx, qy, qz = q.tolist()
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
    ], dtype=np.float64)


def _yaw_to_rotmat(yaw: float) -> np.ndarray:
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _get_camera_calibration(
    metadata: Dict[str, Any], camera_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    calibs = metadata.get("camera_calibrations", {})
    cam_name = CAMERA_KEY_TO_NAME.get(camera_key, camera_key)
    calib = calibs.get(cam_name)
    if calib is None:
        raise KeyError(f"Camera calibration not found for {cam_name}")
    intrinsic = np.array(calib["intrinsic"], dtype=np.float64)
    t = np.array(calib["sensor2lidar_translation"], dtype=np.float64)
    rot = calib["sensor2lidar_rotation"]
    R = _quat_to_rotmat(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
    width, height = int(calib.get("width", 0)), int(calib.get("height", 0))
    return intrinsic, R, t, width, height


def _get_ego_pose(ego_state: EgoState) -> Tuple[np.ndarray, np.ndarray]:
    pose = ego_state.pose
    t_ego = np.array(
        [float(pose.get("x", 0.0)), float(pose.get("y", 0.0)), float(pose.get("z", 0.0))],
        dtype=np.float64,
    )
    R_ego = _quat_to_rotmat(
        float(pose["qw"]), float(pose["qx"]), float(pose["qy"]), float(pose["qz"]),
    )
    return R_ego, t_ego


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
    z = pts_cam[:, 2]
    valid_depth = z > 0.5
    uv = np.full((pts_cam.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid_depth):
        cam_valid = pts_cam[valid_depth]
        uvw = (intrinsic @ cam_valid.T).T
        uv_valid = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)
        uv[valid_depth] = uv_valid
    return uv, valid_depth


def _sample_lidar_indices_by_range(r_xy: np.ndarray, max_points: int) -> np.ndarray:
    """Prefer farther lidar returns when downsampling dense point clouds."""
    num_pts = int(r_xy.shape[0])
    if num_pts <= max_points:
        return np.arange(num_pts)

    # Near returns dominate raw lidar density, so weight by radial distance to
    # keep more far-field structure in the visualization.
    weights = np.clip(np.asarray(r_xy, dtype=np.float64), LIDAR_SAMPLING_MIN_RADIUS_M, None)
    weight_sum = float(weights.sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return np.random.choice(num_pts, max_points, replace=False)

    rng = np.random.default_rng()
    return rng.choice(num_pts, size=max_points, replace=False, p=weights / weight_sum)


def _box3d_corners_global(
    cx: float, cy: float, cz: float,
    l: float, w: float, h: float,
    yaw: float,
) -> np.ndarray:
    """8 corners of a 3D box in global frame.  Order: bottom-4 then top-4."""
    half_l, half_w, half_h = l / 2.0, w / 2.0, h / 2.0
    local = np.array([
        [ half_l,  half_w, -half_h],
        [ half_l, -half_w, -half_h],
        [-half_l, -half_w, -half_h],
        [-half_l,  half_w, -half_h],
        [ half_l,  half_w,  half_h],
        [ half_l, -half_w,  half_h],
        [-half_l, -half_w,  half_h],
        [-half_l,  half_w,  half_h],
    ], dtype=np.float64)
    R = _yaw_to_rotmat(yaw)
    return (R @ local.T).T + np.array([cx, cy, cz], dtype=np.float64)


def _project_3d_box_to_image(
    corners_global: np.ndarray,
    intrinsic: np.ndarray,
    R_cam_to_lidar: np.ndarray,
    t_cam_to_lidar: np.ndarray,
    R_lidar_to_global: np.ndarray,
    t_lidar_to_global: np.ndarray,
    img_w: int,
    img_h: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Project 8 corners → pixel coords.  Returns (uv[8,2], valid[8])."""
    uv, valid = _project_points_global_to_image(
        corners_global, intrinsic, R_cam_to_lidar, t_cam_to_lidar,
        R_lidar_to_global, t_lidar_to_global,
    )
    in_img = (
        valid
        & np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
        & (uv[:, 0] >= -img_w * 0.3) & (uv[:, 0] <= img_w * 1.3)
        & (uv[:, 1] >= -img_h * 0.3) & (uv[:, 1] <= img_h * 1.3)
    )
    if in_img.sum() < 2:
        return None
    return uv, in_img


def _annotation_object_corners_global(obj: ObjectAnnotation) -> Optional[np.ndarray]:
    try:
        cx = float(obj.pose.get("x", 0.0))
        cy = float(obj.pose.get("y", 0.0))
        cz = float(obj.pose.get("z", 0.0))
        l = float(obj.dimensions.get("l", 0.0))
        w = float(obj.dimensions.get("w", 0.0))
        h = float(obj.dimensions.get("h", 0.0))
        yaw = float(obj.pose.get("yaw", 0.0))
    except Exception:
        return None

    if l <= 0.0 or w <= 0.0 or h <= 0.0:
        return None
    return _box3d_corners_global(cx, cy, cz, l, w, h, yaw)


# ---------------------------------------------------------------------------
# Clip / entry discovery
# ---------------------------------------------------------------------------

def _discover_clips(input_root: str, clip_path: str) -> List[ClipContext]:
    if clip_path:
        clip_abs = os.path.abspath(clip_path)
        metadata_path = os.path.join(clip_abs, "metadata.json")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"metadata.json not found in clip path: {clip_abs}")
        return [_build_clip_context(clip_abs)]

    root_abs = os.path.abspath(input_root)
    if not os.path.isdir(root_abs):
        raise FileNotFoundError(f"Input root not found: {root_abs}")

    if os.path.isfile(os.path.join(root_abs, "metadata.json")):
        return [_build_clip_context(root_abs)]

    return [_build_clip_context(path) for path in discover_clips(root_abs)]


def _build_clip_context(clip_path: str) -> ClipContext:
    metadata = _load_json(os.path.join(clip_path, "metadata.json"))
    frames = metadata.get("frames", [])
    if not isinstance(frames, list):
        frames = []
    ts_to_index: Dict[int, int] = {}
    for i, frame in enumerate(frames):
        ts_raw = frame.get("timestamp_us")
        try:
            ts_to_index[int(ts_raw)] = i
        except Exception:
            continue
    return ClipContext(
        clip_path=clip_path,
        clip_name=os.path.basename(clip_path),
        metadata=metadata,
        frames=frames,
        ts_to_index=ts_to_index,
    )


def _reasoning_sort_key(path: str) -> Tuple[int, str]:
    name = os.path.basename(path)
    stem, _ = os.path.splitext(name)
    if stem.isdigit():
        return (int(stem), name)
    return (10**30, name)


def _collect_entries(clips: List[ClipContext], max_entries: int) -> List[ReasoningEntry]:
    entries: List[ReasoningEntry] = []
    for clip in clips:
        reasoning_dir = os.path.join(clip.clip_path, "reasoning")
        if not os.path.isdir(reasoning_dir):
            continue
        json_paths = [
            os.path.join(reasoning_dir, f)
            for f in os.listdir(reasoning_dir)
            if f.lower().endswith(".json")
        ]
        for path in sorted(json_paths, key=_reasoning_sort_key):
            name = os.path.basename(path)
            stem, _ = os.path.splitext(name)
            frame_index = -1
            if stem.isdigit():
                frame_index = clip.ts_to_index.get(int(stem), -1)
            entries.append(
                ReasoningEntry(
                    clip_path=clip.clip_path,
                    clip_name=clip.clip_name,
                    reasoning_path=path,
                    reasoning_name=name,
                    frame_index=frame_index,
                )
            )
    if max_entries > 0:
        return entries[:max_entries]
    return entries


# ---------------------------------------------------------------------------
# Frame data loader
# ---------------------------------------------------------------------------

class FrameData:
    """Container for all data needed to render a single reasoning frame."""

    def __init__(
        self,
        entry: ReasoningEntry,
        clip_ctx: ClipContext,
        reasoning: Dict[str, Any],
    ):
        self.entry = entry
        self.clip_ctx = clip_ctx
        self.reasoning = reasoning
        self.clip_path = entry.clip_path
        self.metadata = clip_ctx.metadata

        self.frame = self._resolve_frame()
        self.ego_state: Optional[EgoState] = None
        self.annotations: Optional[Annotations] = None
        self.static_map: Optional[nuReasoningStaticMap] = None

        self._load_ego_and_annotations()
        self._load_map()

    def _resolve_frame(self) -> Dict[str, Any]:
        entry = self.entry
        clip_ctx = self.clip_ctx
        if 0 <= entry.frame_index < len(clip_ctx.frames):
            return clip_ctx.frames[entry.frame_index]
        data_idx = self.reasoning.get("frame_index")
        if isinstance(data_idx, int) and 0 <= data_idx < len(clip_ctx.frames):
            entry.frame_index = data_idx
            return clip_ctx.frames[data_idx]
        stem, _ = os.path.splitext(entry.reasoning_name)
        if stem.isdigit():
            idx = clip_ctx.ts_to_index.get(int(stem), -1)
            if 0 <= idx < len(clip_ctx.frames):
                entry.frame_index = idx
                return clip_ctx.frames[idx]
        return {}

    def _load_ego_and_annotations(self) -> None:
        if not self.frame:
            return
        ego_rel = self.frame.get("ego_state", "")
        ego_abs = _resolve_path(ego_rel, self.clip_path)
        if ego_abs and os.path.isfile(ego_abs):
            try:
                self.ego_state = _load_pickle(ego_abs)
            except Exception:
                pass
        ann_rel = self.frame.get("annotations", "")
        ann_abs = _resolve_path(ann_rel, self.clip_path)
        if ann_abs and os.path.isfile(ann_abs):
            try:
                self.annotations = _load_pickle(ann_abs)
            except Exception:
                pass

    def _load_map(self) -> None:
        map_rel = self.metadata.get("map_annotation", "map.pkl")
        map_abs = os.path.join(self.clip_path, map_rel)
        if not os.path.isfile(map_abs):
            map_abs = os.path.join(self.clip_path, "map.pkl")
        if os.path.isfile(map_abs):
            try:
                self.static_map = _load_pickle(map_abs)
            except Exception:
                pass

    @property
    def ego_pose_dict(self) -> Dict[str, float]:
        if self.ego_state is None:
            return {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        return self.ego_state.pose

    @property
    def ego_dims_dict(self) -> Dict[str, float]:
        defaults = {"l": 5.176, "w": 2.297, "h": 1.8}
        dims = dict(defaults)
        dims.update(_metadata_ego_dimensions(self.metadata))
        if self.ego_state is not None:
            raw = getattr(self.ego_state, "dimensions", None)
            if isinstance(raw, dict):
                for key in ("l", "w", "h", "vehicle_rear_length"):
                    if key in raw:
                        dims[key] = float(raw[key])
        return dims

    def camera_image_path(self, cam_key: str) -> str:
        sensors = self.frame.get("sensors", {})
        if isinstance(sensors, dict):
            cams = sensors.get("cameras", {})
        else:
            cams = getattr(sensors, "cameras", {})
        if isinstance(cams, dict):
            rel = cams.get(cam_key, "")
        else:
            rel = getattr(cams, cam_key, "")
        return _resolve_path(str(rel), self.clip_path)

    def lidar_path(self) -> str:
        sensors = self.frame.get("sensors", {})
        if isinstance(sensors, dict):
            lidar = sensors.get("lidar", {})
        else:
            lidar = getattr(sensors, "lidar", {})
        if isinstance(lidar, dict):
            rel = lidar.get("point_cloud_path", "")
        else:
            rel = getattr(lidar, "point_cloud_path", "")
        return _resolve_path(str(rel), self.clip_path)

    def camera_reasoning_objects(self, cam_key: str) -> List[Dict[str, Any]]:
        per_camera = self.reasoning.get("per_camera_results")
        if not isinstance(per_camera, dict):
            spatial = self.reasoning.get("Spatial", {})
            if isinstance(spatial, dict):
                per_camera = spatial.get("per_camera_results", {})
            else:
                per_camera = {}
        if not isinstance(per_camera, dict):
            return []
        cam_payload = per_camera.get(cam_key, {})
        if not isinstance(cam_payload, dict):
            return []
        objects = cam_payload.get("objects", [])
        if not isinstance(objects, list):
            return []
        return [obj for obj in objects if isinstance(obj, dict)]

    def annotation_objects(self) -> List[ObjectAnnotation]:
        if self.annotations is None:
            return []
        objects = getattr(self.annotations, "objects", [])
        return [obj for obj in objects if isinstance(obj, ObjectAnnotation)]

    def camera_map_overlays(self, cam_key: str) -> Dict[str, List[List[List[float]]]]:
        spatial = self.reasoning.get("Spatial", {})
        if not isinstance(spatial, dict):
            spatial = {}
        map_payload = spatial.get("map", self.reasoning.get("map", {}))
        if not isinstance(map_payload, dict):
            return {"baseline_paths": [], "crosswalks": []}

        overlays: Dict[str, List[List[List[float]]]] = {
            "baseline_paths": [],
            "crosswalks": [],
        }
        for layer_name in ("baseline_paths", "crosswalks"):
            layer_payload = map_payload.get(layer_name, {})
            if not isinstance(layer_payload, dict):
                continue
            per_camera = layer_payload.get("per_camera_projections", {})
            if not isinstance(per_camera, dict):
                continue
            segments = per_camera.get(cam_key, [])
            if isinstance(segments, list):
                overlays[layer_name] = [seg for seg in segments if isinstance(seg, list)]
        return overlays


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class NuReasoningRenderer:
    """Renders separate images for each visualization type."""

    def __init__(self, dpi: int = 150):
        self.dpi = dpi

    # ---- 1. Per-camera views with reasoning 2D detections -----------------

    def render_camera_views(
        self, fd: FrameData, save_dir: str,
    ) -> int:
        """Save one image per camera view. Returns number of images saved."""
        os.makedirs(save_dir, exist_ok=True)

        saved = 0
        for cam_key in CAMERA_ORDER:
            img_path = fd.camera_image_path(cam_key)
            img = _safe_image(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            ax.imshow(img)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.axis("off")
            ax.set_title(cam_key, fontsize=13, fontweight="bold")

            self._draw_map_overlays_on_ax(ax, fd.camera_map_overlays(cam_key), w, h)
            self._draw_reasoning_2d_boxes_on_ax(ax, fd.camera_reasoning_objects(cam_key), w, h)

            out_path = os.path.join(save_dir, f"cam_{cam_key}.png")
            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            saved += 1

        return saved

    def render_camera_views_3d_boxes(
        self, fd: FrameData, save_dir: str,
    ) -> int:
        """Save one image per camera with projected 3D annotation boxes."""
        os.makedirs(save_dir, exist_ok=True)

        saved = 0
        has_calib = bool(fd.metadata.get("camera_calibrations")) and fd.ego_state is not None
        R_ego: Optional[np.ndarray] = None
        t_ego: Optional[np.ndarray] = None
        if has_calib and fd.ego_state is not None:
            try:
                R_ego, t_ego = _get_ego_pose(fd.ego_state)
            except Exception:
                R_ego, t_ego = None, None

        for cam_key in CAMERA_ORDER:
            img_path = fd.camera_image_path(cam_key)
            img = _safe_image(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            ax.imshow(img)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.axis("off")
            ax.set_title(f"{cam_key} 3D annotation boxes", fontsize=13, fontweight="bold")

            if R_ego is not None and t_ego is not None:
                try:
                    intrinsic, R_cam, t_cam, img_w, img_h = _get_camera_calibration(fd.metadata, cam_key)
                    if img_w == 0:
                        img_w = w
                    if img_h == 0:
                        img_h = h
                    self._draw_annotation_3d_boxes_on_ax(
                        ax,
                        fd.annotation_objects(),
                        intrinsic,
                        R_cam,
                        t_cam,
                        R_ego,
                        t_ego,
                        img_w,
                        img_h,
                    )
                except Exception:
                    pass

            out_path = os.path.join(save_dir, f"cam_{cam_key}.png")
            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            saved += 1

        return saved

    def _draw_map_overlays_on_ax(
        self,
        ax,
        map_overlays: Dict[str, List[List[List[float]]]],
        img_w: int,
        img_h: int,
    ) -> None:
        baseline_color = "#00E5FF"
        baseline_outline = "#003B46"
        crosswalk_fill = "#FFD400"
        crosswalk_edge = "#FF2D55"

        for segment in map_overlays.get("crosswalks", []):
            pts: List[List[float]] = []
            for point in segment:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                x = _safe_float(point[0], default=np.nan)
                y = _safe_float(point[1], default=np.nan)
                if np.isfinite(x) and np.isfinite(y):
                    pts.append([
                        float(np.clip(x, 0.0, float(img_w))),
                        float(np.clip(y, 0.0, float(img_h))),
                    ])
            if len(pts) < 2:
                continue

            polygon = mpatches.Polygon(
                pts,
                closed=True,
                fill=True,
                facecolor=crosswalk_fill,
                edgecolor=crosswalk_edge,
                linewidth=2.0,
                alpha=0.28,
                zorder=2,
            )
            ax.add_patch(polygon)

        for segment in map_overlays.get("baseline_paths", []):
            pts: List[List[float]] = []
            for point in segment:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                x = _safe_float(point[0], default=np.nan)
                y = _safe_float(point[1], default=np.nan)
                if np.isfinite(x) and np.isfinite(y):
                    pts.append([x, y])
            if len(pts) < 2:
                continue

            pts_arr = np.asarray(pts, dtype=np.float64)
            ax.plot(
                pts_arr[:, 0],
                pts_arr[:, 1],
                color=baseline_outline,
                linestyle="--",
                linewidth=4.0,
                alpha=0.95,
                zorder=3,
            )
            ax.plot(
                pts_arr[:, 0],
                pts_arr[:, 1],
                color=baseline_color,
                linestyle="--",
                linewidth=2.4,
                alpha=1.0,
                zorder=4,
            )

    def _draw_reasoning_2d_boxes_on_ax(
        self,
        ax,
        objects: List[Dict[str, Any]],
        img_w: int,
        img_h: int,
    ) -> None:
        for obj in objects:
            bbox = obj.get("detection_bbox_2d")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue

            x1 = _safe_float(bbox[0])
            y1 = _safe_float(bbox[1])
            x2 = _safe_float(bbox[2])
            y2 = _safe_float(bbox[3])
            if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
                continue

            x1 = float(np.clip(min(x1, x2), 0.0, max(float(img_w - 1), 0.0)))
            y1 = float(np.clip(min(y1, y2), 0.0, max(float(img_h - 1), 0.0)))
            x2 = float(np.clip(max(x1, x2), 0.0, float(img_w)))
            y2 = float(np.clip(max(y1, y2), 0.0, float(img_h)))
            if x2 <= x1 or y2 <= y1:
                continue

            cat = str(obj.get("category", "")).lower()
            actor_key = _viz_classify_actor(cat)
            color = ACTOR_STYLES.get(actor_key, ACTOR_STYLES["generic"]).fill_color or "lime"
            rect = mpatches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=2.0,
                alpha=0.95,
            )
            ax.add_patch(rect)

            label_name = str(obj.get("detection_label") or obj.get("category") or "obj")
            track_token = str(obj.get("track_token", ""))
            token_suffix = track_token[:6] if track_token else ""
            label = f"{label_name[:18]}|{token_suffix}" if token_suffix else label_name[:18]
            ax.text(
                x1,
                max(4.0, y1 - 6.0),
                label,
                fontsize=7,
                color="black",
                bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=1),
                ha="left",
                va="bottom",
            )

    def _draw_annotation_3d_boxes_on_ax(
        self,
        ax,
        objects: List[ObjectAnnotation],
        intrinsic: np.ndarray,
        R_cam: np.ndarray,
        t_cam: np.ndarray,
        R_ego: np.ndarray,
        t_ego: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> None:
        for obj in objects:
            cat = str(obj.category or "").lower()
            if cat.startswith("other.") or cat == "other":
                continue

            corners_global = _annotation_object_corners_global(obj)
            if corners_global is None:
                continue

            projected = _project_3d_box_to_image(
                corners_global,
                intrinsic,
                R_cam,
                t_cam,
                R_ego,
                t_ego,
                img_w,
                img_h,
            )
            if projected is None:
                continue

            uv, valid = projected
            actor_key = _viz_classify_actor(cat)
            color = ACTOR_STYLES.get(actor_key, ACTOR_STYLES["generic"]).fill_color or "#00FF66"

            for i0, i1 in BOX_3D_EDGES:
                if not (bool(valid[i0]) and bool(valid[i1])):
                    continue
                ax.plot(
                    [uv[i0, 0], uv[i1, 0]],
                    [uv[i0, 1], uv[i1, 1]],
                    color="black",
                    linewidth=3.8,
                    alpha=0.75,
                    zorder=5,
                )
                ax.plot(
                    [uv[i0, 0], uv[i1, 0]],
                    [uv[i0, 1], uv[i1, 1]],
                    color=color,
                    linewidth=2.2,
                    alpha=0.95,
                    zorder=6,
                )

            valid_uv = uv[valid]
            if valid_uv.shape[0] == 0:
                continue
            anchor_idx = int(np.argmin(valid_uv[:, 1]))
            anchor_x = float(valid_uv[anchor_idx, 0])
            anchor_y = float(valid_uv[anchor_idx, 1])

            label_name = str(obj.category or "obj").split(".")[-1]
            track_token = str(obj.track_token or "")
            token_suffix = track_token[:6] if track_token else ""
            label = f"{label_name[:18]}|{token_suffix}" if token_suffix else label_name[:18]
            ax.text(
                anchor_x,
                max(6.0, anchor_y - 8.0),
                label,
                fontsize=7,
                color="white",
                bbox=dict(facecolor="black", alpha=0.65, edgecolor=color, pad=1.2),
                ha="left",
                va="bottom",
                zorder=7,
            )

    # ---- 2. BEV panels: map, boxes, combined -----------------------------

    def render_bev_panels(
        self, fd: FrameData, save_dir: str,
    ) -> List[str]:
        ego_pose = fd.ego_pose_dict
        ego_dims = fd.ego_dims_dict
        ego_x = float(ego_pose.get("x", 0.0))
        ego_y = float(ego_pose.get("y", 0.0))
        static_map_lookup: Optional[Dict[str, Dict[int, Any]]] = None

        saved: List[str] = []

        # (a) HD map only
        fig_map, ax_map = plt.subplots(1, 1, figsize=(10, 10))
        _viz_configure_bev_ax(ax_map, ego_x, ego_y, BEV_RANGE_M)
        ax_map.set_title("BEV: HD Map", fontsize=13, fontweight="bold")
        if fd.static_map is not None:
            static_map_lookup = _viz_draw_static_map(ax_map, fd.static_map)
        if fd.annotations is not None:
            _viz_draw_traffic_light_states(ax_map, fd.annotations, static_map_lookup)
        _viz_draw_ego(ax_map, ego_pose, ego_dims)
        path_map = os.path.join(save_dir, "bev_map.png")
        fig_map.savefig(path_map, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig_map)
        saved.append(path_map)

        # (b) Bounding boxes only
        fig_box, ax_box = plt.subplots(1, 1, figsize=(10, 10))
        _viz_configure_bev_ax(ax_box, ego_x, ego_y, BEV_RANGE_M)
        ax_box.set_title("BEV: 3D Objects", fontsize=13, fontweight="bold")
        ax_box.set_facecolor("white")
        if fd.annotations is not None:
            _viz_draw_objects(ax_box, fd.annotations)
        _viz_draw_ego(ax_box, ego_pose, ego_dims)

        gt_traj = None
        if fd.ego_state is not None and fd.ego_state.trajectory_future:
            gt_traj = np.asarray(fd.ego_state.trajectory_future, dtype=np.float64)
        if gt_traj is not None:
            _viz_add_trajectory(ax_box, gt_traj, TRAJECTORY_STYLES["future"])
        hist_traj = None
        if fd.ego_state is not None and fd.ego_state.trajectory_history:
            hist_traj = np.asarray(fd.ego_state.trajectory_history, dtype=np.float64)
        if hist_traj is not None:
            _viz_add_trajectory(ax_box, hist_traj, TRAJECTORY_STYLES["history"])

        path_box = os.path.join(save_dir, "bev_objects.png")
        fig_box.savefig(path_box, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig_box)
        saved.append(path_box)

        # (c) Combined map + objects + ego motion + route
        fig_comb, ax_comb = plt.subplots(1, 1, figsize=(10, 10))
        _viz_draw_bev_scene(
            ax_comb,
            fd.ego_state,
            fd.annotations,
            fd.static_map,
            mission_goal=(fd.frame or {}).get("mission_goal"),
            metadata=fd.metadata,
            bev_range_m=BEV_RANGE_M,
        )
        path_comb = os.path.join(save_dir, "bev_combined.png")
        fig_comb.savefig(path_comb, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig_comb)
        saved.append(path_comb)

        return saved

    # ---- 3. Lidar top-down -----------------------------------------------

    def render_lidar(
        self, fd: FrameData, save_path: str,
    ) -> bool:
        lidar_abs = fd.lidar_path()
        if not lidar_abs or not os.path.isfile(lidar_abs):
            return False

        try:
            xyz, intensity = load_pcd_points(lidar_abs)
        except Exception:
            return False

        if xyz is None or xyz.size == 0:
            return False

        x, y, z = xyz[0, :], xyz[1, :], xyz[2, :]
        r = np.hypot(x, y)

        valid = (
            np.isfinite(xyz).all(axis=0)
            & (r >= 2.0) & (r <= LIDAR_RANGE_M)
            & (z >= -2.5) & (z <= 4.0)
        )
        xyz = xyz[:, valid]
        if intensity is not None:
            intensity = intensity[valid]

        if xyz.size == 0:
            return False

        voxels = np.floor(xyz.T / 0.5).astype(np.int32)
        _, unique_idx = np.unique(voxels, axis=0, return_index=True)
        xyz = xyz[:, unique_idx]
        if intensity is not None:
            intensity = intensity[unique_idx]

        num_pts = xyz.shape[1]
        if num_pts > LIDAR_MAX_POINTS:
            r = np.hypot(xyz[0, :], xyz[1, :])
            idx = _sample_lidar_indices_by_range(r, LIDAR_MAX_POINTS)
            xyz = xyz[:, idx]
            intensity = intensity[idx] if intensity is not None else None

        if intensity is not None and intensity.size > 0:
            lo, hi = np.percentile(intensity, [2, 98])
            if hi > lo:
                intensity = np.clip(intensity, lo, hi)
            colors = intensity
        else:
            colors = xyz[2, :]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_aspect("equal")
        ax.set_facecolor("#111111")
        ax.scatter(
            xyz[0, :], xyz[1, :], s=1.0, c=colors, cmap="viridis",
            alpha=0.75, linewidths=0, rasterized=True,
        )
        ax.set_xlim(-LIDAR_RANGE_M, LIDAR_RANGE_M)
        ax.set_ylim(-LIDAR_RANGE_M, LIDAR_RANGE_M)
        ax.set_title(
            f"Lidar Top-Down  ({xyz.shape[1]:,} pts)",
            fontsize=13, fontweight="bold", color="white",
        )
        ax.tick_params(colors="white")
        ax.scatter([0], [0], marker="*", s=80, c="#DE7061", zorder=10,
                   edgecolors="white", linewidths=0.5)

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return True

    # ---- 4. Front camera + ego future trajectory -------------------------

    def render_front_with_trajectory(
        self, fd: FrameData, save_path: str,
    ) -> bool:
        img_path = fd.camera_image_path("front")
        img = _safe_image(img_path)
        if img is None:
            return False

        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.imshow(img)
        h, w = img.shape[:2]
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        ax.set_title("Front Camera + Ego Future Trajectory", fontsize=13, fontweight="bold")

        has_calib = bool(fd.metadata.get("camera_calibrations"))
        traj_points = None
        if fd.ego_state is not None:
            traj_points = fd.ego_state.trajectory_future

        if has_calib and traj_points is not None and fd.ego_state is not None:
            traj_arr = np.array(traj_points, dtype=np.float64)
            if traj_arr.ndim == 2 and traj_arr.shape[0] >= 2 and traj_arr.shape[1] >= 2:
                # Stored future waypoints are [x, y, yaw], not [x, y, z].
                # Project them at the current ego height in the global frame.
                ego_z = float(fd.ego_pose_dict.get("z", 0.0))
                traj_arr = np.column_stack([
                    traj_arr[:, 0],
                    traj_arr[:, 1],
                    np.full(traj_arr.shape[0], ego_z, dtype=np.float64),
                ])

                try:
                    R_ego, t_ego = _get_ego_pose(fd.ego_state)
                    intrinsic, R_cam, t_cam, img_w, img_h = _get_camera_calibration(fd.metadata, "front")
                    if img_w == 0:
                        img_w = w
                    if img_h == 0:
                        img_h = h

                    uv, valid = _project_points_global_to_image(
                        traj_arr[:, :3], intrinsic, R_cam, t_cam, R_ego, t_ego,
                    )
                    in_img = (
                        valid
                        & np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
                        & (uv[:, 0] >= 0) & (uv[:, 0] <= img_w)
                        & (uv[:, 1] >= 0) & (uv[:, 1] <= img_h)
                    )

                    uv_vis = uv[in_img]
                    if uv_vis.shape[0] >= 2:
                        n = uv_vis.shape[0]
                        cmap = plt.cm.RdYlGn_r
                        for k in range(n - 1):
                            t_frac = k / max(n - 1, 1)
                            c = cmap(t_frac)
                            ax.plot(
                                [uv_vis[k, 0], uv_vis[k + 1, 0]],
                                [uv_vis[k, 1], uv_vis[k + 1, 1]],
                                color=c, linewidth=3.5, alpha=0.9,
                            )
                        for k in range(0, n, max(1, n // 8)):
                            t_frac = k / max(n - 1, 1)
                            c = cmap(t_frac)
                            ax.scatter(
                                [uv_vis[k, 0]], [uv_vis[k, 1]],
                                s=45, c=[c], edgecolors="black", linewidths=0.5, zorder=10,
                            )
                except Exception:
                    pass

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return True

    # ---- Top-level render-all for one entry ------------------------------

    def render_entry(
        self,
        entry: ReasoningEntry,
        clip_ctx: ClipContext,
        output_dir: str,
        overwrite: bool = False,
    ) -> int:
        """Returns number of images saved."""
        stem, _ = os.path.splitext(entry.reasoning_name)
        save_dir = os.path.join(output_dir, entry.clip_name, stem)
        os.makedirs(save_dir, exist_ok=True)

        reasoning = _load_json(entry.reasoning_path)
        fd = FrameData(entry, clip_ctx, reasoning)

        saved = 0

        # 1) Per-camera views with reasoning 2D detections (one image per camera)
        cam_dir = os.path.join(save_dir, "cameras")
        need_cams = overwrite or not all(
            os.path.isfile(os.path.join(cam_dir, f"cam_{c}.png"))
            for c in CAMERA_ORDER
        )
        if need_cams:
            saved += self.render_camera_views(fd, cam_dir)

        # 1b) Per-camera views with projected 3D boxes
        cam_3d_dir = os.path.join(save_dir, "cameras_3d_boxes")
        need_cams_3d = overwrite or not all(
            os.path.isfile(os.path.join(cam_3d_dir, f"cam_{c}.png"))
            for c in CAMERA_ORDER
        )
        if need_cams_3d:
            saved += self.render_camera_views_3d_boxes(fd, cam_3d_dir)

        # 2) BEV panels (map, boxes, combined)
        need_bev = overwrite or not all(
            os.path.isfile(os.path.join(save_dir, f))
            for f in ("bev_map.png", "bev_objects.png", "bev_combined.png")
        )
        if need_bev:
            bev_paths = self.render_bev_panels(fd, save_dir)
            saved += len(bev_paths)

        # 3) Lidar
        lidar_path = os.path.join(save_dir, "lidar.png")
        if overwrite or not os.path.isfile(lidar_path):
            if self.render_lidar(fd, lidar_path):
                saved += 1

        # 4) Front + ego trajectory
        traj_path = os.path.join(save_dir, "front_trajectory.png")
        if overwrite or not os.path.isfile(traj_path):
            if self.render_front_with_trajectory(fd, traj_path):
                saved += 1

        return saved


FIG_MARGIN = 0.006
GC_COLLECT_EVERY_FRAMES = 25


class NuReasoningOverview:
    """Composite 8-camera + BEV (+ lidar) viewer used for PNG/MP4 clip overviews."""

    def __init__(
        self,
        clip: ClipContext,
        *,
        bev_range_m: float = BEV_RANGE_M,
        lidar_range_m: float = LIDAR_RANGE_M,
        max_lidar_points: int = 50_000,
        trajectory_dt_s: float = 0.1,
        render_lidar: bool = True,
        dpi: int = 150,
    ):
        if not clip.frames:
            raise ValueError(f"No frames found in clip: {clip.clip_path}")
        self.clip = clip
        self.bev_range_m = bev_range_m
        self.lidar_range_m = lidar_range_m
        self.max_lidar_points = max_lidar_points
        self.trajectory_dt_s = trajectory_dt_s
        self.dpi = dpi
        self.static_map = self._load_map()

        figure_size = (28, 8) if render_lidar else (20, 8)
        self.fig = plt.figure(figsize=figure_size, facecolor="black")
        self.fig.subplots_adjust(
            left=FIG_MARGIN, right=1.0 - FIG_MARGIN,
            bottom=FIG_MARGIN, top=1.0 - FIG_MARGIN,
            wspace=0, hspace=0,
        )
        if render_lidar:
            outer = self.fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 1.06], wspace=0.0)
        else:
            outer = self.fig.add_gridspec(1, 2, width_ratios=[1.25, 1.25], wspace=0.0)
        cam_grid = outer[0].subgridspec(3, 3, hspace=0.0, wspace=0.0)
        self.cam_axes: Dict[str, Any] = {}
        self.center_cam_ax = None
        for row_idx, row in enumerate(CAMERA_LAYOUT):
            for col_idx, cam in enumerate(row):
                ax = self.fig.add_subplot(cam_grid[row_idx, col_idx])
                if cam is None:
                    self.center_cam_ax = ax
                else:
                    self.cam_axes[cam] = ax
        self.bev_ax = self.fig.add_subplot(outer[1])
        self.lidar_ax = self.fig.add_subplot(outer[2]) if render_lidar else None

    def _load_map(self) -> Optional[nuReasoningStaticMap]:
        map_rel = self.clip.metadata.get("map_annotation", "map.pkl")
        map_path = _resolve_path(str(map_rel), self.clip.clip_path)
        if not os.path.isfile(map_path):
            map_path = os.path.join(self.clip.clip_path, "map.pkl")
        if not os.path.isfile(map_path):
            return None
        try:
            return _load_pickle(map_path)
        except Exception:
            return None

    def _load_ego(self, frame: Dict[str, Any]) -> Optional[EgoState]:
        path = _resolve_path(str(frame.get("ego_state", "")), self.clip.clip_path)
        if not path or not os.path.isfile(path):
            return None
        try:
            return _load_pickle(path)
        except Exception:
            return None

    def _load_annotations(self, frame: Dict[str, Any]) -> Optional[Annotations]:
        path = _resolve_path(str(frame.get("annotations", "") or ""), self.clip.clip_path)
        if not path or not os.path.isfile(path):
            ts = frame.get("timestamp_us")
            if ts is not None:
                path = _resolve_path(f"annotations/{ts}.pkl", self.clip.clip_path)
        if not path or not os.path.isfile(path):
            return None
        try:
            return _load_pickle(path)
        except Exception:
            return None

    def _draw_cameras(self, frame: Dict[str, Any], ego_state: Optional[EgoState]) -> None:
        sensors = frame.get("sensors", {})
        cameras = sensors.get("cameras", {}) if isinstance(sensors, dict) else {}
        if self.center_cam_ax is not None and ego_state is not None:
            _viz_draw_camera_center_ego(self.center_cam_ax, ego_state)
        elif self.center_cam_ax is not None:
            self.center_cam_ax.clear()
            self.center_cam_ax.set_facecolor("black")
            self.center_cam_ax.axis("off")
        for cam in CAMERA_ORDER:
            ax = self.cam_axes[cam]
            ax.clear()
            ax.axis("off")
            rel = cameras.get(cam, "") if isinstance(cameras, dict) else ""
            img = _safe_image(_resolve_path(str(rel), self.clip.clip_path))
            if img is None:
                ax.set_facecolor("#111111")
                ax.text(0.5, 0.5, "image missing", color="white", ha="center",
                        va="center", transform=ax.transAxes)
                continue
            ax.imshow(img)
            ax.text(0.02, 0.06, cam, transform=ax.transAxes, color="white", fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5))

    def _draw_lidar(self, frame: Dict[str, Any]) -> None:
        ax = self.lidar_ax
        if ax is None:
            return
        ax.clear()
        ax.set_aspect("equal")
        ax.set_facecolor("#111111")
        plot_range = self.lidar_range_m * 1.04
        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)
        ax.axis("off")
        ax.text(0.02, 0.97, "Lidar top-down", transform=ax.transAxes, color="white",
                fontsize=9, ha="left", va="top",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5))
        sensors = frame.get("sensors", {})
        lidar = sensors.get("lidar", {}) if isinstance(sensors, dict) else {}
        rel = lidar.get("point_cloud_path", "") if isinstance(lidar, dict) else ""
        path = _resolve_path(str(rel), self.clip.clip_path)
        xyz, intensity = None, None
        if path and os.path.isfile(path):
            try:
                xyz, intensity = load_pcd_points(path)
            except Exception:
                xyz = None
        if xyz is None or xyz.size == 0:
            ax.text(0.5, 0.5, "no point cloud for this frame", transform=ax.transAxes,
                    color="#888888", fontsize=9, ha="center", va="center")
            return
        r = np.hypot(xyz[0, :], xyz[1, :])
        mask = (
            np.isfinite(xyz).all(axis=0)
            & (r >= 1.5) & (r <= self.lidar_range_m)
            & (xyz[2, :] >= -2.5) & (xyz[2, :] <= 4.0)
        )
        xyz = xyz[:, mask]
        if intensity is not None:
            intensity = intensity[mask]
        if xyz.size == 0:
            return
        voxels = np.floor(xyz.T / 0.5).astype(np.int32)
        _, uniq = np.unique(voxels, axis=0, return_index=True)
        xyz = xyz[:, uniq]
        if intensity is not None:
            intensity = intensity[uniq]
        if xyz.shape[1] > self.max_lidar_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(xyz.shape[1], self.max_lidar_points, replace=False)
            xyz = xyz[:, idx]
            if intensity is not None:
                intensity = intensity[idx]
        if intensity is not None and intensity.size:
            lo, hi = np.percentile(intensity, [2, 98])
            colors = np.clip(intensity, lo, hi) if hi > lo else intensity
        else:
            colors = xyz[2, :]
        ax.scatter(xyz[0, :], xyz[1, :], s=1.0, c=colors, cmap="viridis",
                   alpha=0.75, linewidths=0)
        ax.scatter([0.0], [0.0], marker="*", s=80, c="#DE7061",
                   edgecolors="white", linewidths=0.5)

    def render_frame(self, frame_idx: int) -> None:
        frame = self.clip.frames[frame_idx]
        ego_state = self._load_ego(frame)
        ann = self._load_annotations(frame)
        self._draw_cameras(frame, ego_state)
        self.bev_ax.clear()
        if ego_state is not None:
            _viz_draw_bev_scene(
                self.bev_ax, ego_state, ann, self.static_map,
                mission_goal=frame.get("mission_goal"),
                metadata=self.clip.metadata,
                bev_range_m=self.bev_range_m,
                trajectory_dt_s=self.trajectory_dt_s,
            )
        if self.lidar_ax is not None:
            self._draw_lidar(frame)
        self.fig.subplots_adjust(
            left=FIG_MARGIN, right=1.0 - FIG_MARGIN,
            bottom=FIG_MARGIN, top=1.0 - FIG_MARGIN,
            wspace=0, hspace=0,
        )

    def save_png(self, frame_idx: int, output_path: str) -> None:
        self.render_frame(frame_idx)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self.fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight",
                         pad_inches=0, facecolor=self.fig.get_facecolor())

    def save_mp4(self, output_path: str, frame_indices: List[int], fps: int) -> None:
        require_ffmpeg()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self.fig.set_dpi(self.dpi)
        writer = FFMpegWriter(
            fps=max(fps, 1), codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        with writer.saving(self.fig, output_path, dpi=self.dpi):
            for output_idx, frame_idx in enumerate(frame_indices, start=1):
                self.render_frame(frame_idx)
                writer.grab_frame(facecolor=self.fig.get_facecolor())
                if output_idx % GC_COLLECT_EVERY_FRAMES == 0:
                    gc.collect()

    def close(self) -> None:
        self.static_map = None
        self.fig.clf()
        plt.close(self.fig)
        gc.collect()


def _select_frame_indices(
    total: int, start_index: int, end_index: int, stride: int, max_frames: int,
) -> List[int]:
    if total <= 0:
        return []
    start = max(0, start_index)
    end = total - 1 if end_index < 0 else min(end_index, total - 1)
    if end < start:
        return []
    indices = list(range(start, end + 1, max(1, stride)))
    if max_frames > 0:
        indices = indices[:max_frames]
    return indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="nuReasoning clip visualizer")
    parser.add_argument("--input-root", type=str, default="./dataset/data/train/part_1")
    parser.add_argument("--clip-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="./nureasoning_viz")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--frame-index", type=int, default=None,
                        help="Render a single composite PNG at this frame index")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="0 = one PNG frame, or all frames when --video")
    parser.add_argument("--max-clips", type=int, default=1, help="0 = all clips")
    parser.add_argument("--video", action="store_true", help="Write one MP4 per clip")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--no-lidar", action="store_true")
    parser.add_argument("--save-panels", action="store_true",
                        help="Also dump the per-reasoning-frame panel PNGs")
    parser.add_argument("--max-entries", type=int, default=0,
                        help="With --save-panels, limit reasoning files (0=all)")
    args = parser.parse_args()
    if not args.video and args.frame_index is None and args.max_frames == 0:
        args.max_frames = 1

    clips = _discover_clips(args.input_root, args.clip_path)
    if args.max_clips > 0:
        clips = clips[: args.max_clips]
    if not clips:
        raise RuntimeError(
            f"No clip folders with metadata.json under {os.path.abspath(args.input_root)}. "
            "If you still have .tar files, wait for the download to finish or run "
            "`python -m nureasoning.dataset.download --extract-only`."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    for clip in clips:
        if args.frame_index is not None:
            indices = [args.frame_index]
            if indices[0] < 0 or indices[0] >= len(clip.frames):
                print(f"Skipping {clip.clip_name}: frame {args.frame_index} out of range")
                continue
        else:
            indices = _select_frame_indices(
                len(clip.frames), args.start_index, args.end_index,
                args.stride, args.max_frames,
            )
        if not indices:
            print(f"Skipping {clip.clip_name}: no selected frames")
            continue

        viewer = NuReasoningOverview(
            clip, dpi=args.dpi, render_lidar=not args.no_lidar,
        )
        try:
            if args.video:
                out_path = os.path.join(args.output_dir, f"{clip.clip_name}.mp4")
                if args.overwrite or not os.path.isfile(out_path):
                    viewer.save_mp4(out_path, indices, fps=args.fps)
                    print(f"Saved {out_path} ({len(indices)} frames)")
                else:
                    print(f"Skipping existing {out_path}")
            else:
                for frame_idx in indices:
                    name = (f"{clip.clip_name}.png" if len(indices) == 1
                            else f"{clip.clip_name}_frame_{frame_idx:04d}.png")
                    out_path = os.path.join(args.output_dir, name)
                    if args.overwrite or not os.path.isfile(out_path):
                        viewer.save_png(frame_idx, out_path)
                        print(f"Saved {out_path}")
                    else:
                        print(f"Skipping existing {out_path}")
        finally:
            viewer.close()

    if args.save_panels:
        entries = _collect_entries(clips, args.max_entries)
        if not entries:
            print("No reasoning JSON files found for --save-panels.")
            return
        clip_map = {clip.clip_path: clip for clip in clips}
        renderer = NuReasoningRenderer(dpi=args.dpi)
        for i, entry in enumerate(entries, start=1):
            n = renderer.render_entry(
                entry, clip_map[entry.clip_path], args.output_dir, overwrite=args.overwrite,
            )
            if i % 5 == 0 or i == len(entries):
                print(f"[panels {i}/{len(entries)}] {entry.clip_name}/{entry.reasoning_name} ({n} images)")


if __name__ == "__main__":
    main()
