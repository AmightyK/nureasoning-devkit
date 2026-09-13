"""
Planning benchmark for evaluating VLA model trajectory predictions.

Computes five normalized sub-scores per clip:
  1. Collision        – BEV polygon overlap with ground-truth future objects
  2. Driveable area   – fraction of trajectory inside lane/road-block polygons
  3. Progress         – distance traveled along the ego route centerline
  4. Comfort          – penalises high lateral acceleration and longitudinal jerk
  5. Human likeness   – final-step L2 error to the ground-truth future trajectory

Final planning score: weighted combination of progress, comfort, and human
likeness, hard-gated by the binary collision and driveable-area checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
from tqdm import tqdm
from scipy.interpolate import PchipInterpolator as _PchipInterpolator
from scipy.signal import savgol_filter as _savgol_filter  

from nureasoning.common.clips import discover_clips
from nureasoning.common.pickle_io import load_pickle as _load_pickle
from nureasoning.common.schema import (
    Annotations,
    EgoState,
    nuReasoningStaticMap,
)
from nureasoning.visualization.plotting import visualize_clip_result, visualize_summary
from nureasoning.nuvla.trajectory_provider import (
    VLATrajectoryProvider,
    add_planning_prompt_arguments,
)
from nureasoning.planning.baselines import get_baseline

BASELINE_MODES = ("constant_velocity", "uniad", "diffusion_drive")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
KEY_FRAME_INDEX = 100
EGO_DIMENSIONS = {"l": 4.640, "w": 2.176, "h": 1.763}
VEHICLE_REAR_LENGTH = 0.79


def set_random_seed(seed: int) -> None:
    """Seed common RNGs used by the benchmark and optional VLA inference."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        logger.info("PyTorch not available; seeded Python and NumPy only")
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    # trajectory
    trajectory_dt_s: float = 0.1
    trajectory_steps: int = 51          # t=0.0s through t=5.0s, inclusive
    comfort_dt_s: float = 0.5           # derivative cadence for comfort checks

    # relative scoring weights for the non-binary metrics
    w_progress: float = 0.3
    w_comfort: float = 0.2
    w_human: float = 0.5

    # collision (no-at-fault)
    collision_iou_threshold: float = 0.0   # any BEV polygon overlap counts
    stopped_speed_threshold: float = 5e-2  # [m/s] for classifying stopped ego / stopped track
    agent_behind_cone_deg: float = 150.0   # full cone width (±75°) for is_agent_behind
    static_at_fault_score: float = 0.5     # non-agent at-fault collision → 0.5

    # comfort thresholds 
    max_lon_accel: float = 2.40            # m/s²   (upper bound)
    min_lon_accel: float = -4.05           # m/s²   (lower bound)
    max_abs_lat_accel: float = 4.89        # m/s²
    max_abs_mag_jerk: float = 8.37         # m/s³   (acceleration-magnitude jerk)
    max_abs_lon_jerk: float = 4.13         # m/s³   (longitudinal jerk)
    max_abs_yaw_accel: float = 1.93        # rad/s²
    max_abs_yaw_rate: float = 0.95         # rad/s

    # human-likeness normalisation from final-step L2 displacement error
    human_fde_full_score_m: float = 1.0   # score stays 1.0 at or below this FDE
    human_fde_threshold_m: float = 8.0    # score is zero at or above this FDE

    # driveable area buffer (metres) 
    driveable_area_buffer_m: float = 0.5

    # visualisation
    vis_range_m: float = 80.0

    # frame used for planning (clamped to the last frame on shorter clips)
    key_frame_index: int = KEY_FRAME_INDEX


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _box_corners_2d(
    cx: float, cy: float, half_l: float, half_w: float, yaw: float,
) -> List[Tuple[float, float]]:
    c, s = math.cos(yaw), math.sin(yaw)
    dx = [half_l, half_l, -half_l, -half_l]
    dy = [half_w, -half_w, -half_w, half_w]
    return [(cx + c * dx[i] - s * dy[i], cy + s * dx[i] + c * dy[i]) for i in range(4)]


def _ego_box_center_from_base_link(
    x: float,
    y: float,
    yaw: float,
    ego_length: float,
) -> Tuple[float, float]:
    center_offset = ego_length / 2.0 - VEHICLE_REAR_LENGTH
    return (
        x + math.cos(yaw) * center_offset,
        y + math.sin(yaw) * center_offset,
    )


def _make_box_polygon(
    cx: float, cy: float, half_l: float, half_w: float, yaw: float,
) -> Polygon:
    return Polygon(_box_corners_2d(cx, cy, half_l, half_w, yaw))


def _polygon_overlap(poly_a: Polygon, poly_b: Polygon) -> float:
    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)
    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)
    return float(poly_a.intersection(poly_b).area)


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Derivative helpers for the comfort metric 
# ---------------------------------------------------------------------------

def _phase_unwrap(headings: np.ndarray) -> np.ndarray:
    """
    remove 2π jumps so successive heading samples differ by at most π.
    """
    two_pi = 2.0 * np.pi
    headings = np.asarray(headings, dtype=np.float64)
    adjustments = np.zeros_like(headings)
    adjustments[..., 1:] = np.cumsum(
        np.round(np.diff(headings, axis=-1) / two_pi), axis=-1
    )
    return headings - two_pi * adjustments


def _approx_derivative(
    values: np.ndarray,
    dt: float,
    deriv_order: int = 1,
    poly_order: int = 2,
    window_length: int = 15,
) -> np.ndarray:
    """
    Approximate the ``deriv_order``-th derivative of a 1-D signal sampled at
    uniform spacing ``dt``.

    Uses a Savitzky-Golay filter
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[-1]
    if n < 2:
        return np.zeros_like(values)

    wl = min(window_length, n)
    if wl % 2 == 0:
        wl -= 1
    if wl >= max(poly_order + 1, 3):
        return _savgol_filter( 
            values,
            polyorder=poly_order,
            window_length=wl,
            deriv=deriv_order,
            delta=dt,
            axis=-1,
        )

    out = values
    for _ in range(deriv_order):
        out = np.gradient(out, dt, edge_order=1, axis=-1)
    return out


# ---------------------------------------------------------------------------
# No-at-fault collision helpers
# ---------------------------------------------------------------------------
_COLLISION_STOPPED_EGO = "STOPPED_EGO_COLLISION"
_COLLISION_STOPPED_TRACK = "STOPPED_TRACK_COLLISION"
_COLLISION_ACTIVE_FRONT = "ACTIVE_FRONT_COLLISION"
_COLLISION_ACTIVE_REAR = "ACTIVE_REAR_COLLISION"
_COLLISION_ACTIVE_LATERAL = "ACTIVE_LATERAL_COLLISION"


def _classify_tracked_object(category: str) -> str:
    """
    Return 'agent' for dynamic agents (vehicles, pedestrians, bicycles),
    'static' for everything else (cones, barriers, generic objects), or
    'ignore' for categories we should skip.
    """
    cat = (category or "").lower()
    if not cat or cat == "other" or cat.startswith("other."):
        return "ignore"
    if ("vehicle" in cat or "car" in cat or "truck" in cat or "bus" in cat
            or "motorcycle" in cat):
        return "agent"
    if "pedestrian" in cat or cat == "human":
        return "agent"
    if "bicycle" in cat or "bike" in cat or "cycle" in cat:
        return "agent"
    # traffic_cone, barrier, generic_object, czone_sign, ...
    return "static"


def _is_agent_in_cone(
    ego_x: float, ego_y: float, ego_yaw: float,
    track_x: float, track_y: float,
    cone_center_rel_yaw: float,
    cone_total_deg: float,
) -> bool:
    """
    Return True if the track center sits inside a cone anchored at the ego
    with a given half-angle. ``cone_center_rel_yaw`` is the cone's centerline
    relative to the ego yaw (0 = forward, π = backward).
    """
    dx = track_x - ego_x
    dy = track_y - ego_y
    if dx == 0.0 and dy == 0.0:
        return True
    bearing = math.atan2(dy, dx)
    rel = _wrap_to_pi(bearing - (ego_yaw + cone_center_rel_yaw))
    half = math.radians(cone_total_deg) / 2.0
    return abs(rel) <= half


def _is_agent_behind(
    ego_x: float, ego_y: float, ego_yaw: float,
    track_x: float, track_y: float,
    cone_total_deg: float = 150.0,
) -> bool:
    return _is_agent_in_cone(ego_x, ego_y, ego_yaw, track_x, track_y,
                             cone_center_rel_yaw=math.pi,
                             cone_total_deg=cone_total_deg)


def _front_bumper_intersects(
    ego_polygon: Polygon,
    ego_yaw: float,
    tracked_object_polygon: Polygon,
) -> bool:
    """
    Return True when the ego's front edge intersects the tracked object.

    The benchmark's ego box is created from a generic [x, y, yaw] pose, so we
    recover the front bumper geometrically from the polygon vertices rather than
    relying on a fixed exterior vertex ordering.
    """
    coords = list(ego_polygon.exterior.coords[:-1])
    if len(coords) < 2:
        return False

    heading = np.array([math.cos(ego_yaw), math.sin(ego_yaw)], dtype=np.float64)
    front_points = sorted(
        coords,
        key=lambda pt: pt[0] * heading[0] + pt[1] * heading[1],
        reverse=True,
    )[:2]
    return LineString(front_points).intersects(tracked_object_polygon)


def _get_collision_type(
    ego_x: float, ego_y: float, ego_yaw: float, ego_speed: float,
    ego_polygon: Polygon,
    tracked_object_polygon: Polygon,
    track_speed: float,
    config: "BenchmarkConfig",
) -> str:
    """
    Classify a collision
    """
    if ego_speed <= config.stopped_speed_threshold:
        return _COLLISION_STOPPED_EGO
    if track_speed <= config.stopped_speed_threshold:
        return _COLLISION_STOPPED_TRACK

    track_center = tracked_object_polygon.centroid
    if _is_agent_behind(ego_x, ego_y, ego_yaw, float(track_center.x), float(track_center.y),
                        config.agent_behind_cone_deg):
        return _COLLISION_ACTIVE_REAR
    if _front_bumper_intersects(ego_polygon, ego_yaw, tracked_object_polygon):
        return _COLLISION_ACTIVE_FRONT
    return _COLLISION_ACTIVE_LATERAL


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_metadata(clip_path: str) -> Dict[str, Any]:
    with open(os.path.join(clip_path, "metadata.json"), "r") as f:
        return json.load(f)


def select_key_frame_idx(
    frames: List[Dict[str, Any]],
    key_frame_index: int = KEY_FRAME_INDEX,
) -> int:
    """Select the planning key frame, clamped for shorter clips."""
    if not frames:
        return 0
    return min(max(int(key_frame_index), 0), len(frames) - 1)


# ---------------------------------------------------------------------------
# Build driveable-area geometry
# ---------------------------------------------------------------------------

def build_driveable_area(
    static_map: nuReasoningStaticMap,
    buffer_m: float = 0.5,
) -> Optional[Polygon | MultiPolygon]:
    """Return a single (Multi)Polygon representing the driveable surface."""
    polygons: List[Polygon] = []

    for lane in static_map.lanes:
        if lane.polygon and len(lane.polygon) >= 3:
            p = Polygon([(pt[0], pt[1]) for pt in lane.polygon])
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty:
                polygons.append(p)

    for inter in static_map.intersections:
        if inter.geometry and len(inter.geometry) >= 3:
            p = Polygon([(pt[0], pt[1]) for pt in inter.geometry])
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty:
                polygons.append(p)

    for road_block in static_map.road_blocks:
        if road_block.geometry and len(road_block.geometry) >= 3:
            p = Polygon([(pt[0], pt[1]) for pt in road_block.geometry])
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty:
                polygons.append(p)
  

    if not polygons:
        return None

    driveable = unary_union(polygons)
    if buffer_m > 0:
        driveable = driveable.buffer(buffer_m)
    return driveable


def build_lane_polygons(static_map: nuReasoningStaticMap) -> List[Polygon]:
    """
    Return a list of individual lane polygons (not unioned).

    Used for the multi-lane check in the no-at-fault collision metric:
    the ego is considered to be in multiple lanes when corners of its
    bounding box fall into more than one lane polygon and no single lane
    polygon contains all four corners.
    """
    lane_polys: List[Polygon] = []
    for lane in static_map.lanes:
        if lane.polygon and len(lane.polygon) >= 3:
            try:
                p = Polygon([(pt[0], pt[1]) for pt in lane.polygon])
                if not p.is_valid:
                    p = p.buffer(0)
                if not p.is_empty:
                    lane_polys.append(p)
            except Exception:
                pass
    return lane_polys


# ---------------------------------------------------------------------------
# Build route line from mission_goal
# ---------------------------------------------------------------------------

_ROUTE_INTERP_SPACING_M = 1.0
_ROUTE_INTERP_MIN_SAMPLES = 64
_ROUTE_INTERP_MAX_SAMPLES = 3000

def _dedupe_route_xy(xy: np.ndarray, min_step_m: float = 1e-4) -> np.ndarray:
    """Drop consecutive duplicates so spline knots remain strictly increasing."""
    if xy.shape[0] == 0:
        return xy

    min_step_sq = min_step_m * min_step_m
    kept: List[np.ndarray] = [xy[0]]
    for idx in range(1, xy.shape[0]):
        delta = xy[idx] - kept[-1]
        if float(delta @ delta) >= min_step_sq:
            kept.append(xy[idx])
    return np.stack(kept, axis=0)


def _build_interpolated_route_xy(route_path: Any) -> Optional[np.ndarray]:
    pts: List[tuple[float, float]] = []
    for p in route_path:
        if isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
            pts.append((float(p[0]), float(p[1])))

    if len(pts) < 2:
        return None

    xy = _dedupe_route_xy(np.asarray(pts, dtype=np.float64))
    if xy.shape[0] < 2:
        return None

    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    route_length = float(segment_lengths.sum())
    if route_length < 1e-6:
        return xy

    chord = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if not np.all(np.diff(chord) > 0.0):
        return xy

    num_samples = int(
        max(
            _ROUTE_INTERP_MIN_SAMPLES,
            min(
                _ROUTE_INTERP_MAX_SAMPLES,
                math.ceil(route_length / _ROUTE_INTERP_SPACING_M) + 1,
            ),
        )
    )
    sample_chord = np.linspace(0.0, route_length, num_samples)
    interp_x = _PchipInterpolator(chord, xy[:, 0])
    interp_y = _PchipInterpolator(chord, xy[:, 1])
    return np.column_stack((
        interp_x(sample_chord),
        interp_y(sample_chord),
    ))


def build_route_line(mission_goal: Any) -> Optional[LineString]:
    if mission_goal is None:
        return None
    if isinstance(mission_goal, dict):
        route_path = mission_goal.get("route_path", [])
    else:
        route_path = getattr(mission_goal, "route_path", [])

    interpolated_xy = _build_interpolated_route_xy(route_path)
    if interpolated_xy is None or len(interpolated_xy) < 2:
        return None

    return LineString(interpolated_xy)


# ---------------------------------------------------------------------------
# Retrieve future object states from subsequent frames
# ---------------------------------------------------------------------------

def load_future_object_states(
    clip_path: str,
    frames: List[Dict],
    key_frame_idx: int,
    n_future: int,
    dt_s: float = 0.5,
) -> List[Annotations]:
    """Load future annotation pickles sampled at the benchmark trajectory interval."""
    future_anns: List[Annotations] = []
    if not frames or n_future <= 0:
        return future_anns

    frame_dt_s = None
    for i in range(1, len(frames)):
        t0 = frames[i - 1].get("relative_time_s")
        t1 = frames[i].get("relative_time_s")
        if isinstance(t0, (int, float)) and isinstance(t1, (int, float)) and t1 > t0:
            frame_dt_s = float(t1) - float(t0)
            break

    if frame_dt_s is not None and frame_dt_s > 0:
        frame_stride = max(1, int(round(dt_s / frame_dt_s)))
    else:
        frame_stride = max(1, int(round(dt_s * 10.0)))

    for step in range(1, n_future + 1):
        fi = key_frame_idx + step * frame_stride
        if fi >= len(frames):
            break
        ann_rel = frames[fi].get("annotations")
        if not ann_rel:
            continue
        ann_path = os.path.join(clip_path, ann_rel)
        if os.path.isfile(ann_path):
            future_anns.append(_load_pickle(ann_path))
    return future_anns


def _normalize_trajectory_to_benchmark_grid(
    trajectory: np.ndarray,
    ego_pose: Dict[str, Any],
    config: BenchmarkConfig,
) -> np.ndarray:
    """Interpolate a trajectory onto t=0, dt, ..., horizon for scoring."""
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] < 3 or len(traj) == 0:
        raise ValueError("trajectory must be a non-empty (N,3) array")
    if not np.all(np.isfinite(traj[:, :3])):
        raise ValueError("trajectory contains non-finite values")

    target_dt = max(config.trajectory_dt_s, 1e-6)
    target_times = np.arange(config.trajectory_steps, dtype=np.float64) * target_dt
    horizon_s = float(target_times[-1]) if len(target_times) else 0.0

    current_pose = np.array([
        float(ego_pose.get("x", 0.0)),
        float(ego_pose.get("y", 0.0)),
        float(ego_pose.get("yaw", 0.0)),
    ], dtype=np.float64)

    first_xy_error = float(np.linalg.norm(traj[0, :2] - current_pose[:2]))
    first_yaw_error = abs(_wrap_to_pi(float(traj[0, 2] - current_pose[2])))
    includes_current = first_xy_error < 0.25 and first_yaw_error < 0.25

    if includes_current:
        source_times = np.linspace(0.0, horizon_s, len(traj), dtype=np.float64)
        source_traj = traj[:, :3]
    else:
        source_dt = horizon_s / max(len(traj), 1)
        source_times = np.arange(1, len(traj) + 1, dtype=np.float64) * source_dt
        source_times = np.concatenate([[0.0], source_times])
        source_traj = np.vstack([current_pose, traj[:, :3]])

    if len(source_times) < 2:
        return np.repeat(current_pose[None, :], config.trajectory_steps, axis=0)

    x = np.interp(target_times, source_times, source_traj[:, 0])
    y = np.interp(target_times, source_times, source_traj[:, 1])
    yaw_unwrapped = np.unwrap(source_traj[:, 2])
    yaw = np.interp(target_times, source_times, yaw_unwrapped)
    yaw = np.array([_wrap_to_pi(float(v)) for v in yaw], dtype=np.float64)
    return np.stack([x, y, yaw], axis=1)


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------

def score_collision(
    planned_traj: np.ndarray,
    ego_state: EgoState,
    current_annotations: Annotations,
    future_annotations: List[Annotations],
    driveable_area: Optional[Polygon | MultiPolygon],
    lane_polygons: List[Polygon],
    config: BenchmarkConfig,
) -> Tuple[float, Dict[str, Any]]:
    """
    No-at-fault collision metric

    The ego polygon is swept along the planned trajectory and checked for
    BEV overlap with ground-truth annotation boxes at the corresponding
    timestep. For each first-time overlap the collision is classified:

      * ACTIVE_FRONT / STOPPED_TRACK   -> always at fault
      * ACTIVE_LATERAL                 -> at fault only when the ego is
                                          simultaneously in multiple lanes
                                          or off the driveable area
      * ACTIVE_REAR / STOPPED_EGO      -> never at fault
      * red-light / already-collided   -> skipped

    At-fault collisions with dynamic agents (vehicles, pedestrians, bicycles)
    drive the score to ``0.0``; at-fault collisions with static objects
    (cones, barriers, generic objects) drive it down to
    ``config.static_at_fault_score``. Non-at-fault
    collisions are tracked so the same object does not penalise later
    timesteps.

    Returns (score, details). ``score`` ∈ {0.0, 0.5, 1.0}.
    """
    ego_dims = EGO_DIMENSIONS
    half_l = ego_dims["l"] / 2.0
    half_w = ego_dims["w"] / 2.0

    dt = max(config.trajectory_dt_s, 1e-6)
    n_steps = len(planned_traj)

    # --- ego speed per timestep (t=0 from ego_state, rest from trajectory diff) ---
    ego_speeds = np.zeros(n_steps, dtype=np.float64)
    if n_steps > 0:
        vel = ego_state.velocity if isinstance(ego_state.velocity, dict) else {}
        vx0 = float(vel.get("vx", 0.0))
        vy0 = float(vel.get("vy", 0.0))
        ego_speeds[0] = math.hypot(vx0, vy0)
    for t in range(1, n_steps):
        dx = float(planned_traj[t, 0] - planned_traj[t - 1, 0])
        dy = float(planned_traj[t, 1] - planned_traj[t - 1, 1])
        ego_speeds[t] = math.hypot(dx, dy) / dt

    # prepare driveable-area geometry for fast ``contains`` checks
    prepared_drivable = prep(driveable_area) if driveable_area is not None else None

    all_ann_frames = [current_annotations] + list(future_annotations)

    score = 1.0
    collided_track_ids: set = set()  # tracks we have already "collided" with
    first_at_fault_step = -1
    first_at_fault_token = ""
    first_at_fault_type = ""
    first_at_fault_target = ""  # "agent" or "static"
    all_overlaps: List[Dict[str, Any]] = []

    for t_idx in range(n_steps):
        px, py, pyaw = planned_traj[t_idx]
        px = float(px); py = float(py); pyaw = float(pyaw)
        ego_cx, ego_cy = _ego_box_center_from_base_link(px, py, pyaw, ego_dims["l"])
        ego_poly = _make_box_polygon(ego_cx, ego_cy, half_l, half_w, pyaw)
        corners = _box_corners_2d(ego_cx, ego_cy, half_l, half_w, pyaw)

        ann_idx = min(t_idx, len(all_ann_frames) - 1)
        ann = all_ann_frames[ann_idx]

        # -- ego area flags (computed lazily, only when needed) --
        ego_area_computed = False
        ego_in_nondrivable = False
        ego_in_multiple_lanes = False

        def _compute_ego_area() -> None:
            nonlocal ego_area_computed, ego_in_nondrivable, ego_in_multiple_lanes
            if ego_area_computed:
                return
            # non-drivable area: any corner outside the drivable union
            if prepared_drivable is not None:
                for cx, cy in corners:
                    if not prepared_drivable.contains(Point(cx, cy)):
                        ego_in_nondrivable = True
                        break
            # multiple lanes: >1 lane polygon contains some corner AND no
            # single lane polygon contains all four corners
            if lane_polygons:
                lanes_hit = 0
                any_lane_contains_all = False
                for lp in lane_polygons:
                    contains_flags = [lp.contains(Point(cx, cy)) for cx, cy in corners]
                    any_contain = any(contains_flags)
                    if any_contain:
                        lanes_hit += 1
                    if all(contains_flags):
                        any_lane_contains_all = True
                        break
                if lanes_hit > 1 and not any_lane_contains_all:
                    ego_in_multiple_lanes = True
            ego_area_computed = True

        for obj in ann.objects:
            target_kind = _classify_tracked_object(obj.category)
            if target_kind == "ignore":
                continue

            ol = float(obj.dimensions.get("l", 0.0))
            ow = float(obj.dimensions.get("w", 0.0))
            if ol <= 0.0 or ow <= 0.0:
                continue

            token = obj.track_token or ""
            # Skip tracks we have already processed as non-at-fault; the
            # reference also filters red-light polygons here, which we do
            # not have in the annotations.
            if token and token in collided_track_ids:
                continue

            ox = float(obj.pose.get("x", 0.0))
            oy = float(obj.pose.get("y", 0.0))
            oyaw = float(obj.pose.get("yaw", 0.0))
            obj_poly = _make_box_polygon(ox, oy, ol / 2.0, ow / 2.0, oyaw)

            overlap = _polygon_overlap(ego_poly, obj_poly)
            if overlap <= config.collision_iou_threshold:
                continue

            # --- classify the collision ---
            obj_vel = obj.velocity if isinstance(obj.velocity, dict) else {}
            track_speed = math.hypot(
                float(obj_vel.get("vx", 0.0)),
                float(obj_vel.get("vy", 0.0)),
            )
            ctype = _get_collision_type(
                ego_cx, ego_cy, pyaw, float(ego_speeds[t_idx]),
                ego_poly,
                obj_poly,
                track_speed,
                config,
            )

            at_fault = False
            if ctype in (_COLLISION_ACTIVE_FRONT, _COLLISION_STOPPED_TRACK):
                at_fault = True
            elif ctype == _COLLISION_ACTIVE_LATERAL:
                _compute_ego_area()
                if ego_in_multiple_lanes or ego_in_nondrivable:
                    at_fault = True
            # ACTIVE_REAR and STOPPED_EGO are never at fault

            all_overlaps.append({
                "t": t_idx,
                "token": token,
                "category": obj.category,
                "collision_type": ctype,
                "at_fault": at_fault,
                "target_kind": target_kind,
            })

            if at_fault:
                this_score = 0.0 if target_kind == "agent" else config.static_at_fault_score
                if this_score < score:
                    score = this_score
                    first_at_fault_step = t_idx
                    first_at_fault_token = token
                    first_at_fault_type = ctype
                    first_at_fault_target = target_kind

                if token:
                    collided_track_ids.add(token)
            else:
                # not at fault -> mark track as already-collided, continue
                if token:
                    collided_track_ids.add(token)

    details = {
        "has_at_fault_collision": score < 1.0,
        "score": score,
        "first_at_fault_step": first_at_fault_step,
        "first_at_fault_object": first_at_fault_token,
        "first_at_fault_type": first_at_fault_type,
        "first_at_fault_target": first_at_fault_target,
        "num_overlaps": len(all_overlaps),
        "num_at_fault_overlaps": sum(1 for o in all_overlaps if o["at_fault"]),
        # keep legacy field names so existing visualisation keeps working
        "has_collision": score < 1.0,
        "collision_timestep": first_at_fault_step,
        "collision_object": first_at_fault_token,
    }
    return float(score), details


def score_driveable_area(
    planned_traj: np.ndarray,
    ego_dims: Dict[str, float],
    driveable_area: Optional[Polygon | MultiPolygon],
    config: BenchmarkConfig,
) -> Tuple[float, Dict[str, Any]]:
    """
    Check whether all four ego-box corners at every trajectory point lie
    within the driveable area.

    Returns (score, details).  score ∈ {0, 1}: 1 if fully compliant.
    """
    if driveable_area is None:
        return 1.0, {"fraction_in": 1.0, "no_map": True}

    half_l = ego_dims["l"] / 2.0
    half_w = ego_dims["w"] / 2.0

    prepared = prep(driveable_area)
    n_points = len(planned_traj)
    minx, miny, maxx, maxy = driveable_area.bounds
    in_count = 0
    evaluated_count = 0
    skipped_out_of_map = 0
    first_violation_step = -1

    for t_idx in range(n_points):
        px, py, pyaw = planned_traj[t_idx]
        ego_cx, ego_cy = _ego_box_center_from_base_link(
            float(px),
            float(py),
            float(pyaw),
            ego_dims["l"],
        )
        corners = _box_corners_2d(ego_cx, ego_cy, half_l, half_w, float(pyaw))

        # If the whole ego box is outside the known map extent, skip this step
        # instead of treating missing map coverage as a driveable-area failure.
        in_map_region = any(
            minx <= cx <= maxx and miny <= cy <= maxy
            for cx, cy in corners
        )
        if not in_map_region:
            skipped_out_of_map += 1
            continue

        evaluated_count += 1
        all_in = all(prepared.contains(Point(cx, cy)) for cx, cy in corners)
        if all_in:
            in_count += 1
        elif first_violation_step < 0:
            first_violation_step = t_idx

    fraction = in_count / evaluated_count if evaluated_count > 0 else 1.0
    is_compliant = (in_count == evaluated_count)
    score = 1.0 if is_compliant else 0.0

    return score, {
        "fraction_in": fraction,
        "is_compliant": is_compliant,
        "first_violation_step": first_violation_step,
        "evaluated_steps": evaluated_count,
        "skipped_out_of_map_steps": skipped_out_of_map,
    }


def score_progress(
    planned_traj: np.ndarray,
    route_line: Optional[LineString],
    gt_traj: Optional[np.ndarray],
    config: BenchmarkConfig,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute how far along the route the planned trajectory progresses,
    normalised by the GT progress.

    score = clamp(planned_progress / gt_progress, 0, 1)
    If no route is available, or if the route appears not to cover the GT
    future, fall back to Euclidean displacement ratio.
    """
    start_xy = planned_traj[0, :2]
    end_xy = planned_traj[-1, :2]

    def _euclidean_progress() -> Tuple[float, float]:
        planned = float(np.linalg.norm(end_xy - start_xy))
        if gt_traj is not None and len(gt_traj) >= 2:
            gt = float(np.linalg.norm(gt_traj[-1, :2] - gt_traj[0, :2]))
        else:
            gt = planned
        return planned, gt

    progress_mode = "route"
    coverage_failure = False
    route_end_gap_m = 0.0
    route_end_margin_m = 0.0

    if route_line is not None and not route_line.is_empty:
        proj_start = route_line.project(Point(start_xy))
        proj_end = route_line.project(Point(end_xy))
        planned_progress = max(0.0, proj_end - proj_start)

        if gt_traj is not None and len(gt_traj) >= 2:
            gt_end = gt_traj[-1, :2]
            proj_gt_end = route_line.project(Point(gt_end))
            gt_progress = max(0.0, proj_gt_end - proj_start)

            # If GT projects to the very end of the provided route but is still
            # meaningfully away from that route, use GT as the progress reference.
            route_end_gap_m = float(route_line.distance(Point(gt_end)))
            route_end_margin_m = float(route_line.length - proj_gt_end)
            gt_euclidean_progress = float(np.linalg.norm(gt_end - gt_traj[0, :2]))
            coverage_failure = (
                route_end_margin_m <= 1.0
                and route_end_gap_m > 2.0
                and gt_euclidean_progress > gt_progress + 2.0
            )
            if coverage_failure:
                planned_progress, gt_progress = _euclidean_progress()
                progress_mode = "euclidean_route_coverage_failure"
        else:
            gt_progress = planned_progress
    else:
        planned_progress, gt_progress = _euclidean_progress()
        progress_mode = "euclidean_no_route"

    if gt_progress < 0.1:
        score = 1.0
    else:
        score = float(np.clip(planned_progress / gt_progress, 0.0, 1.0))

    return score, {
        "planned_m": planned_progress,
        "gt_m": gt_progress,
        "mode": progress_mode,
    }


def score_comfort(
    planned_traj: np.ndarray,
    config: BenchmarkConfig,
) -> Tuple[float, Dict[str, Any]]:
    """
    Comfort metric

    The planned trajectory is numerically differentiated to recover
    acceleration, jerk and yaw derivatives; six sub-metrics are then
    checked against fixed bounds:

      1. longitudinal acceleration    ∈ (min_lon_accel, max_lon_accel)
      2. lateral acceleration         |·| < max_abs_lat_accel
      3. magnitude jerk               |·| < max_abs_mag_jerk
      4. longitudinal jerk            |·| < max_abs_lon_jerk
      5. yaw acceleration             |·| < max_abs_yaw_accel
      6. yaw rate                     |·| < max_abs_yaw_rate

    Score is ``1.0`` iff every sub-metric stays within its bound across the
    entire horizon, otherwise ``0.0``.
    """
    base_dt = max(config.trajectory_dt_s, 1e-6)
    comfort_dt = max(float(getattr(config, "comfort_dt_s", base_dt)), base_dt)
    stride = max(1, int(round(comfort_dt / base_dt)))
    traj_for_comfort = planned_traj[::stride]
    if len(traj_for_comfort) > 0 and not np.array_equal(traj_for_comfort[-1], planned_traj[-1]):
        traj_for_comfort = np.vstack([traj_for_comfort, planned_traj[-1]])

    dt = base_dt * stride
    n = traj_for_comfort.shape[0]
    if n < 3:
        return 1.0, {"note": "trajectory too short", "is_comfortable": True}

    x = traj_for_comfort[:, 0].astype(np.float64)
    y = traj_for_comfort[:, 1].astype(np.float64)
    yaw = _phase_unwrap(traj_for_comfort[:, 2].astype(np.float64))

    # Global-frame acceleration from positions (second derivative).
    ax = _approx_derivative(x, dt, deriv_order=2, poly_order=2, window_length=8)
    ay = _approx_derivative(y, dt, deriv_order=2, poly_order=2, window_length=8)

    # Rotate into ego frame using the instantaneous heading.
    c = np.cos(yaw)
    s = np.sin(yaw)
    lon_accel = ax * c + ay * s
    lat_accel = -ax * s + ay * c

    # Magnitude-acceleration and jerks.
    mag_accel = np.hypot(ax, ay)
    mag_jerk = _approx_derivative(mag_accel, dt, deriv_order=1,
                                  poly_order=2, window_length=15)
    lon_jerk = _approx_derivative(lon_accel, dt, deriv_order=1,
                                  poly_order=2, window_length=15)

    # Yaw rate and yaw acceleration from unwrapped heading.
    yaw_rate = _approx_derivative(yaw, dt, deriv_order=1,
                                  poly_order=2, window_length=15)
    yaw_accel = _approx_derivative(yaw, dt, deriv_order=2,
                                   poly_order=3, window_length=15)

    def _within(values: np.ndarray, lo: float, hi: float) -> bool:
        return bool(np.all((values > lo) & (values < hi)))

    ok_lon_accel = _within(lon_accel, config.min_lon_accel, config.max_lon_accel)
    ok_lat_accel = _within(lat_accel, -config.max_abs_lat_accel, config.max_abs_lat_accel)
    ok_mag_jerk = _within(mag_jerk, -config.max_abs_mag_jerk, config.max_abs_mag_jerk)
    ok_lon_jerk = _within(lon_jerk, -config.max_abs_lon_jerk, config.max_abs_lon_jerk)
    ok_yaw_accel = _within(yaw_accel, -config.max_abs_yaw_accel, config.max_abs_yaw_accel)
    ok_yaw_rate = _within(yaw_rate, -config.max_abs_yaw_rate, config.max_abs_yaw_rate)

    is_comfortable = (
        ok_lon_accel and ok_lat_accel and ok_mag_jerk
        and ok_lon_jerk and ok_yaw_accel and ok_yaw_rate
    )
    score = 1.0 if is_comfortable else 0.0

    details: Dict[str, Any] = {
        "is_comfortable": is_comfortable,
        "comfort_dt_s": float(dt),
        "comfort_num_points": int(n),
        # aggregate signal ranges (helpful for debugging / visualisation)
        "max_lon_accel": float(np.max(lon_accel)),
        "min_lon_accel": float(np.min(lon_accel)),
        "max_abs_lat_accel": float(np.max(np.abs(lat_accel))),
        "max_abs_mag_jerk": float(np.max(np.abs(mag_jerk))),
        "max_abs_lon_jerk": float(np.max(np.abs(lon_jerk))),
        "max_abs_yaw_rate": float(np.max(np.abs(yaw_rate))),
        "max_abs_yaw_accel": float(np.max(np.abs(yaw_accel))),
        # per-sub-metric pass / fail booleans
        "ok_lon_accel": ok_lon_accel,
        "ok_lat_accel": ok_lat_accel,
        "ok_mag_jerk": ok_mag_jerk,
        "ok_lon_jerk": ok_lon_jerk,
        "ok_yaw_accel": ok_yaw_accel,
        "ok_yaw_rate": ok_yaw_rate,
        # legacy keys preserved for the visualisation module
        "max_lat_accel": float(np.max(np.abs(lat_accel))),
        "max_lon_jerk": float(np.max(np.abs(lon_jerk))),
        "max_yaw_rate": float(np.max(np.abs(yaw_rate))),
    }
    return score, details


def score_human_likeness(
    planned_traj: np.ndarray,
    gt_traj: np.ndarray,
    config: BenchmarkConfig,
) -> Tuple[float, Dict[str, Any]]:
    """
    Final-step L2 displacement error between planned and GT trajectories,
    mapped to [0, 1]. The score stays 1.0 for small FDE, smoothly decreases
    between the configured FDE bounds, and becomes 0.0 beyond the threshold.
    ADE is still reported for analysis.
    """
    n = min(len(planned_traj), len(gt_traj))
    if n == 0:
        return 0.0, {"ade": float("inf"), "fde": float("inf")}

    diffs = planned_traj[:n, :2] - gt_traj[:n, :2]
    per_step_err = np.linalg.norm(diffs, axis=1)
    ade = float(np.mean(per_step_err))
    fde = float(per_step_err[-1])

    full_score_fde = config.human_fde_full_score_m
    zero_score_fde = config.human_fde_threshold_m
    passes_fde_threshold = fde <= zero_score_fde
    if fde <= full_score_fde:
        score = 1.0
    elif fde >= zero_score_fde:
        score = 0.0
    else:
        t = float((fde - full_score_fde) / (zero_score_fde - full_score_fde))
        smoothstep = t * t * (3.0 - 2.0 * t)
        score = 1.0 - smoothstep
    return score, {
        "ade": ade,
        "fde": fde,
        "fde_full_score_m": config.human_fde_full_score_m,
        "fde_threshold_m": config.human_fde_threshold_m,
        "passes_fde_threshold": passes_fde_threshold,
    }


# ---------------------------------------------------------------------------
# Per-clip evaluation
# ---------------------------------------------------------------------------

@dataclass
class ClipResult:
    clip_path: str
    clip_name: str
    key_frame_idx: int

    s_collision: float = 0.0
    s_driveable: float = 0.0
    s_progress: float = 0.0
    s_comfort: float = 0.0
    s_human: float = 0.0
    planning_score: float = 0.0

    details_collision: Dict[str, Any] = field(default_factory=dict)
    details_driveable: Dict[str, Any] = field(default_factory=dict)
    details_progress: Dict[str, Any] = field(default_factory=dict)
    details_comfort: Dict[str, Any] = field(default_factory=dict)
    details_human: Dict[str, Any] = field(default_factory=dict)

    planned_traj: Optional[np.ndarray] = None
    gt_traj: Optional[np.ndarray] = None
    ego_pose: Optional[Dict[str, float]] = None
    error: Optional[str] = None


def evaluate_clip(
    clip_path: str,
    config: BenchmarkConfig,
    planned_trajectory: Optional[np.ndarray] = None,
) -> ClipResult:
    """
    Evaluate a single clip.

    Parameters
    ----------
    clip_path : path to the clip directory
    config : benchmark configuration
    planned_trajectory : (N, 3) array of [x, y, yaw] in global coords.
        If None, uses GT trajectory_future as a baseline / sanity check.
    """
    clip_name = os.path.basename(clip_path)
    meta = _load_metadata(clip_path)
    frames = meta.get("frames", [])

    if len(frames) < 3:
        return ClipResult(
            clip_path=clip_path, clip_name=clip_name, key_frame_idx=0,
            error="too few frames",
        )

    # --- pick key frame ---
    key_idx = select_key_frame_idx(frames, config.key_frame_index)

    frame = frames[key_idx]

    # --- load data ---
    ego_state: EgoState = _load_pickle(os.path.join(clip_path, frame["ego_state"]))
    annotations: Annotations = _load_pickle(os.path.join(clip_path, frame["annotations"]))

    # GT trajectory
    gt_raw = ego_state.trajectory_future
    if not gt_raw or len(gt_raw) < 2:
        return ClipResult(
            clip_path=clip_path, clip_name=clip_name, key_frame_idx=key_idx,
            error="no GT trajectory_future",
        )
    gt_traj = _normalize_trajectory_to_benchmark_grid(
        np.array(gt_raw, dtype=np.float64),
        ego_state.pose,
        config,
    )

    # Planned trajectory: use provided or fall back to GT
    if planned_trajectory is not None:
        plan_traj = _normalize_trajectory_to_benchmark_grid(
            np.array(planned_trajectory, dtype=np.float64),
            ego_state.pose,
            config,
        )
    else:
        plan_traj = gt_traj.copy()

    # --- map ---
    map_path = os.path.join(clip_path, meta.get("map_annotation", "map.pkl"))
    static_map: Optional[nuReasoningStaticMap] = None
    if os.path.isfile(map_path):
        static_map = _load_pickle(map_path)

    driveable_area = None
    lane_polygons: List[Polygon] = []
    if static_map is not None:
        driveable_area = build_driveable_area(static_map, config.driveable_area_buffer_m)
        lane_polygons = build_lane_polygons(static_map)

    # --- route ---
    mission_goal = frame.get("mission_goal")
    route_line = build_route_line(mission_goal)

    # --- future annotations ---
    n_future = max(config.trajectory_steps - 1, 0)
    future_anns = load_future_object_states(
        clip_path,
        frames,
        key_idx,
        n_future,
        dt_s=config.trajectory_dt_s,
    )

    # --- compute scores ---
    s_col, d_col = score_collision(
        plan_traj, ego_state, annotations, future_anns,
        driveable_area, lane_polygons, config,
    )
    s_da, d_da = score_driveable_area(plan_traj, EGO_DIMENSIONS, driveable_area, config)
    s_prog, d_prog = score_progress(plan_traj, route_line, gt_traj, config)
    s_comf, d_comf = score_comfort(plan_traj, config)
    s_hum, d_hum = score_human_likeness(plan_traj, gt_traj, config)

    # --- Planning score aggregation ---
    multiplicative_gate = s_col * s_da
    weighted_num = (
        config.w_progress * s_prog
        + config.w_comfort * s_comf
        + config.w_human * s_hum
    )
    weighted_den = config.w_progress + config.w_comfort + config.w_human
    weighted_score = weighted_num / weighted_den if weighted_den > 0 else 0.0
    planning_score = multiplicative_gate * weighted_score

    return ClipResult(
        clip_path=clip_path,
        clip_name=clip_name,
        key_frame_idx=key_idx,
        s_collision=s_col,
        s_driveable=s_da,
        s_progress=s_prog,
        s_comfort=s_comf,
        s_human=s_hum,
        planning_score=planning_score,
        details_collision=d_col,
        details_driveable=d_da,
        details_progress=d_prog,
        details_comfort=d_comf,
        details_human=d_hum,
        planned_traj=plan_traj,
        gt_traj=gt_traj,
        ego_pose=ego_state.pose,
    )


# ---------------------------------------------------------------------------
# Full benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSummary:
    total_clips: int
    evaluated_clips: int
    failed_clips: int
    mean_collision: float
    mean_driveable: float
    mean_progress: float
    mean_comfort: float
    mean_human: float
    mean_ade_m: float
    mean_fde_m: float
    mean_planning_score: float
    results: List[ClipResult] = field(default_factory=list)


def run_benchmark(
    data_root: str,
    config: BenchmarkConfig,
    max_clips: int = 0,
    trajectory_provider=None,
) -> BenchmarkSummary:
    """
    Evaluate all clips under *data_root*.

    Parameters
    ----------
    trajectory_provider : optional callable(clip_path, key_frame_idx, ego_state)
        → np.ndarray (N,3).  When None, GT trajectory is used (sanity-check mode).
    """
    clips = discover_clips(data_root)
    if max_clips > 0:
        clips = clips[:max_clips]
    logger.info("Found %d clips under %s", len(clips), data_root)

    results: List[ClipResult] = []
    for clip_path in tqdm(clips, desc="Evaluating clips"):
        try:
            # Allow external trajectory provider
            planned = None
            if trajectory_provider is not None:
                meta = _load_metadata(clip_path)
                frames = meta.get("frames", [])
                key_idx = select_key_frame_idx(frames, config.key_frame_index)
                frame = frames[key_idx]
                ego_state = _load_pickle(os.path.join(clip_path, frame["ego_state"]))
                planned = trajectory_provider(clip_path, key_idx, ego_state)

            result = evaluate_clip(clip_path, config, planned_trajectory=planned)
            results.append(result)
        except Exception as e:
            logger.warning("Failed on clip %s: %s", clip_path, e)
            results.append(ClipResult(
                clip_path=clip_path,
                clip_name=os.path.basename(clip_path),
                key_frame_idx=-1,
                error=str(e),
            ))

    valid = [r for r in results if r.error is None]
    n_valid = len(valid)
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

    summary = BenchmarkSummary(
        total_clips=len(clips),
        evaluated_clips=n_valid,
        failed_clips=len(results) - n_valid,
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
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Planning trajectory benchmark",
    )
    parser.add_argument(
        "--data_root", type=str,
        default="./dataset/data/validation",
        help="Root directory containing clip subdirectories",
    )
    parser.add_argument("--max_clips", type=int, default=0, help="Cap on clips (0 = all)")
    parser.add_argument(
        "--key_frame_index",
        type=int,
        default=KEY_FRAME_INDEX,
        help="Metadata frame index used for planning (default: 100 ≈ 10 s at 10 Hz; clamped on shorter clips)",
    )
    parser.add_argument("--mode", type=str, default="vla",
                        choices=["gt", "vla", *BASELINE_MODES],
                        help="Trajectory source: 'gt' (oracle sanity check), 'vla' (nuVLA "
                             "checkpoint), or a registered baseline planner from "
                             "nureasoning.planning.baselines")

    # VLA model arguments (used when --mode=vla)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to VLA checkpoint directory (e.g. nureasoning_vla_workspace/epoch_3)")
    parser.add_argument("--num_inference_steps", type=int, default=5,
                        help="Number of ODE integration steps for flow-matching sampling")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for VLA inference")
    add_planning_prompt_arguments(parser)

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for reports and visualisations "
                             "(default: workspace folder, parent of --checkpoint_dir; "
                             "otherwise nureasoning_planning_benchmark_output)")
    parser.add_argument("--vis_clips", type=int, default=5,
                        help="Number of per-clip visualisations to save (0 = none)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for Python, NumPy, and PyTorch if available")

    # Weight overrides for the non-binary metrics
    parser.add_argument("--w_progress", type=float, default=0.3)
    parser.add_argument("--w_comfort", type=float, default=0.2)
    parser.add_argument("--w_human", type=float, default=0.5)

    args = parser.parse_args()
    if args.output_dir is None:
        if args.checkpoint_dir:
            workspace = os.path.dirname(os.path.normpath(args.checkpoint_dir))
            args.output_dir = workspace if workspace else "."
        else:
            args.output_dir = "nureasoning_planning_benchmark_output"
    set_random_seed(args.seed)
    logger.info("Random seed set to %d", args.seed)

    config = BenchmarkConfig(
        w_progress=args.w_progress,
        w_comfort=args.w_comfort,
        w_human=args.w_human,
        key_frame_index=args.key_frame_index,
    )
    logger.info("Key frame index: %d", config.key_frame_index)

    # Select trajectory provider based on mode
    trajectory_provider = None
    if args.mode == "vla":
        if not args.checkpoint_dir:
            parser.error("--checkpoint_dir is required when --mode=vla")
        trajectory_provider = VLATrajectoryProvider(
            checkpoint_dir=args.checkpoint_dir,
            num_inference_steps=args.num_inference_steps,
            device=args.device,
            planning_prompt=args.planning_prompt,
            reasoning_format=args.planning_reasoning_format,
        )
        logger.info("Mode: VLA model from %s", args.checkpoint_dir)
    elif args.mode in BASELINE_MODES:
        trajectory_provider = get_baseline(args.mode)
        logger.info("Mode: baseline planner '%s'", args.mode)
    else:
        logger.info("Mode: GT trajectory (sanity check)")

    summary = run_benchmark(
        args.data_root,
        config,
        max_clips=args.max_clips,
        trajectory_provider=trajectory_provider,
    )

    # --- Print results ---
    print("\n" + "=" * 70)
    print("PLANNING BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Clips evaluated : {summary.evaluated_clips} / {summary.total_clips}")
    print(f"  Failed          : {summary.failed_clips}")
    print("  ---")
    print(f"  Collision       : {summary.mean_collision:.4f}")
    print(f"  Driveable Area  : {summary.mean_driveable:.4f}")
    print(f"  Progress        : {summary.mean_progress:.4f}")
    print(f"  Comfort         : {summary.mean_comfort:.4f}")
    print(f"  Human Likeness  : {summary.mean_human:.4f}")
    print(f"  ADE / FDE       : {summary.mean_ade_m:.3f} m / {summary.mean_fde_m:.3f} m")
    print("  ---")
    print(f"  Planning Score  : {summary.mean_planning_score:.4f}")
    print("=" * 70)

    # Detailed per-clip table
    valid = [r for r in summary.results if r.error is None]
    if valid:
        print(f"\n{'Clip':<50s} {'Col':>5s} {'DA':>5s} {'Prog':>5s} {'Comf':>5s} {'Hum':>5s} {'ADE':>6s} {'FDE':>6s} {'NPS':>6s}")
        print("-" * 100)
        for r in valid:
            name = r.clip_name[:48]
            ade = float(r.details_human.get("ade", 0.0)) if r.details_human else 0.0
            fde = float(r.details_human.get("fde", 0.0)) if r.details_human else 0.0
            print(f"{name:<50s} {r.s_collision:5.2f} {r.s_driveable:5.2f} "
                  f"{r.s_progress:5.2f} {r.s_comfort:5.2f} {r.s_human:5.2f} "
                  f"{ade:6.2f} {fde:6.2f} {r.planning_score:6.3f}")

    # --- Visualisation ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Per-clip visualisations
    if args.vis_clips > 0 and valid:
        for i, r in enumerate(valid[:args.vis_clips]):
            out_path = os.path.join(args.output_dir, f"clip_{i:03d}_{r.clip_name[:40]}.png")
            visualize_clip_result(r, config, logger, save_path=out_path, show=args.show)

    # Summary visualisation
    visualize_summary(
        summary,
        logger,
        save_path=os.path.join(args.output_dir, "benchmark_summary.png"),
        show=args.show,
    )

    # Save JSON report
    report = {
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
        "config": {
            "seed": args.seed,
            "w_progress": config.w_progress,
            "w_comfort": config.w_comfort,
            "w_human": config.w_human,
            "trajectory_dt_s": config.trajectory_dt_s,
            "trajectory_steps": config.trajectory_steps,
            "comfort_dt_s": config.comfort_dt_s,
            "human_fde_full_score_m": config.human_fde_full_score_m,
            "human_fde_threshold_m": config.human_fde_threshold_m,
            "key_frame_index": config.key_frame_index,
        },
        "per_clip": [
            {
                "clip": r.clip_name,
                "clip_path": r.clip_path,
                "key_frame": r.key_frame_idx,
                "collision": r.s_collision,
                "driveable": r.s_driveable,
                "progress": r.s_progress,
                "comfort": r.s_comfort,
                "human": r.s_human,
                "ADE_m": r.details_human.get("ade") if r.details_human else None,
                "FDE_m": r.details_human.get("fde") if r.details_human else None,
                "planning_score": r.planning_score,
                "error": r.error,
                "details": {
                    "collision": r.details_collision,
                    "driveable": r.details_driveable,
                    "progress": r.details_progress,
                    "comfort": {
                        k: v for k, v in r.details_comfort.items()
                    } if r.details_comfort else {},
                    "human": r.details_human,
                },
            }
            for r in summary.results
        ],
    }
    report_path = os.path.join(args.output_dir, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved JSON report to %s", report_path)

    csv_path = os.path.join(args.output_dir, "benchmark_report.csv")
    csv_fieldnames = [
        "clip",
        "key_frame",
        "collision",
        "driveable",
        "progress",
        "comfort",
        "human",
        "ADE_m",
        "FDE_m",
        "planning_score",
        "error",
        "details_collision",
        "details_driveable",
        "details_progress",
        "details_comfort",
        "details_human",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        for r in summary.results:
            writer.writerow({
                "clip": r.clip_name,
                "key_frame": r.key_frame_idx,
                "collision": r.s_collision,
                "driveable": r.s_driveable,
                "progress": r.s_progress,
                "comfort": r.s_comfort,
                "human": r.s_human,
                "ADE_m": r.details_human.get("ade") if r.details_human else None,
                "FDE_m": r.details_human.get("fde") if r.details_human else None,
                "planning_score": r.planning_score,
                "error": r.error,
                "details_collision": json.dumps(r.details_collision, default=str),
                "details_driveable": json.dumps(r.details_driveable, default=str),
                "details_progress": json.dumps(r.details_progress, default=str),
                "details_comfort": json.dumps(r.details_comfort, default=str),
                "details_human": json.dumps(r.details_human, default=str),
            })
    logger.info("Saved CSV report to %s", csv_path)

    summary_csv_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    summary_fieldnames = [
        "total_clips",
        "evaluated_clips",
        "failed_clips",
        "mean_collision",
        "mean_driveable",
        "mean_progress",
        "mean_comfort",
        "mean_human",
        "mean_ADE_m",
        "mean_FDE_m",
        "mean_planning_score",
    ]
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerow({
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
        })
    logger.info("Saved CSV summary to %s", summary_csv_path)

    print(f"\nOutputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
