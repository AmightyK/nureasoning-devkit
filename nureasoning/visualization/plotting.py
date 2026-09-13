from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator as _PchipInterpolator
from shapely.geometry import LineString

from nureasoning.common.schema import Annotations, EgoState, nuReasoningStaticMap
from nureasoning.common.pickle_io import load_pickle as _load_pickle

_ROUTE_INTERP_SPACING_M = 0.50
_ROUTE_INTERP_MIN_SAMPLES = 128
_ROUTE_INTERP_MAX_SAMPLES = 6000
VEHICLE_REAR_LENGTH = 0.79

LIGHT_GREY = "#D3D3D3"
NEW_TAB_10: Dict[int, str] = {
    0: "#4e79a7",
    1: "#f28e2b",
    2: "#e15759",
    3: "#76b7b2",
    4: "#59a14f",
    5: "#edc948",
    6: "#b07aa1",
    7: "#ff9da7",
    8: "#9c755f",
    9: "#bab0ac",
}
ELLIS_5: Dict[int, str] = {
    0: "#DE7061",
    1: "#B0E685",
    2: "#4AC4BD",
    3: "#E38C47",
    4: "#699CDB",
}


@dataclass(frozen=True)
class PlotStyle:
    fill_color: Optional[str] = None
    fill_alpha: float = 0.0
    line_color: Optional[str] = None
    line_alpha: float = 1.0
    line_width: float = 1.0
    line_style: str = "-"
    marker: Optional[str] = None
    marker_size: float = 0.0
    marker_edge_color: Optional[str] = None
    zorder: int = 1


MAP_LAYER_STYLES: Dict[str, PlotStyle] = {
    "lane": PlotStyle(
        fill_color=LIGHT_GREY, fill_alpha=1.0, line_color=LIGHT_GREY, line_alpha=0.0, zorder=1
    ),
    "road_block": PlotStyle(
        fill_color="#0000C0", fill_alpha=0.10, line_color="#0000C0", line_alpha=0.35, zorder=0
    ),
    "intersection": PlotStyle(
        fill_color=LIGHT_GREY, fill_alpha=1.0, line_color=LIGHT_GREY, line_alpha=0.0, zorder=0
    ),
    "crosswalk": PlotStyle(
        fill_color=NEW_TAB_10[6], fill_alpha=0.28, line_color=NEW_TAB_10[6], line_alpha=0.0, zorder=1
    ),
    "stop_polygon": PlotStyle(
        fill_color="#FF0101", fill_alpha=0.18, line_color="#FF0101", line_alpha=0.25, zorder=2
    ),
    "boundary": PlotStyle(
        line_color="#2c3e50", line_alpha=0.40, line_width=0.9, line_style="-", zorder=2
    ),
    "lane_centerline": PlotStyle(
        line_color="#666666", line_alpha=0.50, line_width=0.9, line_style="-", zorder=2
    ),
    "lane_connector": PlotStyle(
        line_color="#CBCBCB", line_alpha=0.95, line_width=1.0, line_style="-", zorder=2
    ),
    "baseline_path": PlotStyle(
        line_color="#666666", line_alpha=0.9, line_width=1.0, line_style="--", zorder=2
    ),
    "traffic_light": PlotStyle(
        fill_color="#f1c40f", fill_alpha=0.95, line_color="black", line_alpha=1.0, zorder=4
    ),
}

ACTOR_STYLES: Dict[str, PlotStyle] = {
    "vehicle": PlotStyle(
        fill_color=ELLIS_5[4], fill_alpha=1.0, line_color="black", line_alpha=1.0, zorder=6
    ),
    "pedestrian": PlotStyle(
        fill_color=NEW_TAB_10[6], fill_alpha=1.0, line_color="black", line_alpha=1.0, zorder=6
    ),
    "bicycle": PlotStyle(
        fill_color=ELLIS_5[3], fill_alpha=1.0, line_color="black", line_alpha=1.0, zorder=6
    ),
    "generic": PlotStyle(
        fill_color=NEW_TAB_10[5], fill_alpha=1.0, line_color="black", line_alpha=1.0, zorder=6
    ),
    "ego": PlotStyle(
        fill_color=ELLIS_5[0], fill_alpha=1.0, line_color="black", line_alpha=1.0, line_width=1.4, zorder=8
    ),
}

TRAJECTORY_STYLES: Dict[str, PlotStyle] = {
    "history": PlotStyle(
        line_color="#2c3e50",
        line_alpha=0.85,
        line_width=2.0,
        marker="o",
        marker_size=3.5,
        zorder=7,
    ),
    "future": PlotStyle(
        line_color=NEW_TAB_10[4],
        line_alpha=1.0,
        line_width=2.0,
        marker="o",
        marker_size=4.0,
        marker_edge_color="black",
        zorder=7,
    ),
    "gt": PlotStyle(
        line_color=NEW_TAB_10[4],
        line_alpha=1.0,
        line_width=2.0,
        marker="o",
        marker_size=4.5,
        marker_edge_color="black",
        zorder=7,
    ),
    "planned": PlotStyle(
        line_color=ELLIS_5[0],
        line_alpha=1.0,
        line_width=2.0,
        marker="o",
        marker_size=4.5,
        marker_edge_color="black",
        zorder=7,
    ),
    "route": PlotStyle(
        line_color="#8e44ad", line_alpha=0.75, line_width=2.0, line_style="-", zorder=3
    ),
}

TL_STATE_COLORS: Dict[str, str] = {
    "red": "#e74c3c",
    "yellow": "#f1c40f",
    "green": "#2ecc71",
    "off": "#95a5a6",
    "unknown": "#7f8c8d",
}


def _load_metadata(clip_path: str) -> Dict[str, Any]:
    with open(os.path.join(clip_path, "metadata.json"), "r") as f:
        return json.load(f)


def _box_corners_2d(
    cx: float, cy: float, half_l: float, half_w: float, yaw: float,
) -> List[tuple[float, float]]:
    c, s = math.cos(yaw), math.sin(yaw)
    dx = [half_l, half_l, -half_l, -half_l]
    dy = [half_w, -half_w, -half_w, half_w]
    return [(cx + c * dx[i] - s * dy[i], cy + s * dx[i] + c * dy[i]) for i in range(4)]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _yaw_from_pose(pose: Dict[str, Any]) -> float:
    if pose is None:
        return 0.0
    if "yaw" in pose:
        return _safe_float(pose.get("yaw"), 0.0)
    qw = _safe_float(pose.get("qw"), 1.0)
    qx = _safe_float(pose.get("qx"), 0.0)
    qy = _safe_float(pose.get("qy"), 0.0)
    qz = _safe_float(pose.get("qz"), 0.0)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _metadata_ego_dimensions(metadata: Optional[Dict[str, Any]]) -> Dict[str, float]:
    raw = (metadata or {}).get("ego_dimensions", {})
    if not isinstance(raw, dict):
        return {}
    dims: Dict[str, float] = {}
    key_map = {
        "length": "l", "width": "w", "height": "h",
        "vehicle_rear_length": "vehicle_rear_length",
        "l": "l", "w": "w", "h": "h",
    }
    for raw_key, normalized in key_map.items():
        if raw_key in raw:
            dims[normalized] = _safe_float(raw[raw_key], 0.0)
    return dims


def _ego_box_center_from_base_link(
    x: float,
    y: float,
    yaw: float,
    ego_length: float,
    vehicle_rear_length: float = VEHICLE_REAR_LENGTH,
) -> tuple[float, float]:
    center_offset = ego_length / 2.0 - vehicle_rear_length
    return (
        x + math.cos(yaw) * center_offset,
        y + math.sin(yaw) * center_offset,
    )


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
    route_path = mission_goal.get("route_path") if isinstance(mission_goal, dict) else getattr(mission_goal, "route_path", None)
    if not route_path or len(route_path) < 2:
        return None
    interpolated_xy = _build_interpolated_route_xy(route_path)
    if interpolated_xy is None or len(interpolated_xy) < 2:
        return None

    return LineString(interpolated_xy)


def _viz_configure_ax(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _viz_configure_bev_ax(ax, ego_x: float, ego_y: float, margin_m: float) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(ego_x - margin_m, ego_x + margin_m)
    ax.set_ylim(ego_y - margin_m, ego_y + margin_m)
    ax.set_facecolor("white")
    _viz_configure_ax(ax)


def _viz_geometry_xy(geometry: Any, min_points: int = 2) -> Optional[np.ndarray]:
    if not geometry:
        return None
    pts = np.asarray(geometry, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < min_points or pts.shape[1] < 2:
        return None
    return pts[:, :2]


def _viz_geometry_center(geometry: Any) -> Optional[np.ndarray]:
    pts = _viz_geometry_xy(geometry, min_points=1)
    if pts is None:
        return None
    return pts.mean(axis=0)


def _viz_add_polygon(ax, geometry: Any, style: PlotStyle) -> None:
    pts = _viz_geometry_xy(geometry, min_points=3)
    if pts is None:
        return
    patch = mpatches.Polygon(
        pts,
        closed=True,
        facecolor=style.fill_color or "none",
        edgecolor=style.line_color or style.fill_color or "none",
        alpha=style.fill_alpha if style.fill_color else style.line_alpha,
        linewidth=style.line_width,
        linestyle=style.line_style,
        zorder=style.zorder,
    )
    ax.add_patch(patch)


def _viz_add_polyline(ax, geometry: Any, style: PlotStyle) -> None:
    pts = _viz_geometry_xy(geometry, min_points=2)
    if pts is None:
        return
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        color=style.line_color,
        alpha=style.line_alpha,
        linewidth=style.line_width,
        linestyle=style.line_style,
        zorder=style.zorder,
    )


def _viz_add_box(
    ax,
    cx: float,
    cy: float,
    length: float,
    width: float,
    yaw: float,
    style: PlotStyle,
    add_heading: bool = True,
) -> None:
    if length <= 0 or width <= 0:
        return
    corners = np.asarray(_box_corners_2d(cx, cy, length / 2.0, width / 2.0, yaw), dtype=np.float64)
    closed = np.vstack([corners, corners[0]])

    if style.fill_color:
        ax.fill(
            closed[:, 0],
            closed[:, 1],
            color=style.fill_color,
            alpha=style.fill_alpha,
            zorder=style.zorder,
        )

    if style.line_color:
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color=style.line_color,
            alpha=style.line_alpha,
            linewidth=style.line_width,
            linestyle=style.line_style,
            zorder=style.zorder,
        )

    if add_heading and style.line_color:
        heading_len = max(1.0, length * 0.35)
        heading = np.array(
            [[cx, cy], [cx + math.cos(yaw) * heading_len, cy + math.sin(yaw) * heading_len]],
            dtype=np.float64,
        )
        ax.plot(
            heading[:, 0],
            heading[:, 1],
            color=style.line_color,
            alpha=style.line_alpha,
            linewidth=style.line_width,
            linestyle=style.line_style,
            zorder=style.zorder + 0.1,
        )


def _viz_sample_trajectory_for_plot(
    trajectory: np.ndarray,
    trajectory_dt_s: float,
    plot_dt_s: float = 0.5,
) -> np.ndarray:
    """Return t=0 and every plot_dt_s waypoint for cleaner BEV trajectory plots."""
    if len(trajectory) <= 1:
        return trajectory

    dt = max(float(trajectory_dt_s), 1e-6)
    stride = max(1, int(round(float(plot_dt_s) / dt)))
    sampled = trajectory[::stride]
    if not np.array_equal(sampled[-1], trajectory[-1]):
        sampled = np.vstack([sampled, trajectory[-1]])
    return sampled


def _viz_add_trajectory(
    ax,
    trajectory: Optional[np.ndarray],
    style: PlotStyle,
    *,
    trajectory_dt_s: float = 0.1,
    plot_dt_s: float = 0.5,
) -> None:
    if trajectory is None or len(trajectory) == 0:
        return
    pts = np.asarray(trajectory, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return
    pts = _viz_sample_trajectory_for_plot(pts, trajectory_dt_s, plot_dt_s)
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        color=style.line_color,
        alpha=style.line_alpha,
        linewidth=style.line_width,
        linestyle=style.line_style,
        marker=style.marker,
        markersize=style.marker_size,
        markeredgecolor=style.marker_edge_color,
        zorder=style.zorder,
    )


def _viz_classify_actor(category: str) -> str:
    cat = (category or "").lower()
    if "pedestrian" in cat or cat == "human":
        return "pedestrian"
    if "bike" in cat or "cycle" in cat:
        return "bicycle"
    if "vehicle" in cat or "car" in cat or "truck" in cat or "bus" in cat:
        return "vehicle"
    return "generic"


def _viz_build_map_lookup(static_map: nuReasoningStaticMap) -> Dict[str, Dict[int, Any]]:
    return {
        "lane_connectors": {
            int(obj.id): obj for obj in getattr(static_map, "lane_connectors", []) if getattr(obj, "id", None) is not None
        },
        "road_blocks": {
            int(obj.id): obj for obj in getattr(static_map, "road_blocks", []) if getattr(obj, "id", None) is not None
        },
        "traffic_lights": {
            int(obj.id): obj for obj in getattr(static_map, "traffic_lights", []) if getattr(obj, "id", None) is not None
        },
    }


def _viz_resolve_tl_center(
    tl_state: Any,
    static_map_lookup: Dict[str, Dict[int, Any]],
) -> Optional[np.ndarray]:
    lane_connector_id = getattr(tl_state, "lane_connector_id", None)
    if lane_connector_id is not None:
        lane_connector = static_map_lookup["lane_connectors"].get(int(lane_connector_id))
        if lane_connector is not None:
            center = _viz_geometry_center(getattr(lane_connector, "geometry", None))
            if center is not None:
                return center

    roadblock_id = getattr(tl_state, "roadblock_id", None)
    if roadblock_id is not None:
        road_block = static_map_lookup["road_blocks"].get(int(roadblock_id))
        if road_block is not None:
            center = _viz_geometry_center(getattr(road_block, "geometry", None))
            if center is not None:
                return center

    tl_id = getattr(tl_state, "id", None)
    if tl_id is not None:
        traffic_light = static_map_lookup["traffic_lights"].get(int(tl_id))
        if traffic_light is not None:
            return _viz_geometry_center(getattr(traffic_light, "geometry", None))

    return None


def _viz_draw_static_map(ax, static_map: nuReasoningStaticMap) -> Dict[str, Dict[int, Any]]:
    for road_block in getattr(static_map, "road_blocks", []):
        _viz_add_polygon(ax, road_block.geometry, MAP_LAYER_STYLES["road_block"])

    for intersection in getattr(static_map, "intersections", []):
        _viz_add_polygon(ax, intersection.geometry, MAP_LAYER_STYLES["intersection"])

    for lane in getattr(static_map, "lanes", []):
        _viz_add_polygon(ax, lane.polygon, MAP_LAYER_STYLES["lane"])
        _viz_add_polyline(ax, getattr(lane, "centerline", None), MAP_LAYER_STYLES["lane_centerline"])

    for crosswalk in getattr(static_map, "crosswalks", []):
        _viz_add_polygon(ax, crosswalk.geometry, MAP_LAYER_STYLES["crosswalk"])

    for stop_polygon in getattr(static_map, "stop_polygons", []):
        _viz_add_polygon(ax, stop_polygon.geometry, MAP_LAYER_STYLES["stop_polygon"])

    # for boundary in getattr(static_map, "boundaries", []):
    #     _viz_add_polyline(ax, boundary.geometry, MAP_LAYER_STYLES["boundary"])

    for lane_connector in getattr(static_map, "lane_connectors", []):
        _viz_add_polyline(ax, lane_connector.geometry, MAP_LAYER_STYLES["lane_connector"])

    for baseline_path in getattr(static_map, "baseline_paths", []):
        _viz_add_polyline(ax, baseline_path.geometry, MAP_LAYER_STYLES["baseline_path"])

    # for traffic_light in getattr(static_map, "traffic_lights", []):
    #     center = _viz_geometry_center(getattr(traffic_light, "geometry", None))
    #     if center is None:
    #         continue
    #     ax.scatter(
    #         [center[0]],
    #         [center[1]],
    #         s=16,
    #         c=MAP_LAYER_STYLES["traffic_light"].fill_color,
    #         edgecolors=MAP_LAYER_STYLES["traffic_light"].line_color,
    #         linewidths=0.4,
    #         alpha=MAP_LAYER_STYLES["traffic_light"].fill_alpha,
    #         zorder=MAP_LAYER_STYLES["traffic_light"].zorder,
    #     )

    return _viz_build_map_lookup(static_map)


def _viz_draw_traffic_light_states(
    ax,
    ann: Annotations,
    static_map_lookup: Optional[Dict[str, Dict[int, Any]]],
) -> None:
    if static_map_lookup is None:
        return
    for tl_state in ann.traffic_light_states:
        center = _viz_resolve_tl_center(tl_state, static_map_lookup)
        if center is None:
            continue
        state = (getattr(tl_state, "state", None) or "unknown").lower()
        color = TL_STATE_COLORS.get(state, TL_STATE_COLORS["unknown"])
        ax.scatter(
            [center[0]],
            [center[1]],
            s=42,
            c=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=7,
        )


def _viz_draw_objects(ax, ann: Annotations) -> None:
    for obj in ann.objects:
        cat = (obj.category or "").lower()
        if cat.startswith("other.") or cat == "other":
            continue
        actor_key = _viz_classify_actor(cat)
        _viz_add_box(
            ax,
            cx=float(obj.pose.get("x", 0.0)),
            cy=float(obj.pose.get("y", 0.0)),
            length=float(obj.dimensions.get("l", 0.0)),
            width=float(obj.dimensions.get("w", 0.0)),
            yaw=float(obj.pose.get("yaw", 0.0)),
            style=ACTOR_STYLES[actor_key],
            add_heading=True,
        )


def _viz_draw_ego(ax, ego_pose: Dict[str, float], ego_dims: Dict[str, float]) -> None:
    length = float(ego_dims.get("l", 5.176))
    yaw = _yaw_from_pose(ego_pose)
    ego_cx, ego_cy = _ego_box_center_from_base_link(
        float(ego_pose.get("x", 0.0)),
        float(ego_pose.get("y", 0.0)),
        yaw,
        length,
        float(ego_dims.get("vehicle_rear_length", VEHICLE_REAR_LENGTH)),
    )
    _viz_add_box(
        ax,
        cx=ego_cx,
        cy=ego_cy,
        length=length,
        width=float(ego_dims.get("w", 2.297)),
        yaw=yaw,
        style=ACTOR_STYLES["ego"],
        add_heading=True,
    )


def _viz_draw_camera_center_ego(ax, ego_state: EgoState) -> None:
    """Ego-state card for the empty cell in the 3x3 camera grid."""
    ax.clear()
    ax.set_facecolor("black")
    ax.axis("off")
    pose = getattr(ego_state, "pose", None) or {}
    velocity = getattr(ego_state, "velocity", None) or {}
    acceleration = getattr(ego_state, "acceleration", None) or {}
    x = _safe_float(pose.get("x", 0.0))
    y = _safe_float(pose.get("y", 0.0))
    z = _safe_float(pose.get("z", 0.0))
    yaw = _yaw_from_pose(pose)
    vx = _safe_float(velocity.get("x", velocity.get("vx", 0.0)))
    vy = _safe_float(velocity.get("y", velocity.get("vy", 0.0)))
    vz = _safe_float(velocity.get("z", velocity.get("vz", 0.0)))
    ax_val = _safe_float(acceleration.get("x", acceleration.get("ax", 0.0)))
    ay_val = _safe_float(acceleration.get("y", acceleration.get("ay", 0.0)))
    az_val = _safe_float(acceleration.get("z", acceleration.get("az", 0.0)))
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    accel = math.sqrt(ax_val * ax_val + ay_val * ay_val + az_val * az_val)
    card = mpatches.FancyBboxPatch(
        (0.06, 0.08), 0.88, 0.84,
        boxstyle="round,pad=0.025,rounding_size=0.04",
        transform=ax.transAxes,
        facecolor="#111827", edgecolor="#6b7280",
        linewidth=1.0, alpha=0.96,
    )
    ax.add_patch(card)
    ax.text(0.5, 0.84, "EGO STATE", transform=ax.transAxes, color="#93c5fd",
            fontsize=8.5, fontweight="bold", ha="center", va="center")
    ax.text(0.5, 0.70, f"{speed:.2f} m/s", transform=ax.transAxes, color="white",
            fontsize=13, fontweight="bold", ha="center", va="center")
    ax.text(0.5, 0.60, f"accel {accel:.2f} m/s^2", transform=ax.transAxes,
            color="#d1d5db", fontsize=7.5, ha="center", va="center")
    ax.text(
        0.10, 0.45,
        "\n".join([
            f"pose   x {x:7.2f}   y {y:7.2f}   z {z:5.2f}",
            f"yaw    {math.degrees(yaw):7.1f} deg",
            f"vel    x {vx:7.2f}   y {vy:7.2f}   z {vz:5.2f}",
            f"acc    x {ax_val:7.2f}   y {ay_val:7.2f}   z {az_val:5.2f}",
        ]),
        transform=ax.transAxes, color="#f9fafb", fontsize=6.5,
        family="monospace", ha="left", va="top", linespacing=1.35,
    )


def _viz_add_data_legend(ax) -> None:
    handles = [
        mpatches.Patch(facecolor=MAP_LAYER_STYLES["lane"].fill_color, edgecolor="none", label="Lane"),
        mpatches.Patch(facecolor=MAP_LAYER_STYLES["crosswalk"].fill_color, edgecolor="none", label="Crosswalk"),
        Line2D([0], [0], color=MAP_LAYER_STYLES["baseline_path"].line_color,
               linestyle="--", linewidth=1.0, label="Baseline"),
        Line2D([0], [0], color=TRAJECTORY_STYLES["route"].line_color, linewidth=2.0, label="Route"),
        mpatches.Patch(facecolor=ACTOR_STYLES["ego"].fill_color, edgecolor="black", label="Ego"),
        mpatches.Patch(facecolor=ACTOR_STYLES["vehicle"].fill_color, edgecolor="black", label="Vehicle"),
        Line2D([0], [0], color=TRAJECTORY_STYLES["history"].line_color, marker="o",
               linewidth=2.0, label="Ego history"),
        Line2D([0], [0], color=TRAJECTORY_STYLES["future"].line_color, marker="o",
               linewidth=2.0, label="Ego future"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TL_STATE_COLORS["red"],
               markeredgecolor="black", label="TL state"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.92, ncol=2)


def _viz_draw_bev_scene(
    ax,
    ego_state: EgoState,
    annotations: Optional[Annotations],
    static_map: Optional[nuReasoningStaticMap],
    mission_goal: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
    bev_range_m: float = 80.0,
    trajectory_dt_s: float = 0.1,
    show_objects: bool = True,
) -> None:
    pose = ego_state.pose if ego_state is not None else {}
    ego_x = _safe_float(pose.get("x", 0.0))
    ego_y = _safe_float(pose.get("y", 0.0))
    _viz_configure_bev_ax(ax, ego_x, ego_y, bev_range_m)
    ax.set_title("BEV: Map, Actors, Ego Motion, Route, TL", fontsize=11)

    lookup = _viz_draw_static_map(ax, static_map) if static_map is not None else None
    route_line = build_route_line(mission_goal)
    if route_line is not None:
        rx, ry = route_line.xy
        style = TRAJECTORY_STYLES["route"]
        ax.plot(rx, ry, color=style.line_color, linewidth=style.line_width,
                alpha=style.line_alpha, linestyle=style.line_style, zorder=style.zorder)
    if annotations is not None:
        _viz_draw_traffic_light_states(ax, annotations, lookup)
        if show_objects:
            _viz_draw_objects(ax, annotations)
    if ego_state is not None:
        _viz_add_trajectory(ax, getattr(ego_state, "trajectory_history", None),
                            TRAJECTORY_STYLES["history"], trajectory_dt_s=trajectory_dt_s)
        _viz_add_trajectory(ax, getattr(ego_state, "trajectory_future", None),
                            TRAJECTORY_STYLES["future"], trajectory_dt_s=trajectory_dt_s)
        ego_dims = _metadata_ego_dimensions(metadata)
        if not ego_dims:
            ego_dims = getattr(ego_state, "dimensions", None) or {}
        _viz_draw_ego(ax, pose, ego_dims if isinstance(ego_dims, dict) else {})
    _viz_add_data_legend(ax)


def _viz_add_clip_legend(ax, include_collision: bool) -> None:
    handles = [
        mpatches.Patch(facecolor=MAP_LAYER_STYLES["lane"].fill_color, edgecolor="none", label="Lane"),
        mpatches.Patch(facecolor=MAP_LAYER_STYLES["crosswalk"].fill_color, edgecolor="none", label="Crosswalk"),
        Line2D([0], [0], color=MAP_LAYER_STYLES["baseline_path"].line_color, linestyle="--", linewidth=1.0, label="Baseline"),
        Line2D([0], [0], color=TRAJECTORY_STYLES["route"].line_color, linewidth=2.0, label="Route"),
        mpatches.Patch(facecolor=ACTOR_STYLES["ego"].fill_color, edgecolor="black", label="Ego"),
        mpatches.Patch(facecolor=ACTOR_STYLES["vehicle"].fill_color, edgecolor="black", label="Vehicle"),
        mpatches.Patch(facecolor=ACTOR_STYLES["pedestrian"].fill_color, edgecolor="black", label="Pedestrian"),
        Line2D([0], [0], color=TRAJECTORY_STYLES["gt"].line_color, marker="o", linewidth=2.0, label="GT"),
        Line2D([0], [0], color=TRAJECTORY_STYLES["planned"].line_color, marker="o", linewidth=2.0, label="Planned"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TL_STATE_COLORS["red"], markeredgecolor="black", label="TL state"),
    ]
    if include_collision:
        handles.append(
            Line2D([0], [0], marker="X", color="red", markeredgecolor="darkred", linewidth=0.0, markersize=8, label="Collision")
        )
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.92, ncol=2)


def _viz_render_score_panel(ax_bar, result: Any) -> None:
    labels = [
        "Collision",
        "Driveable\nArea",
        "Progress",
        "Comfort",
        "Human\nLikeness",
        "Planning\nScore",
    ]
    values = [
        result.s_collision,
        result.s_driveable,
        result.s_progress,
        result.s_comfort,
        result.s_human,
        result.planning_score,
    ]
    colors = [
        "#2ecc71" if value >= 0.8 else "#f39c12" if value >= 0.4 else "#e74c3c"
        for value in values
    ]
    colors[-1] = "#3498db"

    bars = ax_bar.barh(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    for bar, value in zip(bars, values):
        ax_bar.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax_bar.set_xlim(0.0, 1.15)
    ax_bar.set_title("Score Breakdown", fontsize=12, fontweight="bold")
    ax_bar.grid(axis="x", linestyle=":", alpha=0.3)
    ax_bar.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    detail_lines: List[str] = []
    d_col = result.details_collision
    if d_col.get("has_collision"):
        detail_lines.append(f"Collision @ step {d_col['collision_timestep']} ({d_col['collision_object'][:8]})")
    d_da = result.details_driveable
    detail_lines.append(f"In-road fraction: {d_da.get('fraction_in', 0):.2f}")
    d_prog = result.details_progress
    detail_lines.append(f"Progress: {d_prog.get('planned_m', 0):.1f}m / {d_prog.get('gt_m', 0):.1f}m GT")
    d_comf = result.details_comfort
    if d_comf:
        detail_lines.append(f"Max lon accel: {d_comf.get('max_lon_accel', 0):.2f} m/s^2")
        detail_lines.append(f"Max lat accel: {d_comf.get('max_lat_accel', 0):.2f} m/s^2")
        detail_lines.append(f"Max lon jerk: {d_comf.get('max_lon_jerk', 0):.2f} m/s^3")
    d_hum = result.details_human
    detail_lines.append(f"ADE: {d_hum.get('ade', 0):.3f}m  FDE: {d_hum.get('fde', 0):.3f}m")

    ax_bar.text(
        0.02,
        -0.08,
        "\n".join(detail_lines),
        transform=ax_bar.transAxes,
        fontsize=8,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7),
    )


def _viz_finalize_figure(
    fig,
    logger,
    save_path: Optional[str],
    show: bool,
    *,
    tight_rect: Optional[List[float]] = None,
    save_label: str = "figure",
) -> None:
    import matplotlib.pyplot as plt

    if tight_rect is None:
        plt.tight_layout()
    else:
        plt.tight_layout(rect=tight_rect)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved %s to %s", save_label, save_path)
    if show:
        plt.show()
    plt.close(fig)


def visualize_clip_result(
    result: Any,
    config: Any,
    logger,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    if result.error is not None:
        logger.warning("Skipping viz for %s: %s", result.clip_name, result.error)
        return

    fig, (ax_bev, ax_bar) = plt.subplots(
        1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [2.2, 1.0]}
    )

    clip_path = result.clip_path
    meta = _load_metadata(clip_path)
    key_frame = meta["frames"][result.key_frame_idx]
    ann: Annotations = _load_pickle(os.path.join(clip_path, key_frame["annotations"]))

    ego_dims = {"l": 5.176, "w": 2.297}
    ego_state_obj: EgoState = _load_pickle(os.path.join(clip_path, key_frame["ego_state"]))
    ego_dimensions = getattr(ego_state_obj, "dimensions", None) or {}
    if isinstance(ego_dimensions, dict):
        ego_dims["l"] = float(ego_dimensions.get("l", ego_dims["l"]))
        ego_dims["w"] = float(ego_dimensions.get("w", ego_dims["w"]))

    ego_x = result.ego_pose["x"]
    ego_y = result.ego_pose["y"]
    _viz_configure_bev_ax(ax_bev, ego_x, ego_y, config.vis_range_m)
    ax_bev.set_title(
        f"{result.clip_name}\nFrame {result.key_frame_idx}  |  Planning = {result.planning_score:.3f}"
    )

    static_map_lookup: Optional[Dict[str, Dict[int, Any]]] = None
    map_path = os.path.join(clip_path, meta.get("map_annotation", "map.pkl"))
    if os.path.isfile(map_path):
        static_map: nuReasoningStaticMap = _load_pickle(map_path)
        static_map_lookup = _viz_draw_static_map(ax_bev, static_map)

    mission_goal = key_frame.get("mission_goal")
    route_line = build_route_line(mission_goal)
    if route_line is not None:
        rx, ry = route_line.xy
        ax_bev.plot(
            rx,
            ry,
            color=TRAJECTORY_STYLES["route"].line_color,
            linewidth=TRAJECTORY_STYLES["route"].line_width,
            alpha=TRAJECTORY_STYLES["route"].line_alpha,
            linestyle=TRAJECTORY_STYLES["route"].line_style,
            zorder=TRAJECTORY_STYLES["route"].zorder,
        )

    _viz_draw_traffic_light_states(ax_bev, ann, static_map_lookup)
    _viz_draw_objects(ax_bev, ann)
    _viz_draw_ego(ax_bev, result.ego_pose, ego_dims)
    _viz_add_trajectory(
        ax_bev,
        result.gt_traj,
        TRAJECTORY_STYLES["gt"],
        trajectory_dt_s=config.trajectory_dt_s,
    )
    _viz_add_trajectory(
        ax_bev,
        result.planned_traj,
        TRAJECTORY_STYLES["planned"],
        trajectory_dt_s=config.trajectory_dt_s,
    )

    _viz_add_clip_legend(ax_bev, include_collision=False)
    _viz_render_score_panel(ax_bar, result)
    _viz_finalize_figure(fig, logger, save_path, show, save_label="clip viz")


def visualize_summary(
    summary: Any,
    logger,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax1 = axes[0]
    labels = ["Collision", "Driveable", "Progress", "Comfort", "Human\nLikeness", "Planning Score"]
    means = [
        summary.mean_collision,
        summary.mean_driveable,
        summary.mean_progress,
        summary.mean_comfort,
        summary.mean_human,
        summary.mean_planning_score,
    ]
    colors = ["#2ecc71", "#27ae60", "#f39c12", "#e67e22", "#3498db", "#2c3e50"]
    bars = ax1.bar(labels, means, color=colors, edgecolor="black", linewidth=0.8)
    for bar, value in zip(bars, means):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Score")
    ax1.set_title(f"Average Scores  ({summary.evaluated_clips} clips)", fontweight="bold")
    ax1.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)

    ax2 = axes[1]
    valid_planning = [r.planning_score for r in summary.results if r.error is None]
    if valid_planning:
        ax2.hist(valid_planning, bins=20, range=(0, 1), color="#3498db", edgecolor="black", alpha=0.8)
    ax2.set_xlabel("Planning Score")
    ax2.set_ylabel("Count")
    ax2.set_title("Planning Score Distribution", fontweight="bold")

    fig.suptitle(
        f"Planning Benchmark Summary\n"
        f"Total: {summary.total_clips}  |  Evaluated: {summary.evaluated_clips}  |  "
        f"Failed: {summary.failed_clips}  |  Mean Planning: {summary.mean_planning_score:.4f}  |  "
        f"ADE/FDE: {getattr(summary, 'mean_ade_m', 0.0):.3f}/{getattr(summary, 'mean_fde_m', 0.0):.3f} m",
        fontsize=13,
        fontweight="bold",
    )
    _viz_finalize_figure(
        fig,
        logger,
        save_path,
        show,
        tight_rect=[0, 0, 1, 0.92],
        save_label="summary viz",
    )
