"""
nuReasoning reasoning-annotation viewer.

Pretty-prints Spatial / Decision / Counterfactual annotations. With
``--save-figures`` it writes a composed frame (driving + counterfactual text
above left/front/right cameras with spatial 2D boxes). ``--video`` writes an
MP4 around each selected reasoning frame (history, pause, future).

Usage:
    python -m nureasoning.visualization.view_reasoning --clip ./dataset/data/train/part_1/<clip>
    python -m nureasoning.visualization.view_reasoning --clip <clip> --save-figures
    python -m nureasoning.visualization.view_reasoning --clip <clip> --video
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FFMpegWriter
from PIL import Image, ImageDraw, ImageFont

from nureasoning.common.clips import discover_clips
from nureasoning.common.pickle_io import load_pickle
from nureasoning.visualization.ffmpeg_util import require_ffmpeg

WRAP_WIDTH = 100
FIGURE_SIZE = (24, 13)
CAM_BOX_REF = (2816.0, 1856.0)
DEFAULT_VIDEO_ANCHORS = [50, 100, 150]
TEXT_PANEL_W = 2880
TEXT_PANEL_H = 900
TEXT_COL_W = TEXT_PANEL_W // 2

BG_RGB = (18, 18, 18)
TEXT_RGB = (235, 235, 235)
HEADING_RGB = (100, 220, 255)
SUBHEAD_RGB = (255, 200, 70)
SAFE_RGB = (90, 255, 120)
UNSAFE_RGB = (255, 90, 90)
SUBOPT_RGB = (255, 195, 80)
DIM_RGB = (185, 185, 185)
COMP_RGB = (160, 220, 160)
DIV_RGB = (65, 65, 65)
BANNER_BG_RGB = (35, 35, 35)

CATEGORY_COLORS = {
    "car": "#28dcff",
    "vehicle.car": "#28dcff",
    "truck": "#46ff78",
    "vehicle.truck": "#46ff78",
    "bus": "#ffb43c",
    "vehicle.bus": "#ffb43c",
    "pedestrian": "#ff5ab4",
    "human.pedestrian": "#ff5ab4",
    "bicycle": "#b482ff",
    "vehicle.bicycle": "#b482ff",
    "motorcycle": "#78c8ff",
    "vehicle.motorcycle": "#78c8ff",
    "road obstacle": "#46beff",
    "construction.traffic_cone": "#46beff",
}


# ---------------------------------------------------------------------------
# Reasoning-file access helpers (tolerant to key-capitalisation variants)
# ---------------------------------------------------------------------------

def _get_ci(payload: Dict[str, Any], *keys: str) -> Any:
    """Return the first matching key (case-insensitive) from *payload*."""
    if not isinstance(payload, dict):
        return None
    lowered = {str(k).lower(): v for k, v in payload.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _wrap(text: str, indent: str = "    ", width: int = WRAP_WIDTH) -> str:
    return textwrap.fill(
        str(text), width=width, initial_indent=indent, subsequent_indent=indent
    )


# ---------------------------------------------------------------------------
# Section formatters
# ---------------------------------------------------------------------------

def format_spatial(spatial: Dict[str, Any], max_items: int = 10) -> List[str]:
    lines: List[str] = ["[Spatial reasoning]"]

    summary = _get_ci(spatial, "Summary")
    if isinstance(summary, dict) and summary:
        lines.append("  Summary:")
        for key, value in summary.items():
            lines.append(_wrap(f"{key}: {value}", indent="    "))
    elif summary:
        lines.append(_wrap(f"Summary: {summary}", indent="  "))

    per_camera = _get_ci(spatial, "Per_camera_results", "per_camera_results")
    if isinstance(per_camera, dict) and per_camera:
        lines.append("  Per-camera detections:")
        for cam, payload in per_camera.items():
            objects = payload.get("objects", []) if isinstance(payload, dict) else []
            if not objects:
                continue
            lines.append(f"    {cam}: {len(objects)} object(s)")
            for obj in objects[:max_items]:
                if not isinstance(obj, dict):
                    continue
                category = obj.get("category", obj.get("class", "object"))
                token = str(obj.get("track_token", ""))[:6]
                desc = obj.get("description", "")
                entry = f"- {category}" + (f" [{token}]" if token else "")
                if desc:
                    entry += f": {desc}"
                lines.append(_wrap(entry, indent="      "))

    relations = _as_list(_get_ci(spatial, "Object_relations", "object_relations"))
    if relations:
        lines.append("  Object relations:")
        for rel in relations[:max_items]:
            lines.append(_wrap(f"- {json.dumps(rel) if isinstance(rel, dict) else rel}",
                               indent="    "))
    return lines


def format_driving(driving: Dict[str, Any], width: int = WRAP_WIDTH) -> List[str]:
    lines: List[str] = ["[Decision reasoning]"]

    desc = _get_ci(driving, "Scene description", "Description")
    if desc:
        lines.append("  Scene:")
        lines.append(_wrap(desc, width=width))

    critical = _get_ci(driving, "Critical components")
    if isinstance(critical, dict) and critical:
        lines.append("  Critical components:")
        for key, value in critical.items():
            if isinstance(value, dict):
                ctype = value.get("Type", "")
                header = f"* {key}" + (f"  [{ctype}]" if ctype else "")
                lines.append(_wrap(header, indent="    ", width=width))
                for field, field_value in value.items():
                    if field == "Type":
                        continue
                    lines.append(_wrap(f"{field}: {field_value}", indent="      ", width=width))
            else:
                lines.append(_wrap(f"* {key}: {value}", indent="    ", width=width))
    elif critical:
        lines.append(_wrap(f"Critical components: {critical}", indent="  ", width=width))

    decision = _get_ci(driving, "Driving decision")
    if isinstance(decision, dict) and decision:
        lon = _get_ci(decision, "Longitudinal") or "n/a"
        lat = _get_ci(decision, "Lateral") or "n/a"
        lines.append("  Decision:")
        lines.append(_wrap(f"Longitudinal: {lon}", indent="    ", width=width))
        lines.append(_wrap(f"Lateral:      {lat}", indent="    ", width=width))

    trace = _get_ci(driving, "Reasoning trace")
    if trace:
        lines.append("  Reasoning trace:")
        lines.append(_wrap(trace, width=width))
    return lines


def format_counterfactual(counterfactual: Dict[str, Any], width: int = WRAP_WIDTH) -> List[str]:
    lines: List[str] = ["[Counterfactual reasoning]"]

    def _action_name(action: Dict[str, Any]) -> str:
        lon = _get_ci(action, "Longitudinal") or ""
        lat = _get_ci(action, "Lateral") or ""
        if lon or lat:
            return f"{lon} / {lat}".strip(" /")
        return str(_get_ci(action, "action", "name", "Alternative action") or "action")

    def _format_actions(title: str, actions: List[Any]) -> None:
        if not actions:
            return
        lines.append(f"  {title}:")
        for action in actions:
            if isinstance(action, dict):
                risk = _get_ci(action, "risk", "risk_level", "Risk level")
                outcome = _get_ci(action, "outcome", "safety_outcome", "Safety outcome")
                explanation = _get_ci(action, "explanation", "reason", "Reason", "why")
                header = f"* {_action_name(action)}"
                if risk:
                    header += f"  [{risk}]"
                lines.append(_wrap(header, indent="    ", width=width))
                if outcome:
                    lines.append(_wrap(f"outcome: {outcome}", indent="      ", width=width))
                if explanation:
                    lines.append(_wrap(explanation, indent="      ", width=width))
            else:
                lines.append(_wrap(f"* {action}", indent="    ", width=width))

    _format_actions("Alternative actions",
                    _as_list(_get_ci(counterfactual, "Alternative actions")))
    _format_actions("Safety-critical actions",
                    _as_list(_get_ci(counterfactual, "Top safety-critical actions")))
    return lines


def format_reasoning(reasoning: Dict[str, Any], show_spatial: bool = True) -> str:
    sections: List[str] = []
    spatial = _get_ci(reasoning, "Spatial")
    driving = _get_ci(reasoning, "Driving")
    counterfactual = _get_ci(reasoning, "Counterfactual")

    if show_spatial and isinstance(spatial, dict):
        sections.extend(format_spatial(spatial))
    if isinstance(driving, dict):
        sections.extend(format_driving(driving))
    if isinstance(counterfactual, dict):
        sections.extend(format_counterfactual(counterfactual))
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Colored reasoning text panel (PIL, matching the notebook layout)
# ---------------------------------------------------------------------------

def _font_path(bold: bool = False) -> Optional[str]:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidate = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", name)
    return candidate if os.path.isfile(candidate) else None


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = _font_path(bold=bold)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


_FONT_TITLE = _font(44, bold=True)
_FONT_SECTION = _font(32, bold=True)
_FONT_BODY = _font(26)
_FONT_SMALL = _font(22)


def _line_height(font: ImageFont.ImageFont) -> int:
    bbox = font.getbbox("Ag")
    return (bbox[3] - bbox[1]) + 5


def _wrap_to_width(
    text: str,
    font: ImageFont.ImageFont,
    max_w: int,
    draw: ImageDraw.ImageDraw,
) -> List[str]:
    words = str(text).split()
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        probe = " ".join(current + [word])
        if draw.textlength(probe, font=font) <= max_w:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def _risk_color(risk: str, *, for_critical: bool = False) -> Tuple[int, int, int]:
    label = (risk or "").lower()
    if for_critical:
        if "unsafe" in label:
            return UNSAFE_RGB
        if "suboptimal" in label:
            return SUBOPT_RGB
        return SAFE_RGB
    if "safe" in label and "unsafe" not in label:
        return SAFE_RGB
    if "suboptimal" in label:
        return SUBOPT_RGB
    return UNSAFE_RGB


def _render_column(
    img: Image.Image,
    x0: int,
    section_w: int,
    title: str,
    content_fn: Callable,
) -> None:
    draw = ImageDraw.Draw(img)
    margin = 22
    max_w = section_w - 2 * margin
    x = x0 + margin
    y_ref = [16]

    def put(
        text: str,
        font: ImageFont.ImageFont = _FONT_BODY,
        color: Tuple[int, int, int] = TEXT_RGB,
        indent: int = 0,
        sp: int = 3,
    ) -> None:
        for line in _wrap_to_width(text, font, max_w - indent, draw):
            if y_ref[0] + _line_height(font) + 4 > TEXT_PANEL_H - 4:
                return
            draw.text((x + indent, y_ref[0]), line, font=font, fill=color)
            y_ref[0] += _line_height(font) + sp

    def gap(px: int = 8) -> None:
        y_ref[0] += px

    def sep() -> None:
        yy = y_ref[0] + 3
        draw.line([(x, yy), (x0 + section_w - margin, yy)], fill=DIV_RGB, width=1)
        y_ref[0] += 12

    put(title, font=_FONT_TITLE, color=HEADING_RGB)
    sep()
    content_fn(put, gap, sep)


def _driving_content(driving: Dict[str, Any]):
    def fn(put, gap, sep) -> None:
        scene = _get_ci(driving, "Scene description", "Description")
        if scene:
            put("Scene", font=_FONT_SECTION, color=SUBHEAD_RGB)
            put(str(scene), indent=14)
            gap()

        comps = _get_ci(driving, "Critical components")
        if isinstance(comps, dict) and comps:
            put("Critical Components", font=_FONT_SECTION, color=SUBHEAD_RGB)
            for name, info in comps.items():
                if isinstance(info, dict):
                    ctype = info.get("Type", "")
                    header = f"* {name}" + (f"  [{ctype}]" if ctype else "")
                    put(header, color=COMP_RGB, indent=12)
                    for key, value in info.items():
                        if key == "Type":
                            continue
                        put(f"{key}: {value}", font=_FONT_SMALL, color=DIM_RGB, indent=26)
                else:
                    put(f"* {name}: {info}", color=COMP_RGB, indent=12)
            gap()

        decision = _get_ci(driving, "Driving decision")
        if isinstance(decision, dict) and decision:
            put("Decision", font=_FONT_SECTION, color=SUBHEAD_RGB)
            put(
                f"Longitudinal:  {_get_ci(decision, 'Longitudinal') or ''}",
                color=SAFE_RGB, indent=14,
            )
            put(
                f"Lateral:       {_get_ci(decision, 'Lateral') or ''}",
                color=SAFE_RGB, indent=14,
            )
            gap()

        trace = _get_ci(driving, "Reasoning trace")
        if trace:
            put("Reasoning Trace", font=_FONT_SECTION, color=SUBHEAD_RGB)
            put(str(trace), indent=14)

    return fn


def _counterfactual_content(counterfactual: Dict[str, Any]):
    def _action_header(action: Dict[str, Any]) -> str:
        lon = _get_ci(action, "Longitudinal") or ""
        lat = _get_ci(action, "Lateral") or ""
        risk = _get_ci(action, "Risk level", "risk", "risk_level") or ""
        name = f"{lon} / {lat}".strip(" /") or str(
            _get_ci(action, "action", "name", "Alternative action") or "action"
        )
        if risk:
            return f"* {name}   [{risk}]"
        return f"* {name}"

    def _reason(action: Dict[str, Any]) -> str:
        return str(_get_ci(action, "Reason", "explanation", "reason", "why") or "")

    def fn(put, gap, sep) -> None:
        alt = _as_list(_get_ci(counterfactual, "Alternative actions"))
        if alt:
            put("Alternative Actions", font=_FONT_SECTION, color=SUBHEAD_RGB)
            for action in alt:
                if not isinstance(action, dict):
                    put(f"* {action}", indent=12)
                    continue
                risk = str(_get_ci(action, "Risk level", "risk", "risk_level") or "")
                put(_action_header(action), color=_risk_color(risk), indent=12)
                reason = _reason(action)
                if reason:
                    put(reason, font=_FONT_SMALL, color=DIM_RGB, indent=28)
            gap()

        top = _as_list(_get_ci(counterfactual, "Top safety-critical actions"))
        if top:
            put("Safety-Critical Actions", font=_FONT_SECTION, color=SUBHEAD_RGB)
            for action in top:
                if not isinstance(action, dict):
                    put(f"* {action}", indent=12)
                    continue
                risk = str(_get_ci(action, "Risk level", "risk", "risk_level") or "")
                put(
                    _action_header(action),
                    color=_risk_color(risk, for_critical=True),
                    indent=12,
                )
                reason = _reason(action)
                if reason:
                    put(reason, font=_FONT_SMALL, color=DIM_RGB, indent=28)

    return fn


def render_top_section(
    driving: Optional[Dict[str, Any]],
    counterfactual: Optional[Dict[str, Any]],
    status: str = "",
) -> np.ndarray:
    """Return an RGB panel with styled Driving / Counterfactual text."""
    img = Image.new("RGB", (TEXT_PANEL_W, TEXT_PANEL_H), BG_RGB)
    draw = ImageDraw.Draw(img)
    draw.line([(TEXT_COL_W, 0), (TEXT_COL_W, TEXT_PANEL_H)], fill=DIV_RGB, width=2)

    if status:
        banner_color = SAFE_RGB if str(status).upper().startswith("NEW") else DIM_RGB
        text_w = draw.textlength(status, font=_FONT_SMALL)
        box = (TEXT_PANEL_W - text_w - 54, 16, TEXT_PANEL_W - 22, 54)
        draw.rounded_rectangle(box, radius=8, fill=BANNER_BG_RGB, outline=banner_color, width=2)
        draw.text((TEXT_PANEL_W - text_w - 38, 23), status, font=_FONT_SMALL, fill=banner_color)

    driving = driving if isinstance(driving, dict) else {}
    counterfactual = counterfactual if isinstance(counterfactual, dict) else {}
    if driving or counterfactual:
        _render_column(img, 0, TEXT_COL_W, "DRIVING", _driving_content(driving))
        _render_column(
            img, TEXT_COL_W, TEXT_COL_W, "COUNTERFACTUAL",
            _counterfactual_content(counterfactual),
        )
        return np.asarray(img)

    message = "Digesting data for reasoning..."
    for x0, title in ((0, "DRIVING"), (TEXT_COL_W, "COUNTERFACTUAL")):
        margin = 22
        draw.text((x0 + margin, 16), title, font=_FONT_TITLE, fill=HEADING_RGB)
        sep_y = 16 + _line_height(_FONT_TITLE) + 6
        draw.line(
            [(x0 + margin, sep_y), (x0 + TEXT_COL_W - margin, sep_y)],
            fill=DIV_RGB, width=1,
        )
        text_w = draw.textlength(message, font=_FONT_SECTION)
        bbox = _FONT_SECTION.getbbox(message)
        text_h = bbox[3] - bbox[1]
        draw.text(
            (x0 + (TEXT_COL_W - text_w) / 2, (TEXT_PANEL_H - text_h) / 2),
            message, font=_FONT_SECTION, fill=DIM_RGB,
        )
    return np.asarray(img)


def _box_color(category: str) -> str:
    key = (category or "object").lower()
    if key in CATEGORY_COLORS:
        return CATEGORY_COLORS[key]
    for token, color in CATEGORY_COLORS.items():
        if token in key:
            return color
    return "#ebebeb"


def _load_cam_image(clip_path: str, rel_path: str) -> Optional[np.ndarray]:
    if not rel_path:
        return None
    path = os.path.join(clip_path, rel_path)
    if not os.path.isfile(path):
        return None
    try:
        return plt.imread(path)
    except Exception:
        return None


def _draw_spatial_boxes(ax, img: Optional[np.ndarray], cam_payload: Any, draw_boxes: bool) -> None:
    if img is None:
        ax.set_facecolor("#111111")
        ax.axis("off")
        return
    ax.imshow(img)
    ax.axis("off")
    if not draw_boxes:
        return
    objects = cam_payload.get("objects", []) if isinstance(cam_payload, dict) else []
    h, w = img.shape[:2]
    sx, sy = w / CAM_BOX_REF[0], h / CAM_BOX_REF[1]
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        bbox = obj.get("detection_bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
        if x2 <= x1 or y2 <= y1:
            continue
        color = _box_color(str(obj.get("detection_label") or obj.get("category") or ""))
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor=color, linewidth=2.0,
        ))
        label = str(obj.get("detection_label") or obj.get("category") or "obj")[:18]
        center = (obj.get("detection_bbox_3d") or {}).get("center_3d_ego") or {}
        if isinstance(center, dict) and center:
            label = f"{label}  x={float(center.get('x', 0.0)):.1f} y={float(center.get('y', 0.0)):.1f}"
        ax.text(
            x1, max(8.0, y1 - 6.0), label, color="white", fontsize=7,
            bbox=dict(facecolor="black", alpha=0.55, edgecolor=color, pad=1.0),
            ha="left", va="bottom",
        )


def _overlay_ego(ax, clip_path: str, frame: Dict[str, Any]) -> None:
    rel = str(frame.get("ego_state", "") or "")
    path = os.path.join(clip_path, rel) if rel else ""
    if not path or not os.path.isfile(path):
        return
    try:
        ego = load_pickle(path)
    except Exception:
        return
    velocity = getattr(ego, "velocity", None)
    if velocity is None and isinstance(ego, dict):
        velocity = ego.get("velocity")
    acceleration = getattr(ego, "acceleration", None)
    if acceleration is None and isinstance(ego, dict):
        acceleration = ego.get("acceleration")
    velocity = velocity or {}
    acceleration = acceleration or {}
    vx = float(velocity.get("x", velocity.get("vx", 0.0)) or 0.0)
    vy = float(velocity.get("y", velocity.get("vy", 0.0)) or 0.0)
    speed = (vx * vx + vy * vy) ** 0.5
    ax_val = float(acceleration.get("x", acceleration.get("ax", 0.0)) or 0.0)
    accel_label = "Accelerating" if ax_val >= 0 else "Decelerating"
    ax.text(
        0.03, 0.97, f"{speed:.2f} m/s\n{speed * 3.6:.1f} km/h",
        transform=ax.transAxes, color="#ffdc50", fontsize=9, fontweight="bold",
        va="top", ha="left",
        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=2.0),
    )
    ax.text(
        0.97, 0.97, f"{accel_label}\n{ax_val:+.2f} m/s^2",
        transform=ax.transAxes, color="#3cdc3c" if ax_val >= 0 else "#ff5a5a",
        fontsize=9, fontweight="bold", va="top", ha="right",
        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=2.0),
    )


def _has_complete_reasoning(reasoning: Dict[str, Any]) -> bool:
    driving = _get_ci(reasoning, "Driving")
    counterfactual = _get_ci(reasoning, "Counterfactual")
    if not isinstance(driving, dict) or not _get_ci(driving, "Driving decision"):
        return False
    return isinstance(counterfactual, dict) and bool(counterfactual)


def compose_reasoning_figure(
    fig,
    clip_path: str,
    frame: Dict[str, Any],
    reasoning: Optional[Dict[str, Any]],
    status: str = "",
    draw_boxes: bool = False,
) -> None:
    """Draw driving/counterfactual text above left/front/right cameras."""
    fig.clear()
    fig.set_facecolor("#121212")
    payload = reasoning if isinstance(reasoning, dict) else {}
    driving = _get_ci(payload, "Driving") or {}
    counterfactual = _get_ci(payload, "Counterfactual") or {}
    spatial = _get_ci(payload, "Spatial") or {}
    per_camera = _get_ci(spatial, "Per_camera_results", "per_camera_results") or {}
    if not isinstance(per_camera, dict):
        per_camera = {}

    cameras = (frame.get("sensors", {}) or {}).get("cameras", {}) or {}
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.04, wspace=0.02)
    ax_text = fig.add_subplot(gs[0, :])
    ax_l = fig.add_subplot(gs[1, 0])
    ax_f = fig.add_subplot(gs[1, 1])
    ax_r = fig.add_subplot(gs[1, 2])

    top = render_top_section(
        driving if isinstance(driving, dict) else {},
        counterfactual if isinstance(counterfactual, dict) else {},
        status=status,
    )
    ax_text.imshow(top)
    ax_text.axis("off")
    ax_text.set_facecolor("#121212")

    for ax, cam_key, title in (
        (ax_l, "front_left", "LEFT"),
        (ax_f, "front", "FRONT"),
        (ax_r, "front_right", "RIGHT"),
    ):
        img = _load_cam_image(clip_path, cameras.get(cam_key, ""))
        _draw_spatial_boxes(ax, img, per_camera.get(cam_key, {}), draw_boxes)
        ax.set_title(title, color="#aaaaaa", fontsize=10, pad=2)
        if cam_key == "front":
            _overlay_ego(ax, clip_path, frame)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.02, hspace=0.04)


def render_reasoning_figure(
    clip_path: str,
    frame: Dict[str, Any],
    reasoning: Dict[str, Any],
    save_path: str,
    status: str = "",
    draw_boxes: bool = True,
) -> None:
    """Driving + counterfactual text above left/front/right cameras with 2D boxes."""
    fig = plt.figure(figsize=FIGURE_SIZE, facecolor="#121212")
    compose_reasoning_figure(
        fig, clip_path, frame, reasoning, status=status, draw_boxes=draw_boxes,
    )
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _frame_time(frame: Dict[str, Any], idx: int, fps: float) -> float:
    if "relative_time_s" in frame:
        try:
            return float(frame["relative_time_s"])
        except (TypeError, ValueError):
            pass
    return idx / max(fps, 1e-6)


def _select_reasoning_anchors(
    frames: List[Dict[str, Any]],
    target_indexes: Optional[List[int]],
    all_complete: bool,
) -> List[int]:
    reasoning_indexes = [i for i, fr in enumerate(frames) if fr.get("reasoning")]
    if not reasoning_indexes:
        return []
    if all_complete:
        return reasoning_indexes
    targets = target_indexes if target_indexes else list(DEFAULT_VIDEO_ANCHORS)
    meta_index = {int(fr.get("frame_index", i)): i for i, fr in enumerate(frames)}
    selected: set[int] = set()
    for target in targets:
        target = int(target)
        if target in meta_index and meta_index[target] in reasoning_indexes:
            selected.add(meta_index[target])
        elif target in reasoning_indexes:
            selected.add(target)
        else:
            selected.add(min(reasoning_indexes, key=lambda idx: abs(idx - target)))
    return sorted(selected)


def write_reasoning_video(
    clip_path: str,
    output_dir: str,
    *,
    fps: int = 10,
    history_seconds: float = 5.0,
    future_seconds: float = 5.0,
    pause_seconds: float = 3.0,
    target_indexes: Optional[List[int]] = None,
    all_complete: bool = False,
    overwrite: bool = False,
) -> int:
    metadata_path = os.path.join(clip_path, "metadata.json")
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    frames = sorted(meta.get("frames", []), key=lambda fr: fr.get("frame_index", 0))
    source_fps = float(meta.get("frame_rate_hz") or fps)
    frame_repeats = max(1, int(round(fps / max(source_fps, 1e-6))))
    selected = _select_reasoning_anchors(frames, target_indexes, all_complete)
    if not selected:
        return 0

    clip_name = os.path.basename(os.path.normpath(clip_path))
    out_root = os.path.join(output_dir, clip_name)
    os.makedirs(out_root, exist_ok=True)
    written = 0
    for anchor_idx in selected:
        frame = frames[anchor_idx]
        reasoning_path = os.path.join(clip_path, str(frame.get("reasoning", "")))
        if not os.path.isfile(reasoning_path):
            continue
        with open(reasoning_path) as f:
            reasoning = json.load(f)
        if not _has_complete_reasoning(reasoning):
            continue
        t0 = _frame_time(frame, anchor_idx, source_fps)
        window = [
            (i, fr) for i, fr in enumerate(frames)
            if t0 - history_seconds <= _frame_time(fr, i, source_fps) <= t0 + future_seconds
        ]
        stem = os.path.splitext(os.path.basename(str(frame.get("reasoning", f"frame_{anchor_idx}"))))[0]
        out_path = os.path.join(out_root, f"{stem}.mp4")
        if os.path.isfile(out_path) and not overwrite:
            print(f"[skip existing {out_path}]")
            written += 1
            continue
        require_ffmpeg()
        fig = plt.figure(figsize=FIGURE_SIZE, facecolor="#121212")
        writer = FFMpegWriter(
            fps=max(fps, 1), codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        with writer.saving(fig, out_path, dpi=80):
            for i, fr in window:
                t = _frame_time(fr, i, source_fps)
                if abs(t - t0) < 1e-6:
                    status = f"REASONING FRAME — paused {pause_seconds:.0f}s"
                    payload, draw_boxes, repeats = reasoning, True, max(1, int(fps * pause_seconds))
                elif t < t0:
                    status, payload, draw_boxes, repeats = (
                        "Digesting data for reasoning...", {}, False, frame_repeats,
                    )
                else:
                    status, payload, draw_boxes, repeats = (
                        "Future — reasoning unchanged", reasoning, False, frame_repeats,
                    )
                compose_reasoning_figure(
                    fig, clip_path, fr, payload, status=status, draw_boxes=draw_boxes,
                )
                for _ in range(repeats):
                    writer.grab_frame(facecolor=fig.get_facecolor())
        plt.close(fig)
        written += 1
        print(f"[video saved to {out_path}]")
    return written


# ---------------------------------------------------------------------------
# Clip iteration
# ---------------------------------------------------------------------------

def view_clip(
    clip_path: str,
    frame_index: Optional[int] = None,
    stride: int = 10,
    show_spatial: bool = True,
    save_figures: bool = False,
    output_dir: Optional[str] = None,
    write_video: bool = False,
    video_fps: int = 10,
    history_seconds: float = 5.0,
    future_seconds: float = 5.0,
    pause_seconds: float = 3.0,
    target_indexes: Optional[List[int]] = None,
    all_reasoning_videos: bool = False,
    overwrite: bool = False,
) -> int:
    """Print (and optionally render) the reasoning frames of one clip."""
    metadata_path = os.path.join(clip_path, "metadata.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {clip_path}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    clip_name = os.path.basename(os.path.normpath(clip_path))
    frames = metadata.get("frames", [])
    print("=" * WRAP_WIDTH)
    print(f"Clip: {clip_name}")
    print(f"Scenario type: {metadata.get('scenario_type', 'unknown')} | "
          f"location: {metadata.get('clip_location', 'unknown')} | "
          f"frames: {len(frames)}")
    print("=" * WRAP_WIDTH)

    if frame_index is not None:
        selected = [f for f in frames if f.get("frame_index") == frame_index]
        if not selected and 0 <= frame_index < len(frames):
            selected = [frames[frame_index]]
        if selected and not selected[0].get("reasoning"):
            # Reasoning is annotated at ~1 Hz; snap to the nearest annotated
            # frame (the video writer does the same for its anchors).
            reasoned = [f for f in frames if f.get("reasoning")]
            if reasoned:
                nearest = min(
                    reasoned,
                    key=lambda f: abs(int(f.get("frame_index", 0)) - frame_index),
                )
                print(f"frame {frame_index} has no reasoning annotation; "
                      f"showing nearest annotated frame {nearest.get('frame_index')}")
                selected = [nearest]
    else:
        # Annotations are ~1 Hz; a raw metadata stride (10 Hz) often lands only
        # on empty frames. Sample among frames that actually have reasoning.
        reasoned = [f for f in frames if f.get("reasoning")]
        selected = reasoned[::max(stride, 1)]

    num_shown = 0
    for frame in selected:
        reasoning_rel = frame.get("reasoning")
        if not reasoning_rel:
            continue
        reasoning_path = os.path.join(clip_path, reasoning_rel)
        if not os.path.isfile(reasoning_path):
            continue
        with open(reasoning_path, "r") as f:
            reasoning = json.load(f)

        print(f"\n--- frame {frame.get('frame_index')} "
              f"(t={frame.get('relative_time_s')}s) ---")
        mission = frame.get("mission_goal") or {}
        if isinstance(mission, dict) and mission.get("command"):
            print(f"Route command: {mission['command']}")
        print(format_reasoning(reasoning, show_spatial=show_spatial))
        num_shown += 1

        if save_figures:
            out_dir = output_dir or "./reasoning_viz"
            os.makedirs(os.path.join(out_dir, clip_name), exist_ok=True)
            save_path = os.path.join(
                out_dir, clip_name, f"reasoning_frame_{frame.get('frame_index'):04d}.jpg"
            )
            render_reasoning_figure(clip_path, frame, reasoning, save_path)
            print(f"[figure saved to {save_path}]")

    if write_video:
        out_dir = output_dir or "./reasoning_viz"
        anchors = None
        if frame_index is not None:
            anchors = [frame_index]
        elif target_indexes:
            anchors = target_indexes
        write_reasoning_video(
            clip_path, out_dir, fps=video_fps,
            history_seconds=history_seconds, future_seconds=future_seconds,
            pause_seconds=pause_seconds, target_indexes=anchors,
            all_complete=all_reasoning_videos and frame_index is None,
            overwrite=overwrite,
        )

    if num_shown == 0:
        print("No reasoning annotations found for the selected frames "
              "(test-split clips do not ship with reasoning files).")
    return num_shown


def main() -> None:
    parser = argparse.ArgumentParser(description="View nuReasoning reasoning annotations")
    parser.add_argument("--clip", default=None, help="Path to a single clip directory")
    parser.add_argument("--input-root", default="./dataset/data/train/part_1",
                        help="Root containing clip directories (used when --clip is not set)")
    parser.add_argument("--frame-index", type=int, default=None,
                        help="Show a single frame index (default: sample frames with --stride)")
    parser.add_argument("--stride", type=int, default=10,
                        help="Sample every Nth reasoning-annotated frame "
                             "when --frame-index is not set")
    parser.add_argument("--max-clips", type=int, default=1,
                        help="Number of clips to display when using --input-root (0 = all)")
    parser.add_argument("--no-spatial", action="store_true",
                        help="Hide the (verbose) spatial reasoning section")
    parser.add_argument("--save-figures", action="store_true",
                        help="Save composed driving/counterfactual + 3-camera figures")
    parser.add_argument("--video", action="store_true",
                        help="Write an MP4 around selected reasoning frames "
                             "(history, pause on the reasoning frame, then future)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--history-seconds", type=float, default=5.0)
    parser.add_argument("--future-seconds", type=float, default=5.0)
    parser.add_argument("--pause-seconds", type=float, default=3.0)
    parser.add_argument(
        "--anchor-frames", default="50,100,150",
        help="Comma-separated frame indexes to center videos on "
             "(ignored with --frame-index or --all-reasoning-videos)",
    )
    parser.add_argument("--all-reasoning-videos", action="store_true",
                        help="Write a video for every complete reasoning frame")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing figure/video files")
    parser.add_argument("--output-dir", default="./reasoning_viz",
                        help="Output directory for saved figures")
    args = parser.parse_args()

    if args.clip:
        clip_paths = [args.clip]
    else:
        clip_paths = discover_clips(args.input_root)
        if args.max_clips > 0:
            clip_paths = clip_paths[: args.max_clips]

    if not clip_paths:
        raise SystemExit("No clips found; check --clip / --input-root.")

    anchors = [
        int(tok.strip()) for tok in str(args.anchor_frames).split(",") if tok.strip()
    ]
    for clip_path in clip_paths:
        view_clip(
            clip_path,
            frame_index=args.frame_index,
            stride=args.stride,
            show_spatial=not args.no_spatial,
            save_figures=args.save_figures,
            output_dir=args.output_dir,
            write_video=args.video,
            video_fps=args.fps,
            history_seconds=args.history_seconds,
            future_seconds=args.future_seconds,
            pause_seconds=args.pause_seconds,
            target_indexes=anchors,
            all_reasoning_videos=args.all_reasoning_videos,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
