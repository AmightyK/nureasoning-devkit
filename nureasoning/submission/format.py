"""Official challenge submission JSON: one entry per clip.

Each clip record holds the planned ego-frame trajectory **and** the answers
to that clip's reasoning questions. The hosted evaluator later scores the
flattened trajectory and answer lists independently; packing them per clip
keeps the file aligned with the released test layout.

    {
      "meta": {
        "coordinate_frame": "ego",
        "trajectory_dt_s": 0.1,
        "trajectory_horizon_s": 5.0,
        "trajectory_steps": 51,
        "num_clips": 1000
      },
      "clips": [
        {
          "clip": "<clip_dir_name>",
          "clip_token": "<token>",
          "log_name": "<log_name>",
          "target_frame_index": 100,
          "trajectory": [[dx, dy, dtheta], ...],
          "answers": [
            {"question_id": "<id>", "answer": "A"}
          ]
        }
      ]
    }

Trajectories are 51 waypoints at 0.1 s from t = 0 (key frame) to 5 s, in the
key-frame ego frame. Reasoning answers are multiple-choice letters A–D.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SUBMISSION_COORDINATE_FRAME = "ego"
TRAJECTORY_DT_S = 0.1
TRAJECTORY_HORIZON_S = 5.0
TRAJECTORY_STEPS = 51
EXPECTED_TRAJECTORY_SHAPE = (TRAJECTORY_STEPS, 3)


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def global_to_ego(global_traj: np.ndarray, ego_pose: Any) -> np.ndarray:
    """Convert global ``(x, y, yaw)`` waypoints into key-frame ego frame."""
    if isinstance(ego_pose, dict):
        pose = ego_pose
    else:
        pose = getattr(ego_pose, "pose", {}) or {}
    ego_x = float(pose.get("x", 0.0))
    ego_y = float(pose.get("y", 0.0))
    ego_yaw = float(pose.get("yaw", 0.0))
    cos_y = float(np.cos(ego_yaw))
    sin_y = float(np.sin(ego_yaw))

    traj = np.asarray(global_traj, dtype=np.float64)
    out = np.empty_like(traj)
    dx = traj[:, 0] - ego_x
    dy = traj[:, 1] - ego_y
    out[:, 0] = dx * cos_y + dy * sin_y
    out[:, 1] = -dx * sin_y + dy * cos_y
    out[:, 2] = np.array([_wrap_to_pi(float(yaw) - ego_yaw) for yaw in traj[:, 2]])
    return out


def normalize_ego_trajectory(
    trajectory: np.ndarray,
    *,
    horizon_s: float = TRAJECTORY_HORIZON_S,
    dt_s: float = TRAJECTORY_DT_S,
) -> np.ndarray:
    """Return exactly 51 finite ego-frame waypoints on the challenge grid."""
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] < 3 or len(traj) == 0:
        raise ValueError("trajectory must have shape (N, 3+) with at least one point")
    traj = traj[:, :3]
    if not np.all(np.isfinite(traj)):
        raise ValueError("trajectory contains non-finite values")

    expected_steps = int(round(horizon_s / dt_s)) + 1
    includes_current = (
        float(np.linalg.norm(traj[0, :2])) <= 0.25
        and abs(_wrap_to_pi(float(traj[0, 2]))) <= 0.25
    )
    if includes_current:
        source = traj.copy()
        source_times = np.linspace(0.0, horizon_s, len(source), dtype=np.float64)
    else:
        source = np.vstack([np.zeros((1, 3), dtype=np.float64), traj])
        source_times = np.linspace(0.0, horizon_s, len(source), dtype=np.float64)

    source[:, 2] = np.unwrap(source[:, 2])
    target_times = np.linspace(0.0, horizon_s, expected_steps, dtype=np.float64)
    normalized = np.column_stack([
        np.interp(target_times, source_times, source[:, dim])
        for dim in range(3)
    ])
    normalized[:, 2] = (normalized[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    normalized[0] = 0.0
    if normalized.shape != (TRAJECTORY_STEPS, 3):
        raise ValueError(
            f"normalized trajectory has shape {normalized.shape}, "
            f"expected ({TRAJECTORY_STEPS}, 3)"
        )
    return normalized


def trajectory_to_list(trajectory: np.ndarray) -> List[List[float]]:
    traj = np.asarray(trajectory, dtype=np.float64)[:, :3]
    return [[float(x), float(y), float(yaw)] for x, y, yaw in traj]


def make_clip_record(
    *,
    clip: str,
    clip_token: Any,
    log_name: Any,
    target_frame_index: int,
    trajectory: Sequence[Sequence[float]],
    answers: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "clip": clip,
        "clip_token": clip_token,
        "log_name": log_name,
        "target_frame_index": int(target_frame_index),
        "trajectory": [list(point) for point in trajectory],
        "answers": [dict(row) for row in answers],
    }


def make_submission(
    clips: Sequence[Mapping[str, Any]],
    *,
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "coordinate_frame": SUBMISSION_COORDINATE_FRAME,
        "trajectory_dt_s": TRAJECTORY_DT_S,
        "trajectory_horizon_s": TRAJECTORY_HORIZON_S,
        "trajectory_steps": TRAJECTORY_STEPS,
        "num_clips": len(clips),
        "num_answers": sum(len(item.get("answers") or []) for item in clips),
    }
    if extra_meta:
        meta.update(dict(extra_meta))
    return {"meta": meta, "clips": [dict(item) for item in clips]}


def write_submission(payload: Mapping[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def remove_clip_jsonl(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def clip_partial_path(output_path: str) -> str:
    """Sidecar JSONL written after each clip so a crash keeps finished work."""
    root, _ = os.path.splitext(os.path.abspath(output_path))
    return root + ".partial.jsonl"


def append_clip_jsonl(path: str, record: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_clip_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_submission(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("challenge submission must be a JSON object")
    return payload


def flatten_submission(
    payload: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a per-clip submission into planning records and answer records.

    Accepts the official ``clips`` layout and the older split
    ``planning`` / ``reasoning`` (or ``results``) lists.
    """
    clips = payload.get("clips")
    if isinstance(clips, list):
        planning: List[Dict[str, Any]] = []
        reasoning: List[Dict[str, Any]] = []
        for raw in clips:
            if not isinstance(raw, Mapping):
                continue
            record = dict(raw)
            planning.append({
                "clip": record.get("clip"),
                "clip_token": record.get("clip_token"),
                "log_name": record.get("log_name"),
                "target_frame_index": record.get("target_frame_index"),
                "trajectory": record.get("trajectory"),
            })
            for answer in record.get("answers") or record.get("reasoning") or []:
                if not isinstance(answer, Mapping):
                    continue
                row = dict(answer)
                row.setdefault("clip", record.get("clip"))
                reasoning.append(row)
        return planning, reasoning

    planning_raw = payload.get("planning")
    if planning_raw is None:
        planning_raw = payload.get("results")
    reasoning_raw = payload.get("reasoning")
    planning = [dict(x) for x in planning_raw] if isinstance(planning_raw, list) else []
    reasoning = [dict(x) for x in reasoning_raw] if isinstance(reasoning_raw, list) else []
    return planning, reasoning


def _trajectory_ok(raw: Any) -> bool:
    try:
        traj = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return traj.shape == EXPECTED_TRAJECTORY_SHAPE and bool(np.all(np.isfinite(traj)))


def validate_submission(
    payload: Mapping[str, Any],
    *,
    data_root: Optional[str] = None,
) -> List[str]:
    """Return human-readable structural problems (empty means the file is well-formed)."""
    errors: List[str] = []
    if not isinstance(payload, Mapping):
        return ["submission must be a JSON object"]

    meta = payload.get("meta") if isinstance(payload.get("meta"), Mapping) else {}
    frame = str(meta.get("coordinate_frame") or SUBMISSION_COORDINATE_FRAME).lower()
    if frame != SUBMISSION_COORDINATE_FRAME:
        errors.append(f"meta.coordinate_frame must be '{SUBMISSION_COORDINATE_FRAME}', got {frame!r}")

    clips = payload.get("clips")
    if not isinstance(clips, list):
        errors.append("submission must contain a 'clips' list")
        return errors

    seen_clips: set[str] = set()
    seen_questions: set[str] = set()
    for i, raw in enumerate(clips):
        if not isinstance(raw, Mapping):
            errors.append(f"clips[{i}] is not an object")
            continue
        clip = str(raw.get("clip") or "").strip()
        if not clip:
            errors.append(f"clips[{i}] is missing 'clip'")
        elif clip in seen_clips:
            errors.append(f"duplicate clip {clip!r}")
        else:
            seen_clips.add(clip)

        if not _trajectory_ok(raw.get("trajectory")):
            errors.append(
                f"{clip or f'clips[{i}]'}: trajectory must be a finite "
                f"{TRAJECTORY_STEPS}x3 array"
            )

        try:
            int(raw.get("target_frame_index"))
        except (TypeError, ValueError):
            errors.append(f"{clip or f'clips[{i}]'}: missing integer target_frame_index")

        answers = raw.get("answers")
        if answers is None:
            errors.append(f"{clip or f'clips[{i}]'}: missing 'answers' list")
            continue
        if not isinstance(answers, list):
            errors.append(f"{clip or f'clips[{i}]'}: 'answers' must be a list")
            continue
        for j, answer in enumerate(answers):
            if not isinstance(answer, Mapping):
                errors.append(f"{clip}: answers[{j}] is not an object")
                continue
            qid = str(answer.get("question_id") or "").strip()
            if not qid:
                errors.append(f"{clip}: answers[{j}] is missing question_id")
            elif qid in seen_questions:
                errors.append(f"duplicate question_id {qid}")
            else:
                seen_questions.add(qid)
            if answer.get("answer") in (None, ""):
                errors.append(f"{clip}: question {qid or j} has an empty answer")

    if data_root:
        from nureasoning.common.clips import discover_clips
        from nureasoning.submission.clips import load_clip_questions

        expected = discover_clips(data_root)
        expected_names = {os.path.basename(os.path.normpath(path)) for path in expected}
        missing = sorted(expected_names - seen_clips)
        extra = sorted(seen_clips - expected_names)
        if missing:
            errors.append(f"missing {len(missing)} clips (e.g. {missing[:3]})")
        if extra:
            errors.append(f"{len(extra)} unexpected clips (e.g. {extra[:3]})")
        expected_qids: set[str] = set()
        for path in expected:
            for question in load_clip_questions(path):
                qid = str(question.get("question_id") or "").strip()
                if qid:
                    expected_qids.add(qid)
        missing_q = sorted(expected_qids - seen_questions)
        extra_q = sorted(seen_questions - expected_qids)
        if missing_q:
            errors.append(f"missing {len(missing_q)} question ids (e.g. {missing_q[:3]})")
        if extra_q:
            errors.append(f"{len(extra_q)} unexpected question ids (e.g. {extra_q[:3]})")

    return errors
