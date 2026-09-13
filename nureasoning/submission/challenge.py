"""
Generate the official nuReasoning challenge submission.

The public test split ships without ground truth. Each clip is 10 s of
context through the key frame (index 100) and contains:

    <data-root>/<clip_name>/
        metadata.json
        cameras/CAM_M_*/...
        ego_state/*.pkl
        reasoning_questions.json

This script writes one JSON with **one entry per clip**. Each entry has:

  1. **Planning** — a 5 s / 0.1 s ego-frame trajectory.
  2. **Reasoning** — one multiple-choice answer per challenge question.

One-model path: a single nuVLA checkpoint trained with QA
(``--reasoning_mode vqa --vqa_root``) does both jobs:

    python -m nureasoning.submission.challenge \\
        --data-root ./dataset/data/test \\
        --provider nuvla \\
        --checkpoint-dir ./nureasoning_vla_qa_workspace/final \\
        --output challenge_submission.json

Split path: nuVLA (or a baseline) for trajectories, plus a separate reasoning
model for answers:

    python -m nureasoning.submission.challenge \\
        --data-root ./dataset/data/test \\
        --planning-provider vla --checkpoint-dir ./nureasoning_vla_workspace/final \\
        --reasoning-provider api --api-model nureasoning-4b-sft \\
        --output challenge_submission.json
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from nureasoning.common.clips import discover_clips
from nureasoning.nuvla.trajectory_provider import add_planning_prompt_arguments
from nureasoning.planning.baselines import get_baseline
from nureasoning.submission.clips import (
    load_clip_metadata,
    load_clip_sample,
    load_key_frame_ego_state,
    select_key_frame_idx,
)
from nureasoning.submission.format import (
    append_clip_jsonl,
    clip_partial_path,
    global_to_ego,
    load_clip_jsonl,
    make_clip_record,
    remove_clip_jsonl,
    make_submission,
    normalize_ego_trajectory,
    trajectory_to_list,
    validate_submission,
    write_submission,
)
from nureasoning.submission.reasoning import (
    APIChallengeAnswerer,
    BaseChallengeAnswerer,
    answer_row,
    build_reasoning_answerer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_provider(args: argparse.Namespace) -> Callable[..., Optional[np.ndarray]]:
    if args.provider == "vla":
        if not args.checkpoint_dir:
            raise SystemExit("--checkpoint-dir is required for --planning-provider vla")
        from nureasoning.nuvla.trajectory_provider import VLATrajectoryProvider

        return VLATrajectoryProvider(
            checkpoint_dir=args.checkpoint_dir,
            num_inference_steps=args.num_inference_steps,
            device=args.device,
            planning_prompt=getattr(args, "planning_prompt", "auto"),
            reasoning_format=getattr(args, "planning_reasoning_format", None),
        )
    return get_baseline(args.provider)


def plan_clip(
    provider: Callable[..., Optional[np.ndarray]],
    clip_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return one ego-frame trajectory record for a clip directory."""
    clip_name = os.path.basename(os.path.normpath(clip_path))
    meta = metadata if metadata is not None else load_clip_metadata(clip_path)
    frames = meta.get("frames") or []
    key_frame_idx = select_key_frame_idx(frames, meta)
    ego_state = load_key_frame_ego_state(clip_path, frames[key_frame_idx])

    trajectory = provider(clip_path, key_frame_idx, ego_state)
    if trajectory is None or len(trajectory) == 0:
        raise ValueError("provider returned an empty trajectory")
    trajectory = global_to_ego(
        np.asarray(trajectory, dtype=np.float64)[:, :3],
        ego_state,
    )
    trajectory = normalize_ego_trajectory(trajectory)
    return {
        "clip": clip_name,
        "clip_token": meta.get("clip_token"),
        "log_name": meta.get("log_name"),
        "target_frame_index": key_frame_idx,
        "trajectory": trajectory_to_list(trajectory),
    }


def _answer_one_sample(
    answerer: BaseChallengeAnswerer,
    sample: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], int]:
    results: List[Dict[str, Any]] = []
    failed = 0
    for question in sample.get("questions") or []:
        qid = question.get("question_id")
        try:
            pred = answerer.answer(question, sample)
            results.append(answer_row(question, sample, pred))
        except Exception as exc:
            failed += 1
            logger.error("Failed on question %s (%s): %s", qid, sample.get("clip"), exc)
            results.append(answer_row(question, sample, {}, error=str(exc)))
    return results, failed


def _release_clip_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_challenge_submission(
    data_root: str,
    output_path: str,
    *,
    planning_provider: Any = None,
    reasoning_answerer: Optional[BaseChallengeAnswerer] = None,
    nuvla_predictor: Any = None,
    max_clips: int = 0,
    skip_planning: bool = False,
    skip_reasoning: bool = False,
    allow_partial: bool = False,
) -> Dict[str, Any]:
    clip_paths = discover_clips(data_root, max_clips=max_clips)
    if not clip_paths:
        raise FileNotFoundError(f"No clips found under {data_root}")
    logger.info("Found %d clips under %s", len(clip_paths), data_root)

    partial_path = clip_partial_path(output_path)
    already = load_clip_jsonl(partial_path)
    done_clips = {str(row.get("clip") or "") for row in already}
    seen_question_ids: set[str] = set()
    for row in already:
        for answer in row.get("answers") or []:
            qid = str((answer or {}).get("question_id") or "").strip()
            if qid:
                seen_question_ids.add(qid)
    if already:
        logger.info(
            "Resuming from %s (%d clips already written)",
            partial_path,
            len(already),
        )

    planning_failed = 0
    reasoning_failed = 0
    n_total = len(clip_paths)
    for i, clip_path in enumerate(clip_paths, 1):
        sample = load_clip_sample(clip_path)
        clip_name = sample["clip"]
        if not sample.get("clip_path"):
            raise ValueError(f"clip {clip_name} has no clip_path")
        for question in sample.get("questions") or []:
            qid = str(question.get("question_id") or "").strip()
            if not qid:
                raise ValueError(
                    f"question in {sample.get('file_path')} has no question_id"
                )
            if qid in seen_question_ids and clip_name not in done_clips:
                raise ValueError(f"duplicate question_id in input: {qid}")
            if clip_name not in done_clips:
                seen_question_ids.add(qid)

        if clip_name in done_clips:
            logger.info("[%d/%d] %s: already saved, skipping", i, n_total, clip_name)
            continue

        key_frame_idx = int(sample.get("key_frame_index") or 0)
        trajectory: List[List[float]] = []
        answers: List[Dict[str, Any]] = []
        clip_plan_failed = False
        clip_reason_failed = 0

        try:
            if skip_planning:
                pass
            elif nuvla_predictor is not None:
                predicted = nuvla_predictor.predict_clip(sample)
                trajectory = predicted["trajectory"]
                answers = predicted["answers"] if not skip_reasoning else []
                clip_reason_failed = int(predicted.get("num_failed") or 0)
            else:
                if planning_provider is None:
                    raise ValueError("planning_provider is required unless skip_planning=True")
                planned = plan_clip(
                    planning_provider,
                    clip_path,
                    metadata=sample.get("metadata"),
                )
                trajectory = planned["trajectory"]
                key_frame_idx = int(planned["target_frame_index"])
        except Exception as exc:
            clip_plan_failed = True
            planning_failed += 1
            logger.error("[%d/%d] %s planning FAILED (%s)", i, n_total, clip_name, exc)
            if not allow_partial:
                raise RuntimeError(
                    f"planning failed for clip {clip_name}: {exc}"
                ) from exc

        if skip_reasoning:
            answers = []
        elif nuvla_predictor is None:
            if reasoning_answerer is None:
                raise ValueError("reasoning_answerer is required unless skip_reasoning=True")
            if isinstance(reasoning_answerer, APIChallengeAnswerer):
                answers = reasoning_answerer.answer_samples([sample]).get(clip_name, [])
                clip_reason_failed = sum(1 for row in answers if row.get("error"))
            else:
                answers, clip_reason_failed = _answer_one_sample(reasoning_answerer, sample)

        reasoning_failed += clip_reason_failed
        if clip_reason_failed and not allow_partial:
            raise RuntimeError(
                f"reasoning failed for {clip_reason_failed} questions on clip {clip_name}"
            )

        if not clip_plan_failed:
            record = make_clip_record(
                clip=clip_name,
                clip_token=sample.get("clip_token"),
                log_name=sample.get("log_name"),
                target_frame_index=key_frame_idx,
                trajectory=trajectory,
                answers=answers,
            )
            append_clip_jsonl(partial_path, record)
            done_clips.add(clip_name)
            logger.info(
                "[%d/%d] %s: %d waypoints, %d answers -> %s",
                i,
                n_total,
                clip_name,
                len(trajectory),
                len(answers),
                partial_path,
            )
        else:
            logger.info(
                "[%d/%d] %s: %d waypoints, %d answers",
                i,
                n_total,
                clip_name,
                len(trajectory),
                len(answers),
            )
        _release_clip_memory()

    total_failed = planning_failed + reasoning_failed
    if total_failed and not allow_partial:
        raise RuntimeError(
            f"challenge generation failed for {planning_failed} planning clips "
            f"and {reasoning_failed} reasoning questions; no file was written"
        )

    by_name = {str(row.get("clip") or ""): row for row in load_clip_jsonl(partial_path)}
    clips = [
        by_name[os.path.basename(os.path.normpath(path))]
        for path in clip_paths
        if os.path.basename(os.path.normpath(path)) in by_name
    ]
    submission = make_submission(
        clips,
        extra_meta={
            "num_planning_failed": planning_failed,
            "num_reasoning_failed": reasoning_failed,
        },
    )
    check_root = None
    if not allow_partial and max_clips <= 0:
        check_root = data_root
    errors = validate_submission(submission, data_root=check_root)
    if errors and not allow_partial:
        preview = "; ".join(errors[:5])
        raise RuntimeError(f"submission failed validation: {preview}")
    if errors:
        logger.warning("Submission has %d validation issues: %s", len(errors), errors[:5])

    write_submission(submission, output_path)
    remove_clip_jsonl(partial_path)
    logger.info(
        "Wrote challenge submission to %s (clips=%d, answers=%d)",
        output_path,
        len(clips),
        submission["meta"]["num_answers"],
    )
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the nuReasoning challenge submission "
                    "(one clip entry = trajectory + reasoning answers)"
    )
    parser.add_argument(
        "--data-root",
        default="./dataset/data/test",
        help="Test / challenge inference root. Each clip must contain "
             "metadata.json, cameras/, ego_state/, and reasoning_questions.json.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["nuvla", "split"],
                        help="nuvla = one checkpoint for planning and QA; "
                             "split = --planning-provider + --reasoning-provider. "
                             "Default: nuvla if --checkpoint-dir is set and the "
                             "split providers were left at their defaults, else split.",
    )
    parser.add_argument(
        "--planning-provider",
        default="constant_velocity",
        help="Planning provider for --provider split: 'vla' or a registered "
             "baseline (constant_velocity, uniad, diffusion_drive)",
    )
    parser.add_argument(
        "--reasoning-provider",
        default="stub",
        choices=["stub", "vlm", "api"],
        help="stub = dry-run; vlm = in-process HF/nuVLA VLM (same image "
             "layout as nureasoning.nuvla.train); "
             "api = OpenAI-compatible server (trained reasoning model)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="nuVLA checkpoint directory (required for --provider nuvla "
             "and for --planning-provider vla)",
    )
    parser.add_argument(
        "--vlm-model-path",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Base VLM path when no checkpoint is provided for --reasoning-provider vlm",
    )
    parser.add_argument("--num-inference-steps", type=int, default=5,
                        help="Flow-matching inference steps for nuVLA planning")
    add_planning_prompt_arguments(parser, hyphen=True)
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the stub answerer")
    parser.add_argument("--api-model", default=os.environ.get("VLLM_MODEL", "nureasoning-4b-sft"),
                        help="Served model name for --reasoning-provider api")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-urls", default=os.environ.get("API_URLS", ""),
                        help="Comma-separated OpenAI-compatible URLs (multi-replica)")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--concurrent-per-url", type=int, default=8)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--num-forward-frames", type=int, default=2)
    parser.add_argument(
        "--history-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Time-major image order for --reasoning-provider api (default: true)",
    )
    parser.add_argument(
        "--output",
        default="./challenge_submission.json",
        help="Output challenge submission JSON path",
    )
    parser.add_argument("--max-clips", type=int, default=0, help="Limit clips (0 = all)")
    parser.add_argument("--skip-planning", action="store_true")
    parser.add_argument("--skip-reasoning", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write debugging output even when planning or reasoning inference fails",
    )
    parser.add_argument(
        "--validate-only",
        metavar="FILE",
        default=None,
        help="Validate an existing submission JSON against --data-root and exit",
    )
    args = parser.parse_args()

    if args.validate_only:
        from nureasoning.submission.format import load_submission

        payload = load_submission(args.validate_only)
        expected = discover_clips(args.data_root) if os.path.isdir(args.data_root) else []
        n_expected = len(expected)
        n_got = len(payload.get("clips") or [])
        check_root = args.data_root if expected and n_got >= n_expected else None
        if expected and n_got < n_expected:
            logger.warning(
                "Partial submission (%d/%d clips); skipping coverage check against %s",
                n_got, n_expected, args.data_root,
            )
        errors = validate_submission(payload, data_root=check_root)
        if errors:
            raise SystemExit("Invalid submission:\n  " + "\n  ".join(errors))
        print(
            f"OK: {args.validate_only} "
            f"({payload.get('meta', {}).get('num_clips', '?')} clips, "
            f"{payload.get('meta', {}).get('num_answers', '?')} answers)"
        )
        return

    if not os.path.isdir(args.data_root):
        raise SystemExit(f"Challenge data root not found: {args.data_root}")
    if args.skip_planning and args.skip_reasoning:
        raise SystemExit("Cannot skip both planning and reasoning.")

    provider_mode = args.provider
    if provider_mode is None:
        split_requested = (
            args.planning_provider != "constant_velocity"
            or args.reasoning_provider != "stub"
        )
        if args.checkpoint_dir and not split_requested:
            provider_mode = "nuvla"
        else:
            provider_mode = "split"

    nuvla_predictor = None
    planning_provider = None
    reasoning_answerer = None

    if provider_mode == "nuvla":
        if not args.checkpoint_dir:
            raise SystemExit("--checkpoint-dir is required for --provider nuvla")
        from nureasoning.submission.nuvla import NuVLAClipPredictor

        nuvla_predictor = NuVLAClipPredictor(
            checkpoint_dir=args.checkpoint_dir,
            num_inference_steps=args.num_inference_steps,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            planning_prompt=args.planning_prompt,
            planning_reasoning_format=args.planning_reasoning_format,
        )
    else:
        planning_args = argparse.Namespace(
            provider=args.planning_provider,
            checkpoint_dir=args.checkpoint_dir,
            num_inference_steps=args.num_inference_steps,
            device=args.device,
            planning_prompt=args.planning_prompt,
            planning_reasoning_format=args.planning_reasoning_format,
        )
        if not args.skip_planning:
            planning_provider = build_provider(planning_args)
        if not args.skip_reasoning:
            reasoning_answerer = build_reasoning_answerer(args)

    run_challenge_submission(
        data_root=args.data_root,
        output_path=args.output,
        planning_provider=planning_provider,
        reasoning_answerer=reasoning_answerer,
        nuvla_predictor=nuvla_predictor,
        max_clips=args.max_clips,
        skip_planning=args.skip_planning,
        skip_reasoning=args.skip_reasoning,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()
