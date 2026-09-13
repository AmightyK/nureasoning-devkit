#!/usr/bin/env python3
"""Turn generated VQA files into a supervised fine-tuning JSONL.

Input is the output directory of ``python -m nureasoning.vqa.generate``; output
is one JSON object per training example, holding the ordered image paths, a
matching ``image_contexts`` list (nuVLA-style ``t=…, {camera} camera.`` captions),
the user prompt (question, choices and answer-format instruction) and the
assistant target string.

The answer-format instruction appended here is the same one evaluation uses, so
the model is trained to answer in exactly the format the scorer parses.

Example:
  python -m nureasoning.reasoning.build_sft \
    --vqa-dir ./vqa_output \
    --output ./reasoning_workspace/sft_train.jsonl \
    --keyframe-only --num-forward-frames 2 --max-images 16 --history-first \
    --max-spatial-per-frame 10
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from nureasoning.reasoning.modules.image_selection import (
    collect_image_slots,
    resolve_image_slots,
)
from nureasoning.reasoning.modules.prompt_format import build_instruction_suffix
from nureasoning.reasoning.modules.sampling import (
    frame_timestamp_of,
    select_questions,
    vqa_paths,
)


def assistant_target(question: dict[str, Any]) -> str:
    """The supervision string, formatted the way the evaluator parses it."""
    qtype = str(question.get("question_type") or "").lower()
    answer = question.get("answer")
    answer_text = question.get("answer_text")

    if qtype == "choice":
        if isinstance(answer, list):
            return ",".join(str(x).strip() for x in answer)
        if isinstance(answer, str):
            return answer.strip()
        if isinstance(answer_text, str):
            return answer_text.strip()
        return json.dumps(answer, ensure_ascii=False)

    if qtype == "numerical":
        if isinstance(answer, list):
            return json.dumps(answer, ensure_ascii=False)
        return "" if answer is None else str(answer)

    if isinstance(answer_text, str) and answer_text.strip():
        return answer_text.strip()
    if isinstance(answer, str):
        return answer.strip()
    if answer is not None:
        if isinstance(answer, (list, dict)):
            return json.dumps(answer, ensure_ascii=False)
        return str(answer)
    return ""


def build_prompt(question: dict[str, Any]) -> str:
    text = question.get("question") or ""
    choices = question.get("choices")
    if isinstance(choices, dict) and choices:
        lines = [f"{k}) {v}" for k, v in sorted(choices.items())]
        text = text + "\n\nChoices:\n" + "\n".join(lines)
    sample_like = {"question_type": question.get("question_type"),
                   "gold": {"answer": question.get("answer")}}
    return text + build_instruction_suffix(str(question.get("question_type")), sample_like)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vqa-dir", type=Path, default=Path("./vqa_output"),
                   help="Output directory of nureasoning.vqa.generate")
    p.add_argument("-o", "--output", type=Path, required=True,
                   help="Destination JSONL")
    p.add_argument("--workspace", type=Path, default=Path("."),
                   help="Root for resolving workspace-relative image paths "
                        "(generated VQA files use absolute paths)")
    p.add_argument("--max-images", type=int, default=16,
                   help="Cap on images per example (8 cameras x 2 frames = 16)")
    p.add_argument("--num-forward-frames", type=int, default=2,
                   help="Keep only the N most recent frames per camera "
                        "(2 = current frame plus the one 1 s earlier)")
    p.add_argument("--history-first", action="store_true",
                   help="Time-major image order: all cameras at the oldest timestamp "
                        "first, current timestamp last. Required for the mixed-resolution "
                        "history option in training, and must match evaluation")
    p.add_argument("--keyframe-only", action="store_true",
                   help="One frame per clip instead of every annotated frame")
    p.add_argument("--max-spatial-per-frame", type=int, default=None,
                   help="Cap Spatial questions per frame; other categories kept in full")
    p.add_argument("--max-qa-per-frame", type=int, default=None,
                   help="Cap total questions per frame, stratified by question type")
    p.add_argument("--qa-seed", type=int, default=42)
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument("--limit-rows", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.vqa_dir.is_dir():
        raise SystemExit(f"VQA directory not found: {args.vqa_dir}")

    paths = vqa_paths(args.vqa_dir, keyframe_only=args.keyframe_only)
    kind = "keyframes" if args.keyframe_only else "VQA files"
    print(f"Scanning {args.vqa_dir}: {len(paths)} {kind}")
    print(f"  Images: multi-view multi-frame, max={args.max_images}, "
          f"frames/camera={args.num_forward_frames}, "
          f"order={'time-major' if args.history_first else 'camera-major'}, "
          f"captions=nuVLA")
    if args.max_spatial_per_frame is not None:
        print(f"  Spatial cap: {args.max_spatial_per_frame} questions/frame "
              f"(qa_seed={args.qa_seed})")
    elif args.max_qa_per_frame is not None:
        print(f"  Question cap: {args.max_qa_per_frame}/frame, stratified "
              f"(qa_seed={args.qa_seed})")

    if args.limit_files:
        paths = paths[: args.limit_files]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_files = 0
    n_rows = 0
    type_counts: dict[str, int] = defaultdict(int)
    t0 = time.time()

    with args.output.open("w", encoding="utf-8") as out:
        for i, path in enumerate(paths):
            with path.open(encoding="utf-8") as f:
                doc = json.load(f)
            clip_dir = path.parent.name
            frame_ts = frame_timestamp_of(doc, path)

            images, image_contexts = resolve_image_slots(
                collect_image_slots(
                    doc,
                    max_images=args.max_images,
                    num_forward_frames=args.num_forward_frames,
                    history_first=args.history_first,
                ),
                args.workspace,
            )
            if not images:
                continue

            questions = select_questions(
                doc,
                clip_dir=clip_dir,
                frame_timestamp=frame_ts,
                max_spatial_per_frame=args.max_spatial_per_frame,
                max_qa_per_frame=args.max_qa_per_frame,
                qa_seed=args.qa_seed,
            )

            for q in questions:
                target = assistant_target(q)
                if not target:
                    continue
                record = {
                    "sample_id": f"{clip_dir}/{frame_ts}/{q.get('question_id')}",
                    "vqa_path": str(path),
                    "clip_dir": clip_dir,
                    "frame_timestamp": frame_ts,
                    "question_id": q.get("question_id"),
                    "question_type": q.get("question_type"),
                    "category": q.get("category"),
                    "subcategory": q.get("subcategory"),
                    "images": images,
                    "image_contexts": image_contexts,
                    "question": build_prompt(q),
                    "choices": q.get("choices"),
                    "gold": {
                        "answer": q.get("answer"),
                        "answer_text": q.get("answer_text"),
                        "tolerance": q.get("tolerance"),
                        "answer_format": q.get("answer_format"),
                    },
                    "assistant": target,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                type_counts[str(q.get("question_type") or "unknown")] += 1
                n_rows += 1
                if args.limit_rows is not None and n_rows >= args.limit_rows:
                    break
            n_files += 1

            if (i + 1) % 100 == 0 or i + 1 == len(paths):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (len(paths) - i - 1) / rate if rate > 0 else 0.0
                print(f"  [{i+1}/{len(paths)}] {n_rows} rows | {rate:.1f} files/s | "
                      f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s", flush=True)

            if args.limit_rows is not None and n_rows >= args.limit_rows:
                break

    print(f"\nWrote {n_rows} rows from {n_files} VQA files -> {args.output} "
          f"({time.time() - t0:.1f}s)")
    if n_rows:
        print("  Question types:")
        for t, c in sorted(type_counts.items()):
            print(f"    {t}: {c} ({100 * c / n_rows:.1f}%)")


if __name__ == "__main__":
    main()
