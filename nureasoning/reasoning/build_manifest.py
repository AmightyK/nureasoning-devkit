#!/usr/bin/env python3
"""Write the evaluation manifest without running any inference.

``python -m nureasoning.reasoning.evaluate`` builds the manifest itself; use this
entry point when you want to inspect or filter the question set first, then pass
``--skip-manifest`` to the evaluator.

Example:
  python -m nureasoning.reasoning.build_manifest \
    --vqa-dir ./vqa_output_val \
    --output ./reasoning_eval/manifest.jsonl \
    --keyframe-only --history-first --max-spatial-per-frame 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

from nureasoning.reasoning.modules.manifest import build_manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vqa-dir", type=Path, default=Path("./vqa_output"))
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--workspace", type=Path, default=Path("."))
    p.add_argument("--max-images", type=int, default=16)
    p.add_argument("--num-forward-frames", type=int, default=2)
    p.add_argument("--history-first", action="store_true")
    p.add_argument("--keyframe-only", action="store_true")
    p.add_argument("--max-spatial-per-frame", type=int, default=None)
    p.add_argument("--qa-seed", type=int, default=42)
    args = p.parse_args()

    n = build_manifest(
        args.vqa_dir.resolve(),
        args.output,
        workspace=args.workspace.resolve(),
        max_images=args.max_images,
        num_forward_frames=args.num_forward_frames,
        history_first=args.history_first,
        keyframe_only=args.keyframe_only,
        max_spatial_per_frame=args.max_spatial_per_frame,
        qa_seed=args.qa_seed,
    )
    print(f"Wrote {n} questions to {args.output}")


if __name__ == "__main__":
    main()
