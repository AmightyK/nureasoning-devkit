#!/usr/bin/env python3
"""Score a VLM on nuReasoning questions, end to end.

The pipeline builds a manifest from the VQA files, queries an OpenAI-compatible
server for an answer to every question, scores each answer against the ground
truth, and writes stratified metrics.

Start a server first, e.g. for a merged checkpoint on 8 GPUs:

  vllm serve <merged_dir> --served-model-name nureasoning-4b-sft \
    --tensor-parallel-size 8 --max-model-len 24576 --dtype bfloat16 \
    --mm-processor-kwargs '{"max_pixels": 200704}' --enable-prefix-caching

Then:

  python -m nureasoning.reasoning.evaluate \
    --vqa-dir ./vqa_output_val --model nureasoning-4b-sft \
    --output-root ./reasoning_eval \
    --keyframe-only --history-first --max-spatial-per-frame 10

Image settings must match the ones used to build the training data, otherwise
the model sees a different view layout than it was trained on.

Outputs, under ``<output-root>/<model-name>/``:
  reasoning_results/manifest.jsonl                 questions and image lists
  reasoning_results/predictions.jsonl              raw answers plus per-sample metrics
  reasoning_results/evaluation_vs_groundtruth.jsonl  answer next to ground truth
  reasoning_results/evaluation_results.json        headline numbers and artifact paths
  metrics_eval/metrics.yaml                        full stratified metrics
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from nureasoning.reasoning.modules.gpu_utils import count_gpus, parse_api_urls
from nureasoning.reasoning.modules.inference import aggregate, run_inference
from nureasoning.reasoning.modules.manifest import build_manifest, load_manifest
from nureasoning.reasoning.modules.metrics import primary_metric_value


def sanitize_model_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_").replace(" ", "_")


def merge_with_ground_truth(
    samples: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample, pred in zip(samples, predictions, strict=True):
        metrics = pred.get("metrics") or {}
        rows.append({
            "sample_id": sample.get("sample_id"),
            "question_id": sample.get("question_id"),
            "clip_dir": sample.get("clip_dir"),
            "frame_timestamp": sample.get("frame_timestamp"),
            "question_type": sample.get("question_type"),
            "category": sample.get("category"),
            "subcategory": sample.get("subcategory"),
            "question": sample.get("question"),
            "choices": sample.get("choices"),
            "ground_truth": sample.get("gold"),
            "prediction_raw": pred.get("prediction_raw"),
            "latency_s": pred.get("latency_s"),
            "error": pred.get("error"),
            "metrics": metrics,
            "primary_metric": primary_metric_value(
                metrics, str(sample.get("question_type"))
            ),
        })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vqa-dir", type=Path, default=Path("./vqa_output"),
                   help="Output directory of nureasoning.vqa.generate for the split to score")
    p.add_argument("--workspace", type=Path, default=Path("."),
                   help="Root for resolving workspace-relative image paths")
    p.add_argument("--output-root", type=Path, default=Path("./reasoning_eval"))
    p.add_argument("--model", default=os.environ.get("VLLM_MODEL", "nureasoning-4b-sft"),
                   help="Model name the server answers to (--served-model-name)")
    p.add_argument("--model-name", default=None,
                   help="Directory name under --output-root (default: sanitized --model)")

    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL",
                                                        "http://127.0.0.1:8000/v1"))
    p.add_argument("--api-urls", default=os.environ.get("API_URLS", ""),
                   help="Comma-separated server URLs to spread load over, e.g. one "
                        "single-GPU vLLM per GPU")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--concurrent-per-url", type=int, default=None,
                   help="In-flight requests per server (default: 8)")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Extra attempts per question on transient 5xx / 429 / timeouts")

    p.add_argument("--max-images", type=int, default=16)
    p.add_argument("--num-forward-frames", type=int, default=2,
                   help="Most recent frames per camera (2 = current plus 1 s earlier)")
    p.add_argument("--history-first", action="store_true",
                   help="Time-major image order; must match the training data")
    p.add_argument("--keyframe-only", action="store_true",
                   help="One frame per clip")
    p.add_argument("--max-spatial-per-frame", type=int, default=None,
                   help="Cap Spatial questions per frame; the subset is deterministic "
                        "across models and runs, so scores stay comparable")
    p.add_argument("--qa-seed", type=int, default=42)

    p.add_argument("--limit", type=int, default=None, help="Debug: score only the first N")
    p.add_argument("--thinking", action="store_true",
                   help="Leave the backbone's thinking mode on (slower, longer answers)")
    p.add_argument("--skip-manifest", action="store_true",
                   help="Reuse the manifest already in the output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_tag = args.model_name or sanitize_model_name(args.model)
    out_dir = args.output_root.resolve() / model_tag
    results_dir = out_dir / "reasoning_results"
    metrics_dir = out_dir / "metrics_eval"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = results_dir / "manifest.jsonl"
    predictions_path = results_dir / "predictions.jsonl"
    vs_gt_path = results_dir / "evaluation_vs_groundtruth.jsonl"
    summary_path = results_dir / "evaluation_results.json"
    metrics_yaml_path = metrics_dir / "metrics.yaml"

    urls = parse_api_urls(args.api_urls) or [args.base_url.rstrip("/")]
    if args.concurrent_per_url is None:
        # Multimodal prompts are large; very high concurrency tends to make the
        # server return 500s or run out of KV cache.
        args.concurrent_per_url = 8

    t_start = time.perf_counter()
    if args.skip_manifest:
        if not manifest_path.is_file():
            raise SystemExit(f"--skip-manifest given but no manifest at {manifest_path}")
        print(f"Reusing manifest: {manifest_path}")
    else:
        n = build_manifest(
            args.vqa_dir.resolve(),
            manifest_path,
            workspace=args.workspace.resolve(),
            max_images=args.max_images,
            num_forward_frames=args.num_forward_frames,
            history_first=args.history_first,
            keyframe_only=args.keyframe_only,
            max_spatial_per_frame=args.max_spatial_per_frame,
            qa_seed=args.qa_seed,
        )
        print(f"Manifest: {n} questions -> {manifest_path}")

    samples = load_manifest(manifest_path, args.limit)
    if not samples:
        raise SystemExit(
            f"Manifest {manifest_path} is empty. Check --vqa-dir and that the VQA "
            "files contain questions with resolvable image paths."
        )

    n_gpus = count_gpus()
    print(f"GPUs visible: {n_gpus}; servers: {len(urls)}; "
          f"concurrent_per_url={args.concurrent_per_url}; questions={len(samples)}")
    if len(urls) == 1 and n_gpus > 1:
        print(f"Note: one server URL. For multi-GPU inference either serve with "
              f"--tensor-parallel-size {n_gpus}, or start one server per GPU and "
              f"list them in --api-urls.")

    predictions = asyncio.run(
        run_inference(
            samples,
            urls,
            args.model,
            args.concurrent_per_url,
            args.api_key,
            args.thinking,
            max_retries=args.max_retries,
            predictions_path=predictions_path,
        )
    )

    with vs_gt_path.open("w", encoding="utf-8") as f:
        for row in merge_with_ground_truth(samples, predictions):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = aggregate(predictions)
    summary["config"] = {
        "model": args.model,
        "model_name_dir": model_tag,
        "vqa_dir": str(args.vqa_dir),
        "api_urls": urls,
        "num_gpus_detected": n_gpus,
        "concurrent_per_url": args.concurrent_per_url,
        "max_retries": args.max_retries,
        "max_images": args.max_images,
        "num_forward_frames": args.num_forward_frames,
        "history_first": args.history_first,
        "keyframe_only": args.keyframe_only,
        "max_spatial_per_frame": args.max_spatial_per_frame,
        "qa_seed": args.qa_seed,
        "thinking": args.thinking,
        "manifest": str(manifest_path),
        "total_questions_evaluated": len(samples),
        "elapsed_s_pipeline": time.perf_counter() - t_start,
    }

    results = {
        "model": args.model,
        "output_directory": str(out_dir),
        "ground_truth_source": "nuReasoning VQA manifest (gold fields from *_vqa.json)",
        "artifacts": {
            "manifest_jsonl": str(manifest_path),
            "predictions_jsonl": str(predictions_path),
            "evaluation_vs_groundtruth_jsonl": str(vs_gt_path),
            "metrics_yaml": str(metrics_yaml_path),
        },
        "counts": summary["counts"],
        "primary_metric_mean_by_question_type": summary["primary_metric_mean_by_question_type"],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with metrics_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                # Round-trip through JSON so PyYAML only ever sees plain types.
                "metrics": json.loads(json.dumps(summary)),
                "evaluation_summary": results,
            },
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    print(json.dumps(summary["counts"], indent=2))
    print("Primary metric by question type:",
          summary["primary_metric_mean_by_question_type"])
    print(f"Wrote {summary_path}")
    print(f"Wrote {metrics_yaml_path}")


if __name__ == "__main__":
    main()
