"""Evaluation entrypoint for the JEPA planning pipeline.

The native mode reports sparse-trajectory ADE/FDE and wrap-aware heading error
on a held-out planning split.  Only :class:`ObservationBatch` is passed into
``PlanningModel.predict``; the future tensor remains an evaluation target.
The optional benchmark mode delegates map/agent metrics to the existing local
benchmark without modifying it.

Examples::

    python -m nureasoning.jepa_planning.evaluate metrics \
      --checkpoint outputs/jepa_planning/joint.pt \
      --data-root ./dataset/data/validation \
      --device cuda --seeds 42,43,44 \
      --output outputs/jepa_planning/evaluation.json

    python -m nureasoning.jepa_planning.evaluate benchmark \
      --checkpoint outputs/jepa_planning/joint.pt \
      --data-root ./dataset/data/validation \
      --device cuda --seed 42 --key-frame-index 100 \
      --output outputs/jepa_planning/benchmark.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from .config import DataConfig
from .contracts import ObservationBatch
from .data import PlanningDataset, PlanningSample, planning_collate_fn
from .trajectory_provider import (
    JEPAPlanningTrajectoryProvider,
    load_evaluation_model,
)


def trajectory_error_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float | int]:
    """Return additive raw-coordinate trajectory errors for aggregation."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must have shape [B,T,3]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction or target contains non-finite values")

    displacement = torch.linalg.vector_norm(
        prediction[..., :2] - target[..., :2], dim=-1
    )
    heading_delta = prediction[..., 2] - target[..., 2]
    heading_error = torch.abs(
        torch.atan2(torch.sin(heading_delta), torch.cos(heading_delta))
    )
    return {
        "position_sum_m": float(displacement.sum().item()),
        "position_count": int(displacement.numel()),
        "final_position_sum_m": float(displacement[:, -1].sum().item()),
        "trajectory_count": int(displacement.shape[0]),
        "heading_sum_rad": float(heading_error.sum().item()),
        "heading_count": int(heading_error.numel()),
    }


def _observation_sample(sample: PlanningSample) -> tuple[PlanningSample, torch.Tensor]:
    if sample.future is None:
        raise ValueError("metric evaluation sample does not contain a future target")
    observation_sample = PlanningSample(
        video=sample.video,
        camera_ids=sample.camera_ids,
        frame_times_s=sample.frame_times_s,
        ego_state=sample.ego_state,
        history=sample.history,
        command_id=sample.command_id,
        sample_id=sample.sample_id,
        future=None,
    )
    return observation_sample, sample.future


def _observation_only(sample: PlanningSample) -> tuple[ObservationBatch, torch.Tensor]:
    observation_sample, target = _observation_sample(sample)
    batch = planning_collate_fn([observation_sample])
    if not isinstance(batch, ObservationBatch) or hasattr(batch, "future"):
        raise TypeError("metric inference must receive an observation-only batch")
    return batch, target.unsqueeze(0)


def _sample_seed(seed: int, sample_id: str) -> int:
    offset = int.from_bytes(
        hashlib.sha256(sample_id.encode("utf-8")).digest()[:4], "little"
    )
    return (seed + offset) % (2**63 - 1)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    fraction = index - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def evaluate_dataset(
    model: Any,
    dataset: Any,
    *,
    device: str | torch.device = "cpu",
    num_inference_steps: int | None = None,
    seed: int = 42,
    max_samples: int = 0,
    intent_mode: str = "predicted",
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Evaluate one seed while accounting for preprocessing/inference failures.

    The dataset supplies targets for scoring, but a newly constructed
    observation-only sample is the sole model input.  ``predict`` is therefore
    unable to access the future target and initializes from its normal pure
    Gaussian inference path.
    """

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    if intent_mode not in {"predicted", "no_intent", "shuffled"}:
        raise ValueError("intent_mode must be predicted, no_intent, or shuffled")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if intent_mode == "shuffled" and batch_size < 2:
        raise ValueError("shuffled intent evaluation requires batch_size >= 2")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for evaluation but is unavailable")
    model.to(resolved_device)
    model.eval()

    eligible_samples = len(dataset)
    attempted_samples = eligible_samples if max_samples == 0 else min(
        eligible_samples, max_samples
    )
    source_coverage = dict(getattr(dataset, "coverage", {}) or {})
    source_candidates = int(source_coverage.get("candidates", eligible_samples))
    preprocessing_excluded = int(source_coverage.get("excluded", 0))
    excluded_by_reason = dict(source_coverage.get("excluded_by_reason", {}) or {})

    totals: Counter[str] = Counter()
    failures: list[Dict[str, Any]] = []
    latencies: list[float] = []
    evaluated = 0
    with torch.inference_mode():
        for start in range(0, attempted_samples, batch_size):
            observation_samples: list[PlanningSample] = []
            target_samples: list[torch.Tensor] = []
            batch_indices: list[int] = []
            sample_ids: list[str] = []
            for index in range(start, min(start + batch_size, attempted_samples)):
                sample_id = f"index:{index}"
                try:
                    sample = dataset[index]
                    sample_id = str(sample.sample_id)
                    observation_sample, target = _observation_sample(sample)
                    observation_samples.append(observation_sample)
                    target_samples.append(target)
                    batch_indices.append(index)
                    sample_ids.append(sample_id)
                except Exception as error:
                    failures.append(
                        {
                            "index": index,
                            "sample_id": sample_id,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
            if not observation_samples:
                continue
            if intent_mode == "shuffled" and len(observation_samples) < 2:
                error = ValueError(
                    "shuffled intent requires at least two valid samples in every batch"
                )
                for index, sample_id in zip(batch_indices, sample_ids):
                    failures.append(
                        {
                            "index": index,
                            "sample_id": sample_id,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                continue
            try:
                observations = planning_collate_fn(observation_samples)
                if not isinstance(observations, ObservationBatch) or hasattr(
                    observations, "future"
                ):
                    raise TypeError("metric inference must receive observations only")
                observations = observations.to(resolved_device)
                target = torch.stack(target_samples).to(resolved_device)

                _synchronize(resolved_device)
                started = time.perf_counter()
                prediction = model.predict(
                    observations,
                    num_steps=num_inference_steps,
                    seed=_sample_seed(seed, "|".join(sample_ids)),
                    intent_mode=intent_mode,
                )
                _synchronize(resolved_device)
                elapsed = time.perf_counter() - started
                latencies.extend([elapsed / len(sample_ids)] * len(sample_ids))
                if not torch.is_tensor(prediction):
                    raise TypeError("PlanningModel.predict must return a torch.Tensor")
                values = trajectory_error_sums(prediction.float(), target.float())
                totals.update(values)
                evaluated += len(sample_ids)
            except Exception as error:  # coverage must retain individual failures
                for index, sample_id in zip(batch_indices, sample_ids):
                    failures.append(
                        {
                            "index": index,
                            "sample_id": sample_id,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )

    position_count = totals["position_count"]
    trajectory_count = totals["trajectory_count"]
    heading_count = totals["heading_count"]
    prediction_coverage = evaluated / attempted_samples if attempted_samples else 0.0
    limited = max_samples > 0 and max_samples < eligible_samples
    end_to_end_coverage = (
        evaluated / source_candidates if source_candidates and not limited else None
    )
    mean_latency = statistics.fmean(latencies) if latencies else None
    return {
        "seed": seed,
        "intent_mode": intent_mode,
        "batch_size": batch_size,
        "num_inference_steps": num_inference_steps,
        "metrics": {
            "ADE_m": totals["position_sum_m"] / position_count if position_count else None,
            "FDE_m": (
                totals["final_position_sum_m"] / trajectory_count
                if trajectory_count
                else None
            ),
            "heading_error_rad": (
                totals["heading_sum_rad"] / heading_count if heading_count else None
            ),
            "heading_error_deg": (
                math.degrees(totals["heading_sum_rad"] / heading_count)
                if heading_count
                else None
            ),
        },
        "coverage": {
            "source_candidates": source_candidates,
            "preprocessing_eligible": eligible_samples,
            "preprocessing_excluded": preprocessing_excluded,
            "excluded_by_reason": excluded_by_reason,
            "attempted": attempted_samples,
            "evaluated": evaluated,
            "inference_failures": len(failures),
            "source_coverage": source_coverage.get("coverage"),
            "prediction_coverage": prediction_coverage,
            "end_to_end_coverage": end_to_end_coverage,
            "limited_by_max_samples": limited,
        },
        "failures": failures,
        "latency_s": {
            "count": len(latencies),
            "mean": mean_latency,
            "median": statistics.median(latencies) if latencies else None,
            "p95": _percentile(latencies, 0.95),
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "total": sum(latencies),
        },
    }


def summarize_seeds(reports: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize successful metrics across repeated pure-noise seeds."""

    summary: Dict[str, Any] = {"num_seeds": len(reports), "metrics": {}}
    for name in ("ADE_m", "FDE_m", "heading_error_rad", "heading_error_deg"):
        values = [
            float(report["metrics"][name])
            for report in reports
            if report.get("metrics", {}).get(name) is not None
        ]
        summary["metrics"][name] = {
            "mean": statistics.fmean(values) if values else None,
            "std_population": (
                statistics.pstdev(values)
                if len(values) > 1
                else 0.0 if values else None
            ),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return summary


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must contain non-negative integers")
    return seeds


def _write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def _require_data_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(
            f"evaluation data root is unavailable or not a directory: {root}"
        )
    return root


def _metrics_command(args: argparse.Namespace) -> Dict[str, Any]:
    data_root = _require_data_root(args.data_root)
    model, config, checkpoint_state = load_evaluation_model(
        args.checkpoint, device=args.device
    )
    evaluation_data: DataConfig = copy.deepcopy(config.data)
    evaluation_data.val_root = str(data_root)
    evaluation_data.train_clip_fraction = 1.0
    dataset = PlanningDataset(evaluation_data, split="val", observation_only=False)
    reports = [
        evaluate_dataset(
            model,
            dataset,
            device=args.device,
            num_inference_steps=args.num_inference_steps,
            seed=seed,
            max_samples=args.max_samples,
            intent_mode=args.intent_mode,
            batch_size=args.batch_size,
        )
        for seed in args.seeds
    ]
    return {
        "mode": "metrics",
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_stage": checkpoint_state.stage.value,
        "checkpoint_global_step": checkpoint_state.global_step,
        "data_root": str(Path(args.data_root)),
        "intent_mode": args.intent_mode,
        "runs": reports,
        "seed_summary": summarize_seeds(reports),
    }


def _benchmark_command(args: argparse.Namespace) -> Dict[str, Any]:
    # Kept lazy: environments running tensor-only tests need no evaluator GIS
    # dependencies or benchmark annotations.
    from nureasoning.planning.benchmark import BenchmarkConfig, run_benchmark

    data_root = _require_data_root(args.data_root)
    provider = JEPAPlanningTrajectoryProvider(
        args.checkpoint,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        intent_mode=args.intent_mode,
    )
    benchmark_config = BenchmarkConfig(key_frame_index=args.key_frame_index)
    summary = run_benchmark(
        str(data_root),
        benchmark_config,
        max_clips=args.max_clips,
        trajectory_provider=provider,
    )
    failure_details = [
        {
            "clip_path": result.clip_path,
            "clip_name": result.clip_name,
            "error": result.error,
        }
        for result in summary.results
        if result.error is not None
    ]
    return {
        "mode": "benchmark",
        "checkpoint": str(Path(args.checkpoint)),
        "data_root": str(Path(args.data_root)),
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "intent_mode": args.intent_mode,
        "coverage": {
            "total_clips": summary.total_clips,
            "evaluated_clips": summary.evaluated_clips,
            "failed_clips": summary.failed_clips,
            "coverage": (
                summary.evaluated_clips / summary.total_clips
                if summary.total_clips
                else 0.0
            ),
        },
        "failures": failure_details,
        "metrics": {
            "collision": summary.mean_collision,
            "driveable": summary.mean_driveable,
            "progress": summary.mean_progress,
            "comfort": summary.mean_comfort,
            "human": summary.mean_human,
            "ADE_m": summary.mean_ade_m,
            "FDE_m": summary.mean_fde_m,
            "planning_score": summary.mean_planning_score,
        },
        "latency_s": {
            "count": len(provider.latencies_s),
            "mean": (
                statistics.fmean(provider.latencies_s)
                if provider.latencies_s
                else None
            ),
            "median": (
                statistics.median(provider.latencies_s)
                if provider.latencies_s
                else None
            ),
            "p95": _percentile(provider.latencies_s, 0.95),
            "min": min(provider.latencies_s) if provider.latencies_s else None,
            "max": max(provider.latencies_s) if provider.latencies_s else None,
            "total": sum(provider.latencies_s),
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate JEPA planning checkpoints")
    subparsers = parser.add_subparsers(dest="command", required=True)

    metrics = subparsers.add_parser(
        "metrics", help="ADE/FDE/wrapped-heading evaluation on held-out futures"
    )
    metrics.add_argument(
        "--checkpoint", required=True, help="Completed joint .pt checkpoint"
    )
    metrics.add_argument("--data-root", required=True, help="Held-out planning clip root")
    metrics.add_argument("--device", default="cuda")
    metrics.add_argument("--num-inference-steps", type=int, default=None)
    metrics.add_argument("--seeds", type=_parse_seeds, default=[42])
    metrics.add_argument("--max-samples", type=int, default=0, help="0 evaluates all")
    metrics.add_argument(
        "--intent-mode",
        choices=("predicted", "no_intent", "shuffled"),
        default="predicted",
        help="Validation ablation; shuffled requires --batch-size >= 2",
    )
    metrics.add_argument("--batch-size", type=int, default=1)
    metrics.add_argument("--output", required=True, help="Output JSON report")

    benchmark = subparsers.add_parser(
        "benchmark", help="Run the existing local map/agent benchmark callbacks"
    )
    benchmark.add_argument(
        "--checkpoint", required=True, help="Completed joint .pt checkpoint"
    )
    benchmark.add_argument("--data-root", required=True, help="Benchmark clip root")
    benchmark.add_argument("--device", default="cuda")
    benchmark.add_argument("--num-inference-steps", type=int, default=None)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--key-frame-index", type=int, default=100)
    benchmark.add_argument("--max-clips", type=int, default=0, help="0 evaluates all")
    benchmark.add_argument(
        "--intent-mode", choices=("predicted", "no_intent"), default="predicted"
    )
    benchmark.add_argument("--output", required=True, help="Output JSON report")
    return parser


def main(argv: Sequence[str] | None = None) -> Dict[str, Any]:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.num_inference_steps is not None and args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be positive")
    if getattr(args, "max_samples", 0) < 0 or getattr(args, "max_clips", 0) < 0:
        parser.error("sample/clip limits must be non-negative")
    if getattr(args, "seed", 0) < 0:
        parser.error("--seed must be non-negative")
    if getattr(args, "batch_size", 1) < 1:
        parser.error("--batch-size must be positive")
    if getattr(args, "intent_mode", "predicted") == "shuffled" and args.batch_size < 2:
        parser.error("shuffled intent requires --batch-size >= 2")
    report = (
        _metrics_command(args) if args.command == "metrics" else _benchmark_command(args)
    )
    _write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()


__all__ = [
    "build_argument_parser",
    "evaluate_dataset",
    "main",
    "summarize_seeds",
    "trajectory_error_sums",
]
