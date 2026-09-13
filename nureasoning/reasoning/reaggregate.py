#!/usr/bin/env python3
"""Recompute metrics from an existing predictions.jsonl, without calling a model.

Useful after changing a metric definition, or to score a partial run that was
interrupted (predictions are streamed, so the file is always valid JSONL).

Example:
  python -m nureasoning.reasoning.reaggregate \
    ./reasoning_eval/nureasoning-4b-sft/reasoning_results/predictions.jsonl \
    -o ./reasoning_eval/nureasoning-4b-sft/metrics_eval/metrics_recomputed.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nureasoning.reasoning.modules.inference import aggregate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("predictions", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("metrics_from_predictions.json"))
    args = p.parse_args()

    rows = []
    with args.predictions.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(aggregate(rows), f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output} ({len(rows)} predictions)")


if __name__ == "__main__":
    main()
