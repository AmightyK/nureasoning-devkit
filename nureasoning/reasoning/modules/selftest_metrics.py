#!/usr/bin/env python3
"""Quick sanity checks for metrics (no API, no GPU).

Run: python -m nureasoning.reasoning.modules.selftest_metrics
"""
from __future__ import annotations

from nureasoning.reasoning.modules.metrics import (
    compute_dataset_metrics,
    compute_metrics_for_sample,
    score_multiselect,
    score_trajectory,
)


def main() -> None:
    m = compute_metrics_for_sample("choice", "C", {"answer": "C", "answer_text": "car", "choices": {"C": "car"}})
    assert m["choice_letter_exact"] == 1.0
    assert m.get("choice_gold_label_norm")

    m2 = score_multiselect("A, C", {"answer": ["A", "C"]})
    assert m2["multiselect_exact_match"] == 1.0

    gold_traj = [
        [0.5, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    pred = "[[0.5, 0.0, 0.0], [1.0, 1.1, 0.0]]"
    mt = score_trajectory(pred, {"answer": gold_traj, "tolerance": 0.2})
    assert mt.get("trajectory_mean_l2_xy") is not None

    rows = [
        {
            "question_type": "choice",
            "error": None,
            "metrics": {
                "choice_gold_label_norm": "a",
                "choice_pred_label_norm": "a",
            },
        },
        {
            "question_type": "choice",
            "error": None,
            "metrics": {
                "choice_gold_label_norm": "b",
                "choice_pred_label_norm": "a",
            },
        },
    ]
    ds = compute_dataset_metrics(rows)
    assert "choice_macro_f1_answer_text" in ds
    print("selftest_metrics: OK", ds)


if __name__ == "__main__":
    main()
