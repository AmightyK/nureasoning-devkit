"""One-model challenge predictor: nuVLA plans and answers questions.

A checkpoint trained with ``--reasoning_mode vqa`` (``--vqa_root``) uses the
same VLM adapter for multiple-choice answers and the same action expert for
the ego trajectory. Planning prompt defaults to ``auto`` (scene-only for VQA
checkpoints, structured otherwise); override with ``--planning-prompt``.
Challenge answers always use the VQA question prompt.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nureasoning.nuvla.trajectory_provider import VLATrajectoryProvider
from nureasoning.submission.clips import load_key_frame_ego_state
from nureasoning.submission.format import global_to_ego, normalize_ego_trajectory, trajectory_to_list
from nureasoning.submission.reasoning import VLMChallengeAnswerer, answer_row

logger = logging.getLogger(__name__)


class NuVLAClipPredictor:
    """Run planning and QA from a single nuVLA checkpoint."""

    def __init__(
        self,
        checkpoint_dir: str,
        num_inference_steps: int = 5,
        device: str = "cuda",
        max_new_tokens: int = 128,
        planning_prompt: str = "auto",
        planning_reasoning_format: Optional[str] = None,
    ):
        self.planner = VLATrajectoryProvider(
            checkpoint_dir=checkpoint_dir,
            num_inference_steps=num_inference_steps,
            device=device,
            planning_prompt=planning_prompt,
            reasoning_format=planning_reasoning_format,
        )
        reasoning_mode = str(self.planner.cfg.get("reasoning_mode") or "structured")
        logger.info(
            "nuVLA checkpoint reasoning_mode=%s (challenge QA uses VQA prompts)",
            reasoning_mode,
        )
        self.answerer = VLMChallengeAnswerer(
            vlm=self.planner.vlm,
            device=str(self.planner.device),
            max_new_tokens=max_new_tokens,
            torch_module=self.planner._torch,
            train_cfg=self.planner.cfg,
            cam_names=list(self.planner._cam_names),
        )

    def plan_ego_trajectory(
        self,
        clip_path: str,
        key_frame_idx: int,
        ego_state: Any,
    ) -> np.ndarray:
        trajectory = self.planner(clip_path, key_frame_idx, ego_state)
        if trajectory is None or len(trajectory) == 0:
            raise ValueError("nuVLA returned an empty trajectory")
        ego_traj = global_to_ego(
            np.asarray(trajectory, dtype=np.float64)[:, :3],
            ego_state,
        )
        return normalize_ego_trajectory(ego_traj)

    def answer_questions(
        self,
        questions: List[Dict[str, Any]],
        sample: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], int]:
        results: List[Dict[str, Any]] = []
        failed = 0
        for question in questions:
            qid = question.get("question_id")
            try:
                pred = self.answerer.answer(question, sample)
                results.append(answer_row(question, sample, pred))
            except Exception as exc:
                failed += 1
                logger.error("Failed on question %s (%s): %s", qid, sample.get("clip"), exc)
                results.append(answer_row(question, sample, {}, error=str(exc)))
        return results, failed

    def predict_clip(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        clip_path = sample["clip_path"]
        frames = sample.get("frames") or []
        key_frame_idx = int(sample.get("key_frame_index") or 0)
        ego_state = load_key_frame_ego_state(clip_path, frames[key_frame_idx])
        trajectory = self.plan_ego_trajectory(clip_path, key_frame_idx, ego_state)
        answers, failed = self.answer_questions(sample.get("questions") or [], sample)
        return {
            "trajectory": trajectory_to_list(trajectory),
            "answers": answers,
            "num_failed": failed,
        }
