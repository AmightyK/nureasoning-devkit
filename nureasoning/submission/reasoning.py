"""Reasoning answerers for challenge questions."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from nureasoning.reasoning.modules.prompt_format import format_question_prompt

logger = logging.getLogger(__name__)


class BaseChallengeAnswerer(ABC):
    """Interface for answering a single challenge question given observations."""

    @abstractmethod
    def answer(
        self,
        question: Dict[str, Any],
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return ``{"answer": ..., "answer_text": ...}``."""


class StubChallengeAnswerer(BaseChallengeAnswerer):
    """Lightweight heuristic / random answerer for pipeline dry-runs."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def answer(
        self,
        question: Dict[str, Any],
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        del sample
        qtype = str(question.get("question_type", "open")).lower()
        choices = question.get("choices") or {}

        if qtype == "choice" and choices:
            letter = self._rng.choice(sorted(choices.keys()))
            return {"answer": letter, "answer_text": choices[letter]}

        if qtype == "numerical":
            return {"answer": 0.0, "answer_text": "0.0"}

        return {"answer": "", "answer_text": ""}


def parse_model_response(raw: str, question: Dict[str, Any]) -> Dict[str, Any]:
    text = (raw or "").strip()
    first_line = text.splitlines()[0].strip() if text else ""
    qtype = str(question.get("question_type", "open")).lower()
    choices = question.get("choices") or {}

    if qtype == "choice" and choices:
        match = re.search(r"\b([A-Da-d])\b", first_line)
        if match:
            letter = match.group(1).upper()
            return {"answer": letter, "answer_text": choices.get(letter, first_line)}
        lowered = first_line.lower()
        for letter, choice_text in choices.items():
            if str(choice_text).lower() in lowered:
                return {"answer": letter, "answer_text": choice_text}
        letter = sorted(choices.keys())[0]
        return {"answer": letter, "answer_text": choices[letter]}

    if qtype == "numerical":
        match = re.search(r"[-+]?\d*\.?\d+", first_line.replace(",", ""))
        if match:
            value = float(match.group(0))
            if value.is_integer():
                value = int(value)
            return {"answer": value, "answer_text": first_line}
        return {"answer": first_line, "answer_text": first_line}

    return {"answer": first_line or text, "answer_text": first_line or text}


def answer_row(
    question: Dict[str, Any],
    sample: Dict[str, Any],
    pred: Dict[str, Any],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    row = {
        "question_id": question.get("question_id"),
        "answer": pred.get("answer"),
        "answer_text": pred.get("answer_text", ""),
        "question_type": question.get("question_type"),
        "category": question.get("category"),
        "subcategory": question.get("subcategory"),
    }
    if error:
        row["error"] = error
        row["answer"] = None
    del sample
    return row


class VLMChallengeAnswerer(BaseChallengeAnswerer):
    """Answer questions with a nuVLA VLM adapter or a Hugging Face VL model.

    Image layout, resolutions, and (for ``reasoning_mode=vqa``) the user prompt
    match ``nureasoning.nuvla.train``. Challenge questions always use
    ``format_question_prompt``, which is the VQA training target.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        vlm_model_path: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 128,
        vlm: Any = None,
        torch_module: Any = None,
        image_module: Any = None,
        train_cfg: Optional[Dict[str, Any]] = None,
        cam_names: Optional[List[str]] = None,
    ):
        import torch
        from PIL import Image
        from nureasoning.nuvla.models.vlm_backbone import CAMERA_NAMES

        self._torch = torch_module or torch
        self._Image = image_module or Image
        self._cam_names = list(cam_names or CAMERA_NAMES)
        self.device = self._torch.device(
            device if self._torch.cuda.is_available() else "cpu"
        )
        self.max_new_tokens = max_new_tokens
        self.cfg: Dict[str, Any] = dict(train_cfg or {})

        if checkpoint_dir and not self.cfg:
            from nureasoning.nuvla.trajectory_provider import load_vla_training_config

            self.cfg = load_vla_training_config(checkpoint_dir)

        if vlm is not None:
            self.vlm = vlm
            return

        from nureasoning.nuvla.models.vlm_backbone import VLMBackbone, VLMBackboneConfig
        from nureasoning.nuvla.trajectory_provider import vlm_backbone_config_kwargs

        if self.cfg:
            kwargs = vlm_backbone_config_kwargs(self.cfg)
            if vlm_model_path:
                kwargs["model_name_or_path"] = self.cfg.get(
                    "vlm_model_path", vlm_model_path
                )
        else:
            kwargs = {
                "model_name_or_path": vlm_model_path,
                "freeze_vision_encoder": True,
                "lora_rank": 0,
                "lora_alpha": 0,
                "lora_dropout": 0.0,
            }

        self.vlm = VLMBackbone(VLMBackboneConfig(**kwargs)).to(self.device)
        self.vlm.eval()

        if checkpoint_dir:
            adapter_dir = os.path.join(os.path.normpath(checkpoint_dir), "vlm_adapter")
            if os.path.isdir(adapter_dir):
                self.vlm.load_adapter(adapter_dir, device=self.device)
                logger.info("Loaded VLM adapter from %s", adapter_dir)
            else:
                logger.warning("No vlm_adapter found under %s", checkpoint_dir)

    def _load_images(self, sample: Dict[str, Any]) -> Tuple[List[Any], List[str]]:
        clip_path = sample.get("clip_path")
        key_idx = sample.get("key_frame_index")
        if clip_path and os.path.isfile(os.path.join(clip_path, "metadata.json")):
            from nureasoning.nuvla.trajectory_provider import load_vlm_observation

            try:
                frame_idx = int(key_idx if key_idx is not None else 0)
            except (TypeError, ValueError):
                frame_idx = 0
            images, contexts, _ = load_vlm_observation(
                clip_path,
                frame_idx,
                self.cfg,
                cam_names=self._cam_names,
            )
            if images:
                return images, contexts

        Image = self._Image
        current_res = (
            int((self.cfg or {}).get("current_res_w", 448)),
            int((self.cfg or {}).get("current_res_h", 448)),
        )
        image_paths = sample.get("image_paths") or {}
        images: List[Any] = []
        contexts: List[str] = []
        for cam in self._cam_names:
            path = image_paths.get(cam)
            if not path or not os.path.isfile(path):
                continue
            try:
                images.append(
                    Image.open(path).convert("RGB").resize(current_res, Image.LANCZOS)
                )
                contexts.append(f"t=0 (current), {cam} camera.")
            except OSError:
                continue
        return images, contexts

    def answer(
        self,
        question: Dict[str, Any],
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        torch = self._torch
        images, contexts = self._load_images(sample)
        if not images:
            res = (
                int((self.cfg or {}).get("current_res_w", 448)),
                int((self.cfg or {}).get("current_res_h", 448)),
            )
            images = [self._Image.new("RGB", res, (0, 0, 0))]
            contexts = ["t=0 (current), front camera."]

        prompt = format_question_prompt(question)
        inputs = self.vlm.prepare_inputs(
            images=images,
            text_prompt=prompt,
            image_contexts=contexts,
        )
        vlm_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        generate_kwargs: Dict[str, Any] = {
            "input_ids": vlm_inputs["input_ids"],
            "attention_mask": vlm_inputs["attention_mask"],
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }
        for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            if key in vlm_inputs:
                generate_kwargs[key] = vlm_inputs[key]

        with torch.no_grad():
            output_ids = self.vlm.model.generate(**generate_kwargs)
        prompt_len = vlm_inputs["input_ids"].shape[-1]
        generated = self.vlm.processor.batch_decode(
            output_ids[:, prompt_len:],
            skip_special_tokens=True,
        )[0]
        return parse_model_response(generated, question)


class APIChallengeAnswerer(BaseChallengeAnswerer):
    """
    Batch-answer questions via an OpenAI-compatible server (vLLM / SGLang).

    ``answer()`` is unused; call ``answer_samples()`` so images are encoded once
    per frame, matching ``nureasoning.reasoning.evaluate``.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_urls: str = "",
        api_key: str = "EMPTY",
        concurrent_per_url: int = 8,
        thinking: bool = False,
        max_images: int = 16,
        num_forward_frames: int = 2,
        history_first: bool = True,
    ):
        from nureasoning.reasoning.modules.gpu_utils import parse_api_urls

        self.model = model
        self.urls = parse_api_urls(api_urls) or [base_url.rstrip("/")]
        self.api_key = api_key
        self.concurrent_per_url = concurrent_per_url
        self.thinking = thinking
        self.max_images = max_images
        self.num_forward_frames = num_forward_frames
        self.history_first = history_first
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._clients: Optional[List[Any]] = None

    def answer(
        self,
        question: Dict[str, Any],
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise RuntimeError("APIChallengeAnswerer.answer_samples() must be used")

    def _image_payload(self, sample: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        from nureasoning.reasoning.modules.image_selection import (
            collect_image_slots,
            nuvla_image_context,
        )

        vqa_doc = {
            "temporal_multiview_context": sample.get("temporal_multiview_context") or {},
        }
        slots = collect_image_slots(
            vqa_doc,
            max_images=self.max_images,
            num_forward_frames=self.num_forward_frames,
            history_first=self.history_first,
        )
        paths: List[str] = []
        contexts: List[str] = []
        for slot in slots:
            path = slot.get("path")
            if path and os.path.isfile(path):
                paths.append(path)
                contexts.append(
                    nuvla_image_context(
                        str(slot.get("camera") or ""),
                        slot.get("relative_index", 0),
                    )
                )
        if paths:
            return paths, contexts

        image_paths = sample.get("image_paths") or {}
        if isinstance(image_paths, dict):
            for cam, path in sorted(image_paths.items()):
                if path and os.path.isfile(path):
                    paths.append(path)
                    contexts.append(nuvla_image_context(cam, 0))
        return paths, contexts

    def answer_samples(
        self,
        samples: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        by_clip: Dict[str, List[Dict[str, Any]]] = {}
        for sample in samples:
            by_clip[sample["clip"]] = self._answer_one_clip(sample)
        return by_clip

    def _event_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._clients = None
        return self._loop

    def _openai_clients(self) -> List[Any]:
        from openai import AsyncOpenAI

        if not self._clients:
            self._clients = [
                AsyncOpenAI(base_url=u.rstrip("/"), api_key=self.api_key)
                for u in self.urls
            ]
        return self._clients

    def _answer_one_clip(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        from nureasoning.reasoning.modules.inference import clear_b64_cache, run_inference

        image_paths, image_contexts = self._image_payload(sample)
        requests: List[Dict[str, Any]] = []
        questions = list(sample.get("questions") or [])
        for question in questions:
            requests.append({
                "sample_id": question.get("question_id"),
                "question_id": question.get("question_id"),
                "question_type": question.get("question_type"),
                "category": question.get("category"),
                "subcategory": question.get("subcategory"),
                "question": question.get("question"),
                "choices": question.get("choices"),
                "image_paths": image_paths,
                "image_contexts": image_contexts,
                "gold": {"answer": None},
            })
        if not requests:
            return []

        try:
            predictions = self._event_loop().run_until_complete(
                run_inference(
                    requests,
                    self.urls,
                    self.model,
                    self.concurrent_per_url,
                    self.api_key,
                    self.thinking,
                    clients=self._openai_clients(),
                )
            )
        finally:
            clear_b64_cache()

        rows: List[Dict[str, Any]] = []
        clip = sample["clip"]
        for pred, question in zip(predictions, questions):
            parsed = parse_model_response(pred.get("prediction_raw") or "", question)
            row = answer_row(question, {"clip": clip}, parsed)
            if pred.get("error"):
                row["error"] = pred["error"]
                row["answer"] = None
            rows.append(row)
        return rows


def build_reasoning_answerer(args: argparse.Namespace) -> BaseChallengeAnswerer:
    if args.reasoning_provider == "stub":
        return StubChallengeAnswerer(seed=args.seed)
    if args.reasoning_provider == "vlm":
        return VLMChallengeAnswerer(
            checkpoint_dir=args.checkpoint_dir,
            vlm_model_path=args.vlm_model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
    if args.reasoning_provider == "api":
        return APIChallengeAnswerer(
            model=args.api_model,
            base_url=args.base_url,
            api_urls=args.api_urls,
            api_key=args.api_key,
            concurrent_per_url=args.concurrent_per_url,
            thinking=args.thinking,
            max_images=args.max_images,
            num_forward_frames=args.num_forward_frames,
            history_first=args.history_first,
        )
    raise SystemExit(
        f"Unknown reasoning provider '{args.reasoning_provider}'. "
        "Choose from: stub, vlm, api"
    )
