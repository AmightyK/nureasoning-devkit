"""
VLM Backbone: Qwen3-VL / Qwen3.5 wrapper for nuReasoning VLA training.

Provides:
  - Multi-view, multi-frame image encoding with camera/timestep tokens
  - Text reasoning head (CoT + driving decision)
  - PEFT/LoRA integration with save/load helpers
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from nureasoning.common.pretrained import from_pretrained
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)


logger = logging.getLogger(__name__)

CAMERA_NAMES = [
    "front", "front_left", "front_right",
    "left", "right",
    "back", "back_left", "back_right",
]

CAMERA_TOKENS = {cam: f"<camera_{cam}>" for cam in CAMERA_NAMES}
TIMESTEP_TOKEN_FN = lambda t: f"<t={t}>"


@dataclass
class VLMBackboneConfig:
    model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    freeze_vision_encoder: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    lora_dropout: float = 0.05
    current_resolution: Tuple[int, int] = (448, 448)
    history_resolution: Tuple[int, int] = (448, 448)
    reasoning_format: str = "spatial_driving_counterfactual"
    trust_remote_code: bool = True
    attn_implementation: str = "flash_attention_2"
    torch_dtype: torch.dtype = torch.bfloat16


class VLMBackbone(nn.Module):
    """
    Wraps Qwen3-VL or Qwen3.5 for multi-view multi-frame reasoning.

    Forward produces:
      - reasoning_logits: next-token logits on the text reasoning head
      - vlm_features: pooled hidden states from the specified layer
        for conditioning the action expert.
    """

    def __init__(self, config: VLMBackboneConfig):
        super().__init__()
        self.config = config
        self.model_is_vl = self._is_vl_backbone(config.model_name_or_path)

        self.model = self._load_backbone(config)
        self.feature_dim = self._infer_feature_dim()

        processor_kwargs: Dict[str, Any] = {
            "trust_remote_code": config.trust_remote_code,
        }
        if self.model_is_vl:
            processor_kwargs.update({
                "min_pixels": 64 * 28 * 28,
                "max_pixels": 256 * 28 * 28,
            })
        self.processor = from_pretrained(
            AutoProcessor.from_pretrained,
            config.model_name_or_path,
            **processor_kwargs,
        )

        if config.freeze_vision_encoder:
            self._freeze_vision_encoder()

        if config.lora_rank > 0:
            self._apply_lora(config)

    def _infer_feature_dim(self) -> int:
        """
        Infer the language hidden size from the loaded backbone.

        For Qwen3-VL the relevant size usually lives under
        `model.config.text_config.hidden_size`, while text-only models often use
        `model.config.hidden_size`.
        """
        cfg = getattr(self.model, "config", None)
        text_cfg = getattr(cfg, "text_config", None)

        for source in (text_cfg, cfg):
            hidden_size = getattr(source, "hidden_size", None) if source is not None else None
            if isinstance(hidden_size, int) and hidden_size > 0:
                self.config.feature_dim = hidden_size
                return hidden_size

        logger.warning(
            "Could not infer VLM hidden size; falling back to configured feature_dim=%s",
            self.config.feature_dim,
        )
        return self.config.feature_dim

    # ------------------------------------------------------------------
    # Backbone loading helpers
    # ------------------------------------------------------------------

    def _from_pretrained_with_fallback(
        self,
        model_cls: Any,
        model_name_or_path: str,
        kwargs: Dict[str, Any],
    ) -> nn.Module:
        try:
            return from_pretrained(model_cls.from_pretrained, model_name_or_path, **kwargs)
        except Exception:
            reduced_kwargs = dict(kwargs)
            for key in ("attn_implementation", "dtype", "device_map"):
                reduced_kwargs.pop(key, None)
            return from_pretrained(model_cls.from_pretrained, model_name_or_path, **reduced_kwargs)

    def _is_vl_backbone(self, model_name_or_path: str) -> bool:
        try:
            cfg = from_pretrained(
                AutoConfig.from_pretrained,
                model_name_or_path,
                trust_remote_code=self.config.trust_remote_code,
            )
            model_type = str(getattr(cfg, "model_type", "")).lower()
            if "vl" in model_type or "vision" in model_type:
                return True
        except Exception:
            pass
        lowered = model_name_or_path.lower()
        return ("-vl" in lowered) or ("vision" in lowered)

    def _load_backbone(self, config: VLMBackboneConfig) -> nn.Module:
        lowered = config.model_name_or_path.lower()
        is_qwen35 = any(t in lowered for t in ("qwen3.5", "qwen3_5", "qwen35"))

        common_kwargs: Dict[str, Any] = {
            "torch_dtype": config.torch_dtype,
            "attn_implementation": config.attn_implementation,
            "trust_remote_code": config.trust_remote_code,
        }

        if self.model_is_vl:
            is_qwen3_vl = ("qwen3-vl" in lowered) or ("qwen3_vl" in lowered)
            if is_qwen3_vl:
                if Qwen3VLForConditionalGeneration is None:
                    raise ImportError(
                        "Qwen3-VL requires Qwen3VLForConditionalGeneration from transformers."
                    )
                qwen3_vl_kwargs: Dict[str, Any] = {
                    "torch_dtype": config.torch_dtype,
                    "attn_implementation": config.attn_implementation,
                    "trust_remote_code": config.trust_remote_code,
                }
                return self._from_pretrained_with_fallback(
                    Qwen3VLForConditionalGeneration,
                    config.model_name_or_path,
                    qwen3_vl_kwargs,
                )
            return self._from_pretrained_with_fallback(
                AutoModelForCausalLM,
                config.model_name_or_path,
                common_kwargs,
            )

        if is_qwen35:
            if Qwen3_5ForConditionalGeneration is None:
                raise ImportError(
                    "Qwen3.5 requires Qwen3_5ForConditionalGeneration from transformers."
                )
            qwen35_kwargs: Dict[str, Any] = {
                "torch_dtype": config.torch_dtype,
                "attn_implementation": config.attn_implementation,
                "trust_remote_code": config.trust_remote_code,
            }
            return self._from_pretrained_with_fallback(
                Qwen3_5ForConditionalGeneration,
                config.model_name_or_path,
                qwen35_kwargs,
            )

        return self._from_pretrained_with_fallback(
            AutoModelForCausalLM,
            config.model_name_or_path,
            common_kwargs,
        )

    # ------------------------------------------------------------------
    # Freezing & LoRA via PEFT
    # ------------------------------------------------------------------

    def _freeze_vision_encoder(self):
        vision_modules = ["visual", "vision_tower", "vision_model"]
        for name in vision_modules:
            module = getattr(self.model, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False
                logger.info("Froze vision encoder module: %s", name)
                return
        logger.info("No vision encoder module found to freeze.")

    def _apply_lora(self, config: VLMBackboneConfig):
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as exc:
            raise ModuleNotFoundError(
                "LoRA is enabled (lora_rank > 0), but `peft` is not installed. "
                "Install it with `pip install peft`."
            ) from exc

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # PEFT adapter save / load
    # ------------------------------------------------------------------

    def save_adapter(self, save_dir: str) -> None:
        """Save only the PEFT adapter weights (tiny checkpoint)."""
        os.makedirs(save_dir, exist_ok=True)
        if self._is_peft_model():
            self.model.save_pretrained(save_dir)
            logger.info("PEFT adapter saved to %s", save_dir)
        else:
            torch.save(
                {k: v.cpu() for k, v in self.state_dict().items()},
                os.path.join(save_dir, "vlm_backbone.pt"),
            )
            logger.info("Full VLM state saved to %s", save_dir)

    def load_adapter(self, load_dir: str, device: Optional[torch.device] = None) -> None:
        """Load PEFT adapter weights directly into the existing 'default' adapter.

        We avoid `PeftModel.load_adapter` because its behaviour for re-loading an
        already-registered adapter name has varied across PEFT versions. Using
        `set_peft_model_state_dict` guarantees the in-place adapter weights are
        overwritten, and we then audit weight norms to confirm the load worked.
        """
        if self._is_peft_model():
            sft_path = os.path.join(load_dir, "adapter_model.safetensors")
            bin_path = os.path.join(load_dir, "adapter_model.bin")

            if os.path.isfile(sft_path):
                from safetensors.torch import load_file
                state = load_file(sft_path)
            elif os.path.isfile(bin_path):
                state = torch.load(bin_path, map_location="cpu")
            else:
                logger.warning("No adapter checkpoint found in %s", load_dir)
                return

            from peft import set_peft_model_state_dict
            load_result = set_peft_model_state_dict(
                self.model, state, adapter_name="default",
            )
            missing = getattr(load_result, "missing_keys", []) or []
            unexpected = getattr(load_result, "unexpected_keys", []) or []

            if hasattr(self.model, "set_adapter"):
                self.model.set_adapter("default")

            lora_params = [
                (n, p) for n, p in self.model.named_parameters() if "lora_" in n
            ]
            num_lora = len(lora_params)
            if num_lora == 0:
                logger.error(
                    "No LoRA parameters found on the model after load_adapter; "
                    "the LoRA wrappers are missing."
                )
                return

            total_abs = sum(p.detach().float().abs().sum().item() for _, p in lora_params)
            mean_abs = total_abs / max(1, sum(p.numel() for _, p in lora_params))
            sample_name, sample_param = lora_params[0]

            logger.info(
                "PEFT adapter loaded from %s (%d tensors in file -> %d LoRA params on model; "
                "missing=%d, unexpected=%d; active=%s; mean|w|=%.4e)",
                load_dir,
                len(state),
                num_lora,
                len(missing),
                len(unexpected),
                getattr(self.model, "active_adapters", "?"),
                mean_abs,
            )
            logger.info(
                "  audit sample: %s  mean|w|=%.4e  max|w|=%.4e",
                sample_name,
                sample_param.detach().float().abs().mean().item(),
                sample_param.detach().float().abs().max().item(),
            )

            if mean_abs < 1e-10:
                logger.warning(
                    "LoRA weights after load have essentially zero magnitude; "
                    "the adapter likely did NOT load correctly."
                )
            if missing:
                logger.warning("set_peft_model_state_dict missing keys: %d (first: %s)",
                               len(missing), missing[:3])
            if unexpected:
                logger.warning("set_peft_model_state_dict unexpected keys: %d (first: %s)",
                               len(unexpected), unexpected[:3])
        else:
            pt_path = os.path.join(load_dir, "vlm_backbone.pt")
            if os.path.isfile(pt_path):
                state = torch.load(pt_path, map_location=device or "cpu")
                self.load_state_dict(state)
                logger.info("Full VLM state loaded from %s", pt_path)

    def _is_peft_model(self) -> bool:
        try:
            from peft import PeftModel
            return isinstance(self.model, PeftModel)
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_multiview_prompt(
        self,
        num_history_steps: int,
        num_current_cameras: int = 8,
        mission_command: str = "LANE_FOLLOW",
        include_reasoning_prefix: bool = True,
        reasoning_format: Optional[str] = None,
    ) -> str:
        parts: List[str] = []
        analysis = (
            "Analyze the scene and provide the structured reasoning requested below."
            if include_reasoning_prefix
            else "Analyze the scene to support the driving mission."
        )
        parts.append(
            "You are an autonomous driving assistant. "
            "You are given multi-view camera images from a self-driving vehicle "
            "at multiple timesteps. Each image is paired with a text label that "
            f"identifies its timestep and camera view. {analysis}\n\n"
        )
        parts.append(
            f"You are given {num_history_steps} history timesteps plus the current "
            f"timestep across {num_current_cameras} cameras.\n"
        )
        parts.append(f"Mission command: {mission_command}\n\n")

        if include_reasoning_prefix:
            fmt = str(
                reasoning_format or self.config.reasoning_format
            ).lower().strip()
            if fmt == "driving":
                parts.append(
                    "Based on the multi-view multi-frame observations, provide:\n"
                    "1. [Driving] scene description, critical components, decision, and trace\n\n"
                )
            elif fmt == "spatial_driving":
                parts.append(
                    "Based on the multi-view multi-frame observations, provide:\n"
                    "1. [Spatial] scene/layout, object relations, and map context\n"
                    "2. [Driving] scene description, critical components, decision, and trace\n\n"
                )
            elif fmt == "spatial":
                parts.append(
                    "Based on the multi-view multi-frame observations, provide:\n"
                    "1. [Spatial] scene/layout, object relations, and map context\n\n"
                )
            elif fmt == "driving_counterfactual":
                parts.append(
                    "Based on the multi-view multi-frame observations, provide:\n"
                    "1. [Driving] scene description, critical components, decision, and trace\n"
                    "2. [Counterfactual] alternative and unsafe actions with risk rationale\n\n"
                )
            else:
                parts.append(
                    "Based on the multi-view multi-frame observations, provide:\n"
                    "1. [Spatial] scene/layout, object relations, and map context\n"
                    "2. [Driving] scene description, critical components, decision, and trace\n"
                    "3. [Counterfactual] alternative and unsafe actions with risk rationale\n\n"
                )

        return "".join(parts)

    # ------------------------------------------------------------------
    # Processor / input preparation
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        images: List[Any],
        text_prompt: str,
        image_contexts: Optional[List[str]] = None,
        assistant_response: Optional[str] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build model inputs from images + prompt, optionally with a ground-truth
        assistant response for supervised fine-tuning.

        Two modes:

        1. ``assistant_response is None`` (inference / generation):
           Produces a user-turn conversation with ``add_generation_prompt=True`` so
           the model is primed to produce the assistant reply. Returns the usual
           processor outputs plus ``prompt_length`` (= full sequence length, since
           everything is prefix).

        2. ``assistant_response`` provided (training / teacher-forced eval):
           Produces a two-turn user + assistant conversation with
           ``add_generation_prompt=False``. The response sits in the assistant
           turn so the template delimiters match what ``.generate`` will see at
           inference. Returns additionally:
             - ``labels``: input_ids with every prefix token masked to -100 so
               the cross-entropy is only computed on the assistant response.
             - ``prompt_length``: exact number of tokens in the user prefix
               (including ``<|im_start|>assistant\\n``), so the action expert
               can condition on the prompt-side features.
        """
        if not images:
            raise ValueError("prepare_inputs requires at least one image.")

        content: List[Dict[str, Any]] = []
        if image_contexts is not None and len(image_contexts) != len(images):
            raise ValueError(
                f"image_contexts length ({len(image_contexts)}) must match "
                f"images length ({len(images)})"
            )
        if image_contexts is None:
            content.extend([{"type": "image", "image": img} for img in images])
        else:
            for img, context in zip(images, image_contexts):
                content.append({"type": "text", "text": context})
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text_prompt})

        user_msg = {"role": "user", "content": content}

        prefix_text = self.processor.apply_chat_template(
            [user_msg], tokenize=False, add_generation_prompt=True,
        )

        if assistant_response is None:
            full_text = prefix_text
        else:
            assistant_msg = {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_response}],
            }
            full_text = self.processor.apply_chat_template(
                [user_msg, assistant_msg],
                tokenize=False,
                add_generation_prompt=False,
            )

        inputs = self.processor(
            text=[full_text],
            images=images,
            padding=True,
            return_tensors="pt",
        )

        tok = self.processor.tokenizer
        prefix_tok = tok(prefix_text, add_special_tokens=False)["input_ids"]
        full_tok = tok(full_text, add_special_tokens=False)["input_ids"]
        assistant_token_count = max(0, len(full_tok) - len(prefix_tok))
        total_len = int(inputs["input_ids"].shape[-1])
        prompt_length = total_len - assistant_token_count

        inputs["prompt_length"] = torch.tensor([prompt_length], dtype=torch.long)

        if assistant_response is not None:
            built_labels = inputs["input_ids"].clone()
            built_labels[:, :prompt_length] = -100
            # Also mask any right-side padding (if padding=True added any).
            if "attention_mask" in inputs:
                built_labels[inputs["attention_mask"] == 0] = -100 
            inputs["labels"] = built_labels
        elif labels is not None:
            inputs["labels"] = labels

        return inputs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        prompt_lengths: Optional[torch.Tensor] = None,
        return_features: bool = True,
    ) -> Dict[str, torch.Tensor]:
        model_inputs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "output_hidden_states": return_features,
        }
        if self.model_is_vl and pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                model_inputs["image_grid_thw"] = image_grid_thw
            if mm_token_type_ids is not None:
                model_inputs["mm_token_type_ids"] = mm_token_type_ids

        outputs = self.model(**model_inputs)

        result: Dict[str, torch.Tensor] = {"logits": outputs.logits}
        if outputs.loss is not None:
            result["loss"] = outputs.loss

        if return_features and getattr(outputs, "hidden_states", None):
            hidden = outputs.hidden_states[-1]  # [B, seq_len, D] last layer
            if prompt_lengths is not None:
                max_prompt_len = prompt_lengths.max().item()
                B, _, D = hidden.shape
                reduced = hidden.new_zeros(B, max_prompt_len, D)
                for i in range(B):
                    pl = prompt_lengths[i].item()
                    reduced[i, :pl] = hidden[i, :pl]
                result['vlm_features'] = reduced
            else:
                result['vlm_features'] = hidden

        return result

    def get_feature_dim(self) -> int:
        return self.feature_dim
