#!/usr/bin/env python3
"""LoRA supervised fine-tuning of a VLM on nuReasoning question-answer data.

Reads the JSONL written by ``python -m nureasoning.reasoning.build_sft`` and
trains a LoRA adapter on top of a frozen image-text-to-text backbone. Only the
assistant answer is supervised; the prompt tokens are masked out of the loss.

The backbone class is taken from the checkpoint's own config, so the same
command trains Qwen3.5, Qwen3-VL or any other ``transformers`` VLM.

Example:
  torchrun --nproc_per_node=8 -m nureasoning.reasoning.train \
    --config nureasoning/reasoning/configs/qwen3.5-4b-multiframe.yaml
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoProcessor, Trainer, TrainingArguments

from nureasoning.common.pretrained import from_pretrained, resolve_pretrained_path
from nureasoning.reasoning.modules.model_factory import resolve_model_class

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "qwen3.5-4b-multiframe.yaml"


def require_cuda_or_exit() -> None:
    if torch.cuda.is_available():
        return
    if os.environ.get("TRAIN_ALLOW_CPU", "").strip().lower() in ("1", "true", "yes"):
        print("WARNING: CUDA unavailable — training on CPU (very slow).", flush=True)
        return
    raise SystemExit(
        "ERROR: torch.cuda.is_available() is False.\n"
        "Install a CUDA build of PyTorch (see environment.yml), or set "
        "TRAIN_ALLOW_CPU=1 to run on CPU anyway.\n"
    )


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(raw: str, config_dir: Path) -> Path:
    """Absolute paths as given; relative ones against the CWD, then the config dir."""
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    alt = (config_dir / p).resolve()
    return alt if alt.exists() else p.resolve()


def resolve_model_ref(raw: str, config_dir: Path) -> str:
    """A local checkpoint directory, or a Hugging Face repo id left untouched."""
    p = Path(raw).expanduser()
    if p.is_absolute() or p.exists() or (config_dir / p).exists():
        return str(resolve_path(raw, config_dir))
    return raw


def _ddp_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _ddp_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _ensure_process_group() -> None:
    """Init DDP if torchrun launched us and nobody has called init yet.

    ``resolve_output_dir`` has to broadcast before Hugging Face Trainer exists,
    so it cannot wait for TrainingArguments to set the process group up.
    """
    if _ddp_world_size() <= 1:
        return
    import torch.distributed as dist
    if dist.is_initialized():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def _claim_run_dir(base: Path) -> Path:
    """Create ``<name>_YYYY-MM-DD_HHMMSS``, adding microseconds on collision."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    candidate = base.parent / f"{base.name}_{stamp}"
    while True:
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
            candidate = base.parent / f"{base.name}_{stamp}"


def resolve_output_dir(cfg: dict[str, Any]) -> Path:
    """Pick a run directory on rank 0 and broadcast it.

    Under torchrun every rank used to call this independently. The date-only
    path was created by one rank, then another rank saw it and appended
    ``_HHMMSS``, so adapter weights and processor files landed in two folders.
    """
    base = Path(str(cfg.get("output_dir") or "./nureasoning_reasoning_workspace")).expanduser()
    if not cfg.get("output_append_run_date", True):
        return base

    _ensure_process_group()
    if _ddp_rank() == 0:
        chosen = str(_claim_run_dir(base))
    else:
        chosen = ""

    if _ddp_world_size() > 1:
        import torch.distributed as dist
        payload = [chosen]
        dist.broadcast_object_list(payload, src=0)
        chosen = payload[0]
    return Path(chosen)


def resize_to_pixel_budget(im: Image.Image, max_px: int) -> Image.Image:
    """Downscale *im* to at most *max_px* pixels, keeping the aspect ratio.

    Both sides are rounded to a multiple of 28, the patch size the Qwen image
    processors expect.
    """
    w, h = im.size
    if w * h <= max_px:
        return im
    ratio = (max_px / (w * h)) ** 0.5
    return im.resize(
        (max(28, round(w * ratio / 28) * 28), max(28, round(h * ratio / 28) * 28)),
        Image.LANCZOS,
    )


def open_images(
    paths: list[str],
    *,
    history_max_pixels: int | None = None,
    num_current_images: int = 0,
) -> list[Image.Image]:
    """Load images, optionally shrinking the history frames.

    With a time-major image order the last *num_current_images* entries are the
    current timestamp; everything before that is history and can be fed at lower
    resolution to save vision tokens.
    """
    out: list[Image.Image] = []
    n_history = len(paths) - num_current_images if history_max_pixels else 0
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        if i < n_history:
            im = resize_to_pixel_budget(im, history_max_pixels)
        out.append(im)
    return out


@dataclass
class Row:
    images: list[str]
    question: str
    assistant: str
    image_contexts: list[str] | None = None


class ReasoningSFTDataset(TorchDataset):
    def __init__(self, rows: list[Row]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"idx": idx, "row": self.rows[idx]}


def process_sample(
    processor: Any,
    row: Row,
    idx: int,
    prompt_len_cache: dict[int, int],
    max_seq_length: int,
    *,
    history_max_pixels: int | None = None,
    num_current_images: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    images = open_images(
        row.images,
        history_max_pixels=history_max_pixels,
        num_current_images=num_current_images,
    )
    user_content: list[dict[str, Any]] = []
    contexts = row.image_contexts
    if contexts is not None and len(contexts) == len(images):
        for im, ctx in zip(images, contexts):
            user_content.append({"type": "text", "text": ctx})
            user_content.append({"type": "image", "image": im})
    else:
        user_content.extend({"type": "image", "image": im} for im in images)
    user_content.append({"type": "text", "text": row.question})

    full = processor.apply_chat_template(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": row.assistant}]},
        ],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = full["input_ids"]
    attention_mask = full.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    if idx not in prompt_len_cache:
        prompt_only = processor.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_len_cache[idx] = int(prompt_only["input_ids"].shape[1])
    prompt_len = prompt_len_cache[idx]

    token_len = int(input_ids.shape[1])
    if token_len > max_seq_length:
        raise RuntimeError(
            f"Token length {token_len} exceeds max_seq_length {max_seq_length}. "
            "Lower max_images or max_pixels, or raise max_seq_length."
        )

    labels = input_ids.clone()
    if prompt_len > 0:
        labels[:, :prompt_len] = -100

    seq_len = input_ids.shape[1]
    text_extra: dict[str, torch.Tensor] = {}
    vision: dict[str, torch.Tensor] = {}
    for k, v in full.items():
        if k in ("input_ids", "attention_mask") or not isinstance(v, torch.Tensor):
            continue
        if v.dim() == 2 and v.shape == (1, seq_len):
            text_extra[k] = v.squeeze(0)
        else:
            vision[k] = v

    return input_ids.squeeze(0), attention_mask.squeeze(0), labels.squeeze(0), text_extra, vision


def collate_fn_factory(
    processor: Any,
    max_seq_length: int,
    *,
    history_max_pixels: int | None = None,
    num_current_images: int = 0,
):
    prompt_len_cache: dict[int, int] = {}
    pad_id = processor.tokenizer.pad_token_id or 0

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        all_ids: list[torch.Tensor] = []
        all_mask: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        text_extra_lists: dict[str, list[torch.Tensor]] = {}
        vision_lists: dict[str, list[torch.Tensor]] = {}

        for item in batch:
            ids, mask, labels, text_extra, vision = process_sample(
                processor,
                item["row"],
                int(item["idx"]),
                prompt_len_cache,
                max_seq_length,
                history_max_pixels=history_max_pixels,
                num_current_images=num_current_images,
            )
            all_ids.append(ids)
            all_mask.append(mask)
            all_labels.append(labels)
            for k, v in text_extra.items():
                text_extra_lists.setdefault(k, []).append(v)
            for k, v in vision.items():
                vision_lists.setdefault(k, []).append(v)

        max_len = max(ids.shape[0] for ids in all_ids)
        padded_ids: list[torch.Tensor] = []
        padded_mask: list[torch.Tensor] = []
        padded_labels: list[torch.Tensor] = []
        padded_text_extra: dict[str, list[torch.Tensor]] = {k: [] for k in text_extra_lists}

        for i, (ids, mask, labels) in enumerate(zip(all_ids, all_mask, all_labels)):
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                padded_ids.append(torch.cat([ids, ids.new_full((pad_len,), pad_id)]))
                padded_mask.append(torch.cat([mask, mask.new_zeros(pad_len)]))
                padded_labels.append(torch.cat([labels, labels.new_full((pad_len,), -100)]))
                for k, lst in text_extra_lists.items():
                    padded_text_extra[k].append(torch.cat([lst[i], lst[i].new_zeros(pad_len)]))
            else:
                padded_ids.append(ids)
                padded_mask.append(mask)
                padded_labels.append(labels)
                for k, lst in text_extra_lists.items():
                    padded_text_extra[k].append(lst[i])

        out: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(padded_mask),
            "labels": torch.stack(padded_labels),
        }
        for k, tensors in padded_text_extra.items():
            out[k] = torch.stack(tensors)
        for k, tensors in vision_lists.items():
            out[k] = torch.cat(tensors, dim=0)
        return out

    return collate


class MultimodalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


def load_rows(train_jsonl: Path, max_images: int | None, max_samples: int | None) -> list[Row]:
    rows: list[Row] = []
    with train_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            images = list(obj.get("images") or [])
            contexts = obj.get("image_contexts")
            if max_images is not None and len(images) > max_images:
                images = images[:max_images]
                if isinstance(contexts, list):
                    contexts = contexts[:max_images]
            if not isinstance(contexts, list) or len(contexts) != len(images):
                contexts = None
            assistant = obj.get("assistant") or ""
            if not images or not assistant:
                continue
            rows.append(Row(
                images=images,
                question=obj.get("question") or "",
                assistant=assistant,
                image_contexts=list(contexts) if contexts is not None else None,
            ))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--resume-from-checkpoint", type=str, default=None,
                    help="Trainer checkpoint directory to resume from, e.g. "
                         "<output_dir>/checkpoint-1600")
    args = ap.parse_args()

    if not args.config.is_file():
        raise SystemExit(f"Config not found: {args.config}")
    cfg = load_config(args.config)
    config_dir = args.config.resolve().parent

    model_path = resolve_pretrained_path(str(resolve_model_ref(str(cfg["model_path"]), config_dir)))
    train_jsonl = resolve_path(str(cfg["train_jsonl"]), config_dir)
    if not train_jsonl.is_file():
        raise SystemExit(
            f"Training data not found: {train_jsonl}\n"
            "Build it first with: python -m nureasoning.reasoning.build_sft"
        )

    out_dir = resolve_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir (resolved): {out_dir}", flush=True)

    max_images = cfg.get("max_images")
    rows = load_rows(
        train_jsonl,
        int(max_images) if max_images is not None else None,
        int(cfg["max_samples"]) if cfg.get("max_samples") is not None else None,
    )
    if not rows:
        raise SystemExit(f"No training rows loaded from {train_jsonl}")
    print(f"Loaded {len(rows)} training examples from {train_jsonl}", flush=True)

    require_cuda_or_exit()

    processor = from_pretrained(AutoProcessor.from_pretrained, str(model_path), trust_remote_code=True)
    if getattr(processor.tokenizer, "pad_token_id", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    max_pixels = cfg.get("max_pixels")
    if max_pixels is None and cfg.get("max_image_longest_edge") is not None:
        max_pixels = int(cfg["max_image_longest_edge"]) ** 2
    if max_pixels is not None:
        processor.image_processor.size["longest_edge"] = int(max_pixels)
        print(f"Image processor: max_pixels={int(max_pixels):,} "
              f"(~{int(int(max_pixels) ** 0.5)}px square)", flush=True)

    history_max_pixels = cfg.get("history_max_pixels")
    if history_max_pixels is not None:
        history_max_pixels = int(history_max_pixels)
    num_current_images = int(cfg.get("num_current_images", 8))
    if history_max_pixels:
        print(f"Mixed resolution: {num_current_images} current images at "
              f"max_pixels={max_pixels or 'default'}, history at "
              f"{history_max_pixels} (~{int(history_max_pixels ** 0.5)}px)", flush=True)

    use_bf16 = bool(cfg.get("bf16", True))
    use_fp16 = bool(cfg.get("fp16", False)) and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    model_cls = resolve_model_class(model_path, cfg.get("model_class"))
    print(f"Backbone class: {model_cls.__name__}", flush=True)
    model = from_pretrained(
        model_cls.from_pretrained,
        str(model_path),
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    )

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model = get_peft_model(
        model,
        LoraConfig(
            r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.0)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(cfg.get("target_modules") or ["q_proj", "v_proj"]),
        ),
    )
    model.print_trainable_parameters()

    collate = collate_fn_factory(
        processor,
        int(cfg.get("max_seq_length", 8192)),
        history_max_pixels=history_max_pixels,
        num_current_images=num_current_images,
    )

    per_device_bs = int(cfg.get("per_device_train_batch_size", 1))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 8))
    n_epochs = float(cfg.get("num_train_epochs", 1))
    n_gpus = max(1, int(os.environ.get("WORLD_SIZE", torch.cuda.device_count())))
    max_steps = cfg.get("max_steps")

    if max_steps:
        total_steps = int(max_steps)
    else:
        total_steps = int((len(rows) // (per_device_bs * n_gpus * grad_accum)) * n_epochs)

    if cfg.get("warmup_steps") is not None:
        warmup_steps = int(cfg["warmup_steps"])
    else:
        warmup_steps = max(1, round(total_steps * float(cfg.get("warmup_ratio", 0.03))))

    print(f"Schedule: {total_steps} steps, {warmup_steps} warmup, {n_gpus} GPU(s), "
          f"effective batch = {per_device_bs * n_gpus * grad_accum}", flush=True)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        num_train_epochs=n_epochs,
        max_steps=int(max_steps) if max_steps else -1,
        warmup_steps=warmup_steps,
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "cosine")),
        logging_steps=int(cfg.get("logging_steps", 5)),
        logging_first_step=True,
        save_steps=int(cfg.get("save_steps", 200)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        seed=int(cfg.get("seed", 42)),
    )

    trainer = MultimodalTrainer(
        model=model,
        args=targs,
        train_dataset=ReasoningSFTDataset(rows),
        data_collator=collate,
    )
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}", flush=True)
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_model(str(out_dir))
    if _ddp_rank() == 0:
        processor.save_pretrained(str(out_dir))
        with (out_dir / "train_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump({**cfg, "output_dir_resolved": str(out_dir)}, f,
                           allow_unicode=True, default_flow_style=False)
        print(f"Saved LoRA adapter and processor to {out_dir}")
        print("Next: python -m nureasoning.reasoning.merge_lora "
              f"--base {model_path} --adapter {out_dir} --out <merged_dir>")


if __name__ == "__main__":
    main()
