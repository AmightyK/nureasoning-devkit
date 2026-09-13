#!/usr/bin/env python3
"""Merge a trained LoRA adapter into the base weights for serving.

Inference servers such as vLLM load a plain checkpoint far more efficiently than
a base model plus adapter, so evaluation runs against a merged copy. The output
directory contains the merged weights, the config and the processor, and can be
passed straight to ``vllm serve``.

Example:
  python -m nureasoning.reasoning.merge_lora \
    --base Qwen/Qwen3.5-4B \
    --adapter ./reasoning_workspace/lora_4b_multiframe \
    --out ./reasoning_workspace/merged_4b_multiframe
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download, split_torch_state_dict_into_shards
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoProcessor

from nureasoning.reasoning.modules.model_factory import resolve_model_class

WEIGHTS_NAME = "model.safetensors"
# Yields model.safetensors for a single shard, model-00001-of-000NN.safetensors
# when split, which is the layout serving stacks expect.
WEIGHTS_PATTERN = "model{suffix}.safetensors"


def resolve_base_dir(base: str) -> Path | None:
    """Local directory holding the base checkpoint files.

    A Hugging Face repo id is mapped to its cache snapshot, which the load above
    has already populated. Returns None when neither can be found, in which case
    the auxiliary-head copy is skipped rather than silently dropping tensors.
    """
    local = Path(base).expanduser()
    if local.is_dir():
        return local.resolve()
    try:
        return Path(snapshot_download(base, allow_patterns=["*.safetensors", "*.json"]))
    except Exception:  # noqa: BLE001 - offline, private repo, or a bad id
        return None


def load_keys_with_prefix(base_dir: Path, prefix: str) -> dict[str, torch.Tensor]:
    """Read tensors whose name starts with *prefix* straight from the base checkpoint.

    LoRA never touches auxiliary heads such as Qwen3.5's multi-token-prediction
    head, and ``from_pretrained`` does not always materialise them, so they are
    copied over verbatim. Backbones without such a head yield an empty dict.
    """
    index_path = base_dir / "model.safetensors.index.json"
    out: dict[str, torch.Tensor] = {}

    if index_path.exists():
        weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
        wanted = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
        for shard in sorted(set(wanted.values())):
            with safe_open(str(base_dir / shard), framework="pt") as sf:
                for key, shard_name in wanted.items():
                    if shard_name == shard:
                        out[key] = sf.get_tensor(key)
        return out

    single = base_dir / WEIGHTS_NAME
    if single.exists():
        with safe_open(str(single), framework="pt") as sf:
            for key in sf.keys():
                if key.startswith(prefix):
                    out[key] = sf.get_tensor(key)
    return out


def save_sharded(state_dict: dict[str, torch.Tensor], out: Path, max_shard_size: str) -> int:
    """Write *state_dict* as safetensors, sharding when it exceeds *max_shard_size*."""
    split = split_torch_state_dict_into_shards(
        state_dict,
        filename_pattern=WEIGHTS_PATTERN,
        max_shard_size=max_shard_size,
    )
    for filename, keys in split.filename_to_tensors.items():
        shard = {k: state_dict[k] for k in keys}
        save_file(shard, str(out / filename), metadata={"format": "pt"})

    if split.is_sharded:
        (out / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": split.metadata, "weight_map": split.tensor_to_filename},
                       indent=2)
        )
    return len(split.filename_to_tensors)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True,
                    help="Base checkpoint: a local directory or a Hugging Face repo id")
    ap.add_argument("--adapter", type=Path, required=True, help="Trained LoRA directory")
    ap.add_argument("--out", type=Path, required=True, help="Destination for merged weights")
    ap.add_argument("--model-class", default=None,
                    help="Override the transformers class (default: from the base config)")
    ap.add_argument("--copy-key-prefix", default="mtp",
                    help="Copy tensors with this prefix from the base checkpoint "
                         "(auxiliary heads LoRA does not train)")
    ap.add_argument("--max-shard-size", default="5GB")
    args = ap.parse_args()

    local_base = Path(args.base).expanduser()
    base = str(local_base.resolve()) if local_base.is_dir() else args.base
    adapter = args.adapter.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    model_cls = resolve_model_class(base, args.model_class)
    print(f"Loading {model_cls.__name__} from {base}", flush=True)
    model = model_cls.from_pretrained(
        str(base),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()

    state_dict = model.state_dict()
    print(f"Merged state_dict: {len(state_dict)} tensors", flush=True)

    if getattr(model.config, "tie_word_embeddings", False):
        for key in [k for k in state_dict if k.startswith("lm_head")]:
            del state_dict[key]
            print(f"  dropped tied key: {key}", flush=True)

    if args.copy_key_prefix:
        base_dir = resolve_base_dir(base)
        if base_dir is None:
            print(f"  WARNING: could not locate the base checkpoint files for {base}; "
                  f"skipping the '{args.copy_key_prefix}*' copy", flush=True)
        else:
            extra = load_keys_with_prefix(base_dir, args.copy_key_prefix)
            if extra:
                state_dict.update(extra)
                print(f"  copied {len(extra)} '{args.copy_key_prefix}*' tensors from the base",
                      flush=True)

    cpu_state = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    n_shards = save_sharded(cpu_state, out, args.max_shard_size)
    model.config.save_pretrained(str(out))
    AutoProcessor.from_pretrained(str(adapter), trust_remote_code=True).save_pretrained(str(out))

    print(f"Merged model + processor -> {out} "
          f"({len(cpu_state)} tensors in {n_shards} file(s))", flush=True)


if __name__ == "__main__":
    main()
