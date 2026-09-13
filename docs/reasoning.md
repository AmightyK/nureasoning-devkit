# Reasoning benchmark: fine-tuning and evaluation

`nureasoning.reasoning` is the reasoning part of the benchmark. It trains a vision-language model to answer the Spatial / Decision / Counterfactual questions generated from the reasoning annotations, and scores its answers. No planning action expert is involved; for the joint vision-language-action model see [nuVLA](nuvla.md).

The same code produced the fine-tuned rows of the paper's reasoning tables, so a run started from these entry points is directly comparable to the published numbers. For a runnable walkthrough see [`tutorials/nureasoning_reasoning_tutorial.ipynb`](../tutorials/nureasoning_reasoning_tutorial.ipynb).

## Pipeline

```
              reasoning annotations
                      │  vqa.generate
                      ▼
                 *_vqa.json  ──────────────────────┐
   training           │                            │  evaluation
                      │  reasoning.build_sft       │  reasoning.evaluate
                      ▼                            ▼
                sft_train.jsonl               manifest.jsonl
                      │  reasoning.train           │  answers from an
                      ▼                            │  OpenAI-compatible server
                 LoRA adapter                      ▼
                      │  reasoning.merge_lora  predictions.jsonl
                      ▼                            │  scoring
              merged checkpoint ──► served ────────┤
                                                   ▼
                                             metrics.yaml
```

| Step | Command |
| --- | --- |
| Build SFT data | `python -m nureasoning.reasoning.build_sft` |
| LoRA fine-tune | `python -m nureasoning.reasoning.train` |
| Merge adapter | `python -m nureasoning.reasoning.merge_lora` |
| Build manifest only | `python -m nureasoning.reasoning.build_manifest` |
| Evaluate | `python -m nureasoning.reasoning.evaluate` |
| Rescore predictions | `python -m nureasoning.reasoning.reaggregate` |

## What the model sees

Every example is **multi-view multi-frame**: **8 cameras × 2 frames = 16 images**, the current frame and the frame 1 second earlier, for each camera. There is no `--image-mode`. `build_sft`, `evaluate`, `build_manifest`, and challenge `--reasoning-provider api` all read `camera_sequences` the same way.

Image layout, captions, and question sampling **must be identical in `build_sft` and `evaluate`** — a model trained on one order or prompt and scored on another loses several points for no good reason. The published-style runs use:

```
--num-forward-frames 2 --max-images 16
--history-first --keyframe-only --max-spatial-per-frame 10 --qa-seed 42
```

`--num-forward-frames 2` keeps the two most recent timestamps per camera. `--history-first` emits them **time-major**: all 8 cameras at t−1 s, then all 8 at t=0. Without it the order is **camera-major** (each camera's own history in a run). Time-major is what mixed-resolution training needs, since the current frames are then the last images in the list.

Each image is preceded by the same identifier nuVLA training uses (`load_vlm_observation` / `VLATrainer`):

```
t=-1, back camera.
[image]
…
t=0 (current), front camera.
[image]
```

Those strings are stored as `image_contexts` next to `images` (SFT JSONL) or `image_paths` (eval manifest). A JSONL built before this field exists is still trained unlabeled; rebuild SFT if you want the captions.

Two flags control how much data each clip contributes:

- `--keyframe-only` keeps one frame per clip instead of every annotated frame, which stops a single clip from flooding the set with near-duplicate views. Spatial questions exist on every reasoning frame (~1 Hz) but Decision / Counterfactual questions only on key frames (every 5 s), so the frame nearest the middle of the clip that carries Decision / Counterfactual questions is used — on released clips that is the t ≈ 10 s key frame. A clip with no such frame falls back to the nearest frame with any questions.
- `--max-spatial-per-frame 10` caps the Spatial category, which is by far the most numerous, and keeps Decision and Counterfactual questions in full. The subset is chosen by an RNG seeded on `(--qa-seed, clip, frame)`, so every model and every re-run scores the exact same questions. `42` is the published seed.

### Training prompt

The user turn is built by `nureasoning.reasoning.build_sft.build_prompt`, then interleaved with the labeled images in `reasoning.train`. It is **not** `format_question_prompt` (that wrapper — “You are evaluating a driving scene…” and `A. ` choices — is nuVLA VQA mix and challenge `--reasoning-provider vlm`).

After the 16 caption+image pairs, the text is:

1. The question from the VQA file
2. Choices as `A) …` / `B) …` (sorted letters), if present
3. An answer-format suffix from `build_instruction_suffix` (the same module evaluation uses)

The assistant target is only the string the scorer parses (`B`, or `A,C` for multi-select). Prompt and image tokens are masked out of the loss.

Typical multiple-choice example:

```
t=-1, back camera.
[image]
t=-1, back_left camera.
[image]
…
t=0 (current), right camera.
[image]
Which of the following best describes the car in the left camera?

Choices:
A) parked on the left
B) crossing from left to right
C) receding ahead
D) not visible

Answer format: respond with ONLY the single capital letter (A, B, C, or D) that matches the correct choice. No other text.
```

Assistant: `B`

Other suffixes: multi-select letters (`A or A, C`); a single number; `[x, y]` in meters; `[[t, x, y], …]` for trajectories.

Challenge `--reasoning-provider api` (vLLM / OpenAI-compatible) uses this same labeled layout and `build_prompt` text. Serve a checkpoint trained **with** `image_contexts`; an older unlabeled merge plus captions at inference is a train/serve mismatch.

## 1. Generate questions

Training and evaluation both start from VQA files. Generate them once per split:

```bash
python -m nureasoning.vqa.generate \
  --data-root ./dataset/data/train --output ./vqa_output_train
python -m nureasoning.vqa.generate \
  --data-root ./dataset/data/validation --output ./vqa_output_val
```

## 2. Build the SFT dataset

```bash
python -m nureasoning.reasoning.build_sft \
  --vqa-dir ./vqa_output_train \
  --output ./reasoning_workspace/sft_train_multiframe.jsonl \
  --keyframe-only --num-forward-frames 2 --max-images 16 --history-first \
  --max-spatial-per-frame 10
```

Each line holds the ordered image paths, matching `image_contexts` captions, the user prompt, and the assistant target. The prompt is the question plus `A)` choices and an answer-format instruction (`respond with ONLY the single capital letter`, `respond with ONLY a JSON array [[t, x, y], ...]`, and so on), so the model is trained to answer in exactly the shape the scorer parses.

### `build_sft` flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--vqa-dir` | `./vqa_output` | Output of `nureasoning.vqa.generate` |
| `-o` / `--output` | required | Destination JSONL |
| `--workspace` | `.` | Root for workspace-relative image paths (generated VQA files use absolute paths) |
| `--max-images` | `16` | Cap on images per example (8 cameras × 2 frames) |
| `--num-forward-frames` | `2` | Most recent frames kept per camera |
| `--history-first` | off | Time-major order; required for mixed-resolution training |
| `--keyframe-only` | off | One frame per clip |
| `--max-spatial-per-frame` | unset | Cap Spatial questions per frame; Decision / Counterfactual kept in full |
| `--max-qa-per-frame` | unset | Blunter cap on *all* questions per frame, stratified by question type. Do not combine with `--max-spatial-per-frame`. **Not** exposed on `evaluate` — use `--max-spatial-per-frame` for comparable scores |
| `--qa-seed` | `42` | Seed for the per-frame question subset |
| `--limit-files` | unset | Debug: only the first N VQA files |
| `--limit-rows` | unset | Debug: stop after N training examples |

## 3. Fine-tune

Configs live in `nureasoning/reasoning/configs/`. Point `train_jsonl` at the JSONL you just built and `model_path` at the backbone, then:

```bash
torchrun --standalone --nproc_per_node=8 -m nureasoning.reasoning.train \
  --config nureasoning/reasoning/configs/qwen3.5-4b-multiframe.yaml
```

Only the LoRA adapter is trained, and the loss is computed on the assistant answer alone; prompt and image tokens are masked out. The backbone class comes from the checkpoint's own `architectures` field, so Qwen3.5, Qwen3-VL and other image-text-to-text models all work without code changes — set `model_class` in the config only if you need to override it.

Shipped configs (all 16-image, `max_pixels: 200704`, effective batch 64 on 8 GPUs):

| Config | Backbone | Per-GPU batch × accum | `save_steps` |
| --- | --- | --- | --- |
| `qwen3.5-4b-multiframe.yaml` | `Qwen/Qwen3.5-4B` | 4 × 2 | 200 |
| `qwen3.5-9b-multiframe.yaml` | `Qwen/Qwen3.5-9B` | 2 × 4 | 200 |
| `qwen3-vl-8b-multiframe.yaml` | `Qwen/Qwen3-VL-8B-Instruct` | 2 × 4 | 200 |

CLI flags are only `--config` (defaults to the 4B yaml) and `--resume-from-checkpoint <output_dir>/checkpoint-N`, which restores optimizer and scheduler state. Everything else is in the yaml.

Relative paths in the yaml are looked up against the working directory first and the config's directory second. `output_dir` is always relative to the working directory. `model_path` is left alone when it is a Hugging Face repo id.

### Model and data

| Key | 4B default | Meaning |
| --- | --- | --- |
| `model_path` | `Qwen/Qwen3.5-4B` | Local checkpoint directory or Hub repo id |
| `model_class` | unset | Force a `transformers` class. Leave unset to read `architectures` from the checkpoint |
| `train_jsonl` | `./reasoning_workspace/sft_train_multiframe.jsonl` | JSONL from `build_sft` |
| `max_samples` | `null` | Cap examples for a smoke run; `null` uses the whole file |
| `max_images` | `16` | Truncate each JSONL image list (and matching `image_contexts`) if it is longer. Image *order* and captions are already baked into the JSONL |
| `seed` | `42` | Hugging Face Trainer seed |

### Images and sequence length

| Key | 4B default | Meaning |
| --- | --- | --- |
| `max_pixels` | `200704` | Per-image pixel budget for the Qwen processor (`448 × 448`). Together with `max_images` this sets the vision-token count, which is what usually drives an out-of-memory error. Alias: `max_image_longest_edge` (squared) if `max_pixels` is unset |
| `max_seq_length` | `28672` | Must fit prompt plus vision tokens. Training aborts rather than silently truncating |
| `history_max_pixels` | unset | Downscale history frames to this budget. Needs time-major JSONL (`--history-first`) so that the last `num_current_images` entries are the current timestamp. Saves memory but makes train and eval inputs differ; published runs leave it off |
| `num_current_images` | `8` | How many trailing images stay at `max_pixels` when mixed resolution is on |

### LoRA

| Key | 4B default | Meaning |
| --- | --- | --- |
| `lora_r` | `32` | Rank |
| `lora_alpha` | `64` | Scaling (`alpha / rank`) |
| `lora_dropout` | `0.05` | Dropout on LoRA layers |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | Linear layers that receive adapters |

### Optimizer and schedule

Effective batch is `num_gpus × per_device_train_batch_size × gradient_accumulation_steps`. The reference runs use **64**. When you train on fewer GPUs, raise `gradient_accumulation_steps` to keep that product (for example 3 GPUs × batch 4 × accum 6 = 72, or drop batch to 2).

| Key | 4B default | Meaning |
| --- | --- | --- |
| `learning_rate` | `5e-5` | AdamW peak LR |
| `num_train_epochs` | `1` | Full passes over the JSONL |
| `max_steps` | `null` | If set, overrides the epoch-derived step count |
| `warmup_ratio` | `0.03` | Fraction of steps that linearly warm up, then cosine decay. Ignored when `warmup_steps` is set |
| `warmup_steps` | unset | Absolute warmup length; takes precedence over `warmup_ratio` |
| `weight_decay` | `0.01` | AdamW weight decay |
| `lr_scheduler_type` | `cosine` | Hugging Face scheduler name |
| `per_device_train_batch_size` | `4` | Micro-batch per GPU |
| `gradient_accumulation_steps` | `2` | Optimizer step every N micro-batches |
| `logging_steps` | `5` | Log loss every N optimizer steps (`logging_first_step` is always on) |
| `save_steps` | `200` | Write `checkpoint-N/` every N steps |
| `save_total_limit` | `3` | Keep only the last N checkpoints |

### Precision, memory, output

| Key | 4B default | Meaning |
| --- | --- | --- |
| `bf16` | `true` | BFloat16 (preferred) |
| `fp16` | `false` | Float16; ignored when `bf16` is on |
| `gradient_checkpointing` | `true` | Trades compute for activation memory |
| `output_dir` | `./reasoning_workspace/lora_qwen3.5-4b-multiframe` | Run directory stem |
| `output_append_run_date` | `true` | Rank 0 creates ``<output_dir>_YYYY-MM-DD_HHMMSS`` and broadcasts the path so every GPU writes to the same folder |

Set `TRAIN_ALLOW_CPU=1` only to debug the data path on a machine with no GPU; a real run needs the CUDA build from `environment.yml`.

## 4. Merge the adapter

Inference servers load a plain checkpoint far faster than base plus adapter:

```bash
python -m nureasoning.reasoning.merge_lora \
  --base Qwen/Qwen3.5-4B \
  --adapter ./reasoning_workspace/lora_qwen3.5-4b-multiframe_<run-stamp> \
  --out ./reasoning_workspace/merged_qwen3.5-4b-multiframe
```

Tied `lm_head` weights are dropped, and auxiliary heads that LoRA never touches (such as Qwen3.5's multi-token-prediction head) are copied from the base checkpoint, which some serving configurations require. Backbones without such a head are handled automatically.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--base` | required | Base checkpoint: local directory or Hub repo id (must match `model_path` used in training) |
| `--adapter` | required | Trained LoRA directory (the dated folder printed at the end of training) |
| `--out` | required | Destination for merged weights + processor |
| `--model-class` | unset | Override the `transformers` class (default: from the base config) |
| `--copy-key-prefix` | `mtp` | Copy tensors with this prefix from the base checkpoint. Empty string skips the copy |
| `--max-shard-size` | `5GB` | Safetensors shard size for the merged files |

## 5. Serve and evaluate

Two processes, two conda envs, two terminals:

| Terminal | Env | Role |
| --- | --- | --- |
| 1 | `nureasoning` | Train, merge, then `evaluate`. Never install vLLM here. |
| 2 | `vllm` | `vllm serve` the merged checkpoint. Leave it running; it owns the GPUs. |

`evaluate` does not load weights. It POSTs chat-completions to `http://127.0.0.1:8000/v1` (or `--api-urls`). Do not `pip install vllm` into `nureasoning` — it overwrites the training PyTorch stack. Env setup is in [installation](installation.md#reasoning-training--testing).

```bash
# Terminal 2 — vLLM env. Blocks until you stop it.
conda activate vllm
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve ./reasoning_workspace/merged_qwen3.5-4b-multiframe \
  --served-model-name nureasoning-4b-sft \
  --tensor-parallel-size 8 \
  --max-model-len 24576 \
  --dtype bfloat16 \
  --mm-processor-kwargs '{"max_pixels": 200704}' \
  --enable-prefix-caching
```

| Flag | Published | Meaning |
| --- | --- | --- |
| `--served-model-name` | `nureasoning-4b-sft` | Name `evaluate --model` must send |
| `--tensor-parallel-size` | `8` | GPUs on this server; set to the GPU count you actually have |
| `--max-model-len` | `24576` | Context window. Must cover 16 images plus the prompt; can be lower than training `max_seq_length` because generation does not include the assistant target |
| `--dtype` | `bfloat16` | Match training `bf16` |
| `--mm-processor-kwargs max_pixels` | `200704` | **Must** match training `max_pixels` |
| `--enable-prefix-caching` | on | Encode each frame's images once and reuse the KV cache for every question on that frame |

With 16 images per request, startup takes several minutes: multimodal warmup and KV-cache profiling dominate. Then, still in terminal 1 (`nureasoning`):

```bash
python -m nureasoning.reasoning.evaluate \
  --vqa-dir ./vqa_output_val \
  --model nureasoning-4b-sft \
  --output-root ./reasoning_eval \
  --keyframe-only --num-forward-frames 2 --max-images 16 --history-first \
  --max-spatial-per-frame 10
```

Questions that share a frame are sent together and their images are encoded once, so with prefix caching the vision tokens of a frame are prefilled once for all of its questions. Predictions stream to disk in manifest order, so an interrupted run still leaves a scoreable `predictions.jsonl`.

To spread the load over several single-GPU servers instead of one tensor-parallel server, list them all:

```bash
python -m nureasoning.reasoning.evaluate ... \
  --api-urls http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1
```

Any OpenAI-compatible endpoint works, so the same command scores an unmodified base model or a hosted API — that is how the zero-shot rows of the paper's tables were produced.

### Image layout (must match `build_sft`)

Eval rebuilds the same multi-view multi-frame list and `image_contexts` captions as training. There is no `--image-mode`.

| Flag | Default | Published | Meaning |
| --- | --- | --- | --- |
| `--vqa-dir` | `./vqa_output` | `./vqa_output_val` | VQA files for the split to score |
| `--workspace` | `.` | `.` | Root for workspace-relative image paths |
| `--max-images` | `16` | `16` | Images per question |
| `--num-forward-frames` | `2` | `2` | Frames per camera |
| `--history-first` | off | **on** | Time-major order |
| `--keyframe-only` | off | **on** | One frame per clip |
| `--max-spatial-per-frame` | unset | `10` | Spatial cap; Decision / Counterfactual in full |
| `--qa-seed` | `42` | `42` | Must match the seed used when building the SFT JSONL |

### Server and output

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `$VLLM_MODEL` or `nureasoning-4b-sft` | Name the server answers to (`--served-model-name`) |
| `--model-name` | sanitized `--model` | Directory name under `--output-root` |
| `--output-root` | `./reasoning_eval` | Results land in `<output-root>/<model-name>/` |
| `--base-url` | `$OPENAI_BASE_URL` or `http://127.0.0.1:8000/v1` | Single server |
| `--api-urls` | `$API_URLS` or unset | Comma-separated servers to shard frames across; overrides `--base-url` |
| `--api-key` | `$OPENAI_API_KEY` or `EMPTY` | Sent as the Bearer token; local vLLM ignores it |
| `--concurrent-per-url` | `8` | In-flight requests per server. Lower this if the server returns 500s or runs out of KV cache |
| `--max-retries` | `3` | Extra attempts on 429 / 5xx / timeouts, with exponential backoff |

### Run control and decoding

| Flag | Default | Meaning |
| --- | --- | --- |
| `--limit` | unset | Score only the first N questions (smoke test) |
| `--skip-manifest` | off | Reuse `reasoning_results/manifest.jsonl` already in the output directory (after `build_manifest`, or after an interrupted run) |
| `--thinking` | off | Leave the backbone's thinking mode on. Default is off (`enable_thinking: false`, `top_k: 20`), which is what the published numbers use |

Decoding is otherwise fixed in code: `temperature=0.2`, `top_p=0.9`, and `max_tokens` is 128 for most questions, 512 for free-text, and 2048 for trajectory answers. Changing those requires editing `nureasoning/reasoning/modules/inference.py`.

`python -m nureasoning.reasoning.build_manifest` takes the same image-layout flags as `evaluate` and writes the JSONL without calling the server. Write it to `<output-root>/<model-name>/reasoning_results/manifest.jsonl` and pass `--skip-manifest` so evaluate does not rebuild it.

## Outputs

```
<output-root>/<model-name>/
├── reasoning_results/
│   ├── manifest.jsonl                      one line per question, with images and captions
│   ├── predictions.jsonl                   raw answer and per-question metrics
│   ├── evaluation_vs_groundtruth.jsonl     answer next to ground truth, for inspection
│   └── evaluation_results.json             headline numbers and artifact paths
└── metrics_eval/
    └── metrics.yaml                        full stratified metrics
```

`evaluation_vs_groundtruth.jsonl` carries `clip_dir`, `category` and `subcategory` on every row, which is what you slice when you want a breakdown by scenario type — read `scenario_type` from each clip's `metadata.json` and group the rows by it.

## Metrics

| Question type | Reported | Also computed |
| --- | --- | --- |
| `choice` | letter exact match, or multi-select F1 when the answer is a list of letters | precision/recall, accuracy on normalized option text, macro-F1 |
| `numerical` (scalar) | accuracy within tolerance (per-question `tolerance` when present, otherwise `rtol=0.05`, `atol=0.5`) | absolute and relative error, MAE, RMSE |
| `numerical` (coordinate `[x, y]`) | hit rate within tolerance | L2 error |
| `numerical` (trajectory `[[t, x, y], ...]`) | hit rate within tolerance | mean L2 and RMSE over waypoints |
| `text` | ROUGE-L F1 when `rouge-score` is installed, otherwise token F1 | exact match, macro-F1 over free-form labels |
| `categorical` | label accuracy | normalized label fields |

Everything is aggregated per question type, per category, per subcategory, and per (question type, category) pair.

To recompute metrics after changing a definition, or to score a partial run, skip inference entirely:

```bash
python -m nureasoning.reasoning.reaggregate \
  ./reasoning_eval/nureasoning-4b-sft/reasoning_results/predictions.jsonl \
  -o ./reasoning_eval/nureasoning-4b-sft/metrics_eval/metrics_recomputed.json
```

A dependency-free check of the scoring functions:

```bash
python -m nureasoning.reasoning.modules.selftest_metrics
```

## Troubleshooting

**The manifest is empty.** The VQA directory is wrong, or its files carry image paths that do not resolve. `nureasoning.vqa.generate` writes absolute image paths; if you moved the dataset after generating, regenerate the VQA files or pass `--workspace` for workspace-relative paths.

**Training aborts on token length.** Lower `max_images` or `max_pixels`, or raise `max_seq_length`. Vision tokens, not the prompt, are almost always the cause.

**The server returns 500s under load.** Lower `--concurrent-per-url`. Multimodal requests with 16 images are large, and vLLM prefers a modest number of them in flight over a deep queue.

**Scores are far below the published ones.** Check that `--history-first`, `--num-forward-frames`, `--max-images`, `--max-spatial-per-frame` and `--qa-seed` match between the training data and the evaluation run, that eval and challenge API send the same `image_contexts` captions the JSONL was built with, and that `--mm-processor-kwargs max_pixels` on the server matches `max_pixels` in the training config. `--thinking` must stay off unless you intentionally want chain-of-thought. Do not serve an unlabeled (pre-caption) merge with captions at inference.

**`merge_lora` cannot find `adapter_config.json`.** Point `--adapter` at the directory that contains `adapter_model.safetensors`. Older multi-GPU runs could split files across a date-only folder and a `_HHMMSS` sibling; current training broadcasts one run directory to every rank.
