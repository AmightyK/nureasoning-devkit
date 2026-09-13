# nuVLA: training, dataloader, validation, and planning benchmark

nuVLA is the reference vision-language-action (VLA) model of this devkit. Code lives under `nureasoning.nuvla`.

## Architecture

```
┌──────────────────┐      ┌────────────────────┐      ┌─────────────────────┐
│  Multi-view      │      │  VLM Backbone      │      │  Action Expert      │
│  Multi-frame     ├─────>│  (Qwen3-VL)        ├─────>│  (Flow-Match DiT)   │
│  Images          │      │                    │      │                     │
│                  │      │  Reasoning Loss    │      │  Action Loss        │
│                  │      │  (text generation) │      │  (trajectory MSE)   │
└──────────────────┘      └────────────────────┘      └─────────────────────┘
```

- **VLM backbone** (`nureasoning.nuvla.models.vlm_backbone`): Qwen3-VL, optionally LoRA-adapted. The language head is supervised with a reasoning text target (`--reasoning_mode`): structured Spatial / Driving / Counterfactual annotations, or generated VQA multiple-choice Q&A. Hidden states from the backbone condition the action expert. Vision attention defaults to `flash_attention_2` and falls back if that kernel is unavailable.
- **Action expert** (`nureasoning.nuvla.models.action_expert`): flow-matching DiT. It denoises future ego waypoints `(x, y, θ)` in the ego frame, conditioned on VLM hidden states, ego velocity/acceleration, and a short history trajectory. The action expert trains on trajectories, regardless of reasoning mode.

Joint loss (per step, before optimizer).

```
total = reasoning_loss + action_loss_weight * action_loss
```

## Dataloader

`nureasoning.nuvla.models.data_loader` provides `NuReasoningVLADataset`, `VLADataConfig`, `vla_collate_fn`, and `build_dataloader`. A clip is used only if it has a reasoning JSON and enough prior frames for the requested history. Each sample packs:

- current-frame **8 cameras** at `--current_res_*` plus `--num_history_steps` earlier frames (index stride `--history_stride` at 10 Hz, 10 ⇒ 1 s) at `--history_res_*`, flattened time-major then camera-major with labels such as `t=-1, front camera.` / `t=0 (current), front camera.`,
- ego dynamics `(vx, vy, ax, ay)` and up to 6 history waypoints from `ego_state`,
- a VLM user prompt and assistant target:
  - **`structured`**: `build_multiview_prompt` plus Spatial / Driving / Counterfactual text (`--reasoning_format`),
  - **`vqa`**: `format_question_prompt` plus the scorer-format answer (letter, number, `[x, y]`, trajectory JSON, or text) when a matching file exists under `--vqa_root`; otherwise the structured text,
- the future ego trajectory (`--num_trajectory_points` waypoints, default **10 points over 5 s**). Official clips are 10 Hz; the dataloader takes every 0.5 s pose (every 5th frame) as the action-expert target.

`--data_root` / `--test_data_root` may be a split folder (`dataset/data/train`) or a single `part_*` folder; clip discovery is recursive.

## Training

Hub repo ids are cached under `./models/.hf` unless `HF_HOME` is set. A local snapshot is used if `--vlm_model_path` is a directory (or `./models/<name>` exists):

```bash
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir ./models/Qwen3-VL-2B-Instruct

# Structured reasoning (default when --vqa_root is omitted)
python -m nureasoning.nuvla.train \
  --data_root ./dataset/data/train \
  --test_data_root ./dataset/data/validation \
  --output_dir ./nureasoning_vla_workspace \
  --vlm_model_path ./models/Qwen3-VL-2B-Instruct \
  --reasoning_mode structured

# Multi-GPU (DDP). --batch_size is per GPU.
torchrun --nproc_per_node=8 -m nureasoning.nuvla.train \
  --data_root ./dataset/data/train \
  --test_data_root ./dataset/data/validation \
  --output_dir ./nureasoning_vla_workspace \
  --vlm_model_path ./models/Qwen3-VL-2B-Instruct \
  --reasoning_mode structured
```

With the defaults, global batch is `batch_size × GPUs` (2 on one GPU, 16 on 8 GPUs) and the optimizer steps every `--gradient_accumulation_steps` (4) mini-batches, so the **effective batch is 8 × GPUs**.

Checkpoints are written under `--output_dir`:

```
<output_dir>/
  training_config.json      # CLI args (parent of epoch_* / final)
  train.log
  epoch_<k>/
    vlm_adapter/
    action_expert.pt
    training_state.pt       # optimizers / schedulers / step
  final/                    # same layout after the last epoch
```

Resume with `--resume_from <epoch_dir>`. Eval and challenge inference load `training_config.json` from the workspace root (parent of the checkpoint) first, then from the checkpoint directory itself.

### Paths and schedule

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data_root` | `./dataset/data/train` | Training clips |
| `--test_data_root` | `./dataset/data/validation` | Held-out clips for epoch eval (omit / empty skips eval) |
| `--vlm_model_path` | `Qwen/Qwen3-VL-2B-Instruct` | Hub id or local snapshot |
| `--output_dir` | `./nureasoning_vla_workspace_spatial_driving_counterfactual` | Workspace / checkpoints |
| `--batch_size` | `2` | Per-GPU micro-batch |
| `--gradient_accumulation_steps` | `4` | Optimizer step every N micro-batches |
| `--epochs` | `10` | Full passes over the train set |
| `--num_workers` | `8` | DataLoader workers |
| `--log_interval` | `10` | Log train metrics every N micro-batches |
| `--eval_interval` | `1` | Run trajectory eval every N epochs |
| `--save_interval` | `1` | Save a checkpoint every N epochs |
| `--resume_from` | unset | Path to an `epoch_*` or `final` directory |

### Optimizers and loss

| Flag | Default | Meaning |
| --- | --- | --- |
| `--vlm_lr` | `5e-5` | AdamW LR for trainable VLM / LoRA params |
| `--action_lr` | `1e-4` | AdamW LR for the DiT action expert |
| `--weight_decay` | `0.01` | Applied to both optimizers |
| `--max_grad_norm` | `1.0` | Clip each tower separately; `0` disables |
| `--warmup_ratio` | `0.05` | Fraction of optimizer steps that linearly warm up, then cosine decay to `1e-6` |
| `--action_loss_weight` | `1.0` | Multiplier on flow-matching loss vs reasoning CE |

### VLM / LoRA / reasoning text

`--reasoning_mode` selects **one** VLM text target. The modes are exclusive: `--reasoning_mode structured` rejects `--vqa_root`; `--reasoning_mode vqa` requires a VQA directory. If `--reasoning_mode` is omitted, `vqa` is selected when `--vqa_root` is set, otherwise `structured`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--lora_rank` | `32` | LoRA rank. **`0` trains the full VLM** (no adapters) |
| `--lora_alpha` | `64` | LoRA scaling (`alpha / rank`) |
| `--lora_dropout` | `0.1` | Dropout on LoRA layers |
| `--freeze_vision_encoder` | off | Freeze the vision tower; language (and LoRA) still train |
| `--reasoning_mode` | unset (`structured` unless `--vqa_root`) | `structured` or `vqa` |
| `--reasoning_format` | `spatial_driving_counterfactual` | Annotation sections for **structured** mode (ignored as the VLM target in **vqa** mode) |
| `--reasoning_max_items_per_list` | `10` | Cap on listed objects / alternative actions in structured text |
| `--vqa_root` | unset | Output of `nureasoning.vqa.generate`. Required for `--reasoning_mode vqa` |
| `--qa_seed` | `42` | RNG seed for which VQA question is sampled per clip |

LoRA (when rank > 0) targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

`--reasoning_format` controls both the structured user prompt and which JSON fields are serialized as the assistant target:

| Value | Spatial | Driving | Counterfactual |
| --- | --- | --- | --- |
| `spatial` | yes |  |  |
| `driving` |  | yes |  |
| `spatial_driving` | yes | yes |  |
| `driving_counterfactual` |  | yes | yes |
| `spatial_driving_counterfactual` | yes | yes | yes |

Driving covers scene description, critical components, longitudinal/lateral decision, and the reasoning trace.

### Train with QA (one checkpoint for planning and reasoning questions)

Generate VQA from the training annotations, then train with `--reasoning_mode vqa` so the VLM text target is generated questions of every type (choice, numerical, categorical, text). Each sample draws one question, with types sampled uniformly so spatial multiple-choice does not dominate. The action expert still trains on trajectories:

```bash
python -m nureasoning.vqa.generate \
  --data-root ./dataset/data/train --output ./vqa_output_train

python -m nureasoning.nuvla.train \
  --data_root ./dataset/data/train \
  --test_data_root ./dataset/data/validation \
  --reasoning_mode vqa \
  --vqa_root ./vqa_output_train \
  --output_dir ./nureasoning_vla_qa_workspace
```

Passing `--vqa_root` without `--reasoning_mode` selects vqa mode automatically. A VQA question is used whenever a matching `*_vqa.json` exists; clips without VQA keep structured Spatial / Driving / Counterfactual text.

The same checkpoint is then used with `--provider nuvla` on the [challenge submission](submission.md) page. Planning uses `--planning_prompt` (default `auto`: structured multi-view for structured-trained checkpoints, scene-only for `--reasoning_mode vqa`). Answers use the VQA question prompt.

### Action expert (flow-matching DiT)

The DiT token sequence is `[state tokens ; action tokens]`. Cross-attention reads VLM features; with `--interleave_self_attention` (default on) layers alternate cross- and self-attention (GR00T pattern). Flow matching uses optimal-transport interpolation with `t ~ Beta(α, β)`:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--action_hidden_dim` | `512` | DiT width |
| `--num_dit_layers` | `12` | Transformer depth |
| `--num_dit_heads` | `8` | Attention heads (`hidden_dim` must be divisible by this) |
| `--dropout` | `0.1` | Attention / MLP dropout |
| `--mlp_ratio` | `4.0` | FFN expansion |
| `--interleave_self_attention` | on | Alternate cross- and self-attention blocks |
| `--num_inference_steps` | `5` | Euler steps when sampling a trajectory at eval time during training |
| `--num_timestep_buckets` | `1000` | Discrete grid for the flow-matching time embedding |
| `--noise_beta_alpha` | `2.5` | Beta `α` for training-time `t` |
| `--noise_beta_beta` | `1.5` | Beta `β` for training-time `t` |

Waypoints are normalized with a fixed scale `(50 m, 20 m, π rad)` inside the expert. Reported train metrics include per-axis MSE (`mse_x`, `mse_y`, `mse_theta`) in that normalized space.

### Images and trajectories

| Flag | Default | Meaning |
| --- | --- | --- |
| `--num_history_steps` | `1` | Number of past camera frames besides the current frame |
| `--history_stride` | `10` | Camera-history stride in 10 Hz frames (10 ⇒ 1 s) |
| `--current_res_w` / `--current_res_h` | `448` | Resize for the 8 current-time cameras |
| `--history_res_w` / `--history_res_h` | `448` | Resize for history cameras |
| `--trajectory_future_seconds` | `5.0` | Horizon of the action target |
| `--frame_rate_hz` | `10.0` | Native clip rate; unit of `--history_stride` and metadata fallback |
| `--num_trajectory_points` | `10` | Action-expert waypoints (10 × 0.5 s = 5 s) |
| `--max_history_traj_points` | `6` | Ego-history waypoints at 0.5 s (3 s) |

A sample is skipped if `frame_index < num_history_steps * camera_history_stride`, so one second of camera history always exists on 10 Hz clips.

## Validation

`python -m nureasoning.nuvla.evaluate` loads `training_config.json` from the parent of `--checkpoint_dir` (or from the checkpoint directory). Override only what the eval CLI exposes.

```bash
# Trajectory metrics (ADE / FDE / heading error) on the validation split
python -m nureasoning.nuvla.evaluate \
  --checkpoint_dir ./nureasoning_vla_workspace/final --mode planning

# Reasoning text generation
python -m nureasoning.nuvla.evaluate \
  --checkpoint_dir ./nureasoning_vla_workspace/final \
  --mode reasoning --num_reasoning_samples 5 --visualize

# Both, with figures
python -m nureasoning.nuvla.evaluate \
  --checkpoint_dir ./nureasoning_vla_workspace/final \
  --test_data_root ./dataset/data/validation \
  --mode both --visualize
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint_dir` | required | An `epoch_*` or `final` directory |
| `--test_data_root` | from `training_config.json` | Validation clips |
| `--output_dir` | workspace folder (parent of `--checkpoint_dir`) | Metrics, logs, optional figures |
| `--mode` | `both` | `reasoning`, `planning`, or `both` |
| `--keyframe_time_s` | `10.0` | One sample per clip, nearest this time; **negative = every frame** |
| `--num_inference_steps` | `4` | Flow-matching Euler steps for planning (can differ from train’s default 5) |
| `--num_reasoning_samples` | `10` | How many clips to decode in reasoning mode |
| `--max_new_tokens` | `4096` | Generation length |
| `--temperature` / `--top_p` | `0.1` / `0.1` | Nucleus sampling (low values ≈ greedy) |
| `--visualize` | off | Save per-sample figures |
| `--max_vis_samples` | `50` | Cap on visualization count |
| `--num_workers` | `4` | DataLoader workers |
| `--log_interval` | `100` | Progress log period |
| `--planning_prompt` | `auto` | Planning user prompt: `auto`, `structured`, or `minimal` |
| `--planning_reasoning_format` | checkpoint | Override structured sections for planning (`spatial`, `driving`, …) |

Planning mode scores ADE, FDE, and heading error over the selected frames (keyframe by default). The planning user prompt is `--planning_prompt` (default `auto`). Reasoning mode generates assistant text from the structured multi-view prompt (not the challenge VQA wording).

Open-loop ADE/FDE is **not** the challenge planning metric. For collision, driveable area, progress, comfort, and human likeness on the validation split, see [Planning benchmark](#planning-benchmark). For the official challenge JSON (trajectory plus multiple-choice answers), see [challenge submission](submission.md).

## Planning benchmark

`python -m nureasoning.planning.benchmark` scores a **nuVLA checkpoint** on clips that have ground truth (typically the validation split). At the key frame (index 100, t ≈ 10 s) the action expert samples a 5 s trajectory. The scorer resamples it onto a 0.1 s grid (51 waypoints, t = 0 through 5 s) in the map frame and compares it to future objects, the HD map, the route, and the human trajectory.

The planning user prompt is selectable (`nureasoning.nuvla.trajectory_provider`):

| `--planning_prompt` | Prompt |
| --- | --- |
| `auto` (default) | Structured multi-view prompt if the checkpoint was trained with `--reasoning_mode structured`; scene-only (cameras + mission) if trained with `vqa` |
| `structured` | Always request Spatial / Driving / Counterfactual sections (`--reasoning_format`, overridable with `--planning_reasoning_format`) |
| `minimal` | Cameras + mission only; no annotation-style request |

```bash
# nuVLA checkpoint on validation (ground truth available)
python -m nureasoning.planning.benchmark \
  --data_root ./dataset/data/validation \
  --mode vla \
  --checkpoint_dir ./nureasoning_vla_workspace/final
```

Five sub-scores per clip:

1. **Collision** — BEV overlap with future objects, classified as at-fault vs not. At-fault with a dynamic agent scores 0; with a static object, 0.5; otherwise 1.
2. **Driveable area** — 1 if every ego-box corner on the plan stays inside lane / road-block polygons (0.5 m buffer); otherwise 0.
3. **Progress** — distance along the ego route centerline, clamped to `[0, 1]` relative to the human (Euclidean fallback if there is no usable route).
4. **Comfort** — 1 only if accel, jerk, yaw-rate, and yaw-accel stay within fixed bounds for the whole horizon; otherwise 0.
5. **Human likeness** — 1 at ≤ 1 m FDE, 0 at ≥ 8 m, smoothstep in between. ADE/FDE are also reported.

The nuReasoning planning score (NPS) multiplies a weighted mix of progress, comfort, and human likeness by the collision and driveable-area gates:

```
NPS = S_collision × S_driveable × (0.3 S_progress + 0.2 S_comfort + 0.5 S_human)
```

Weights are overridable (`--w_progress`, `--w_comfort`, `--w_human`). Leaving the driveable area, or an at-fault collision with a dynamic agent, zeros the clip; a static at-fault collision halves it.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data_root` | `./dataset/data/validation` | Clip tree with map, annotations, and future ego |
| `--mode` | `vla` | `vla` (checkpoint), `gt` (oracle trajectory), or a baseline name (`constant_velocity`, `uniad`, `diffusion_drive`) |
| `--checkpoint_dir` | required for `vla` | An `epoch_*` or `final` directory |
| `--num_inference_steps` | `5` | Flow-matching Euler steps |
| `--key_frame_index` | `100` | Planning frame (clamped on shorter clips) |
| `--max_clips` | `0` | Cap; `0` = all clips |
| `--output_dir` | workspace folder (parent of `--checkpoint_dir`; else `nureasoning_planning_benchmark_output`) | JSON / CSV reports and figures |
| `--vis_clips` | `5` | Per-clip BEV overlays (`0` = none) |
| `--device` | `cuda` | Inference device |
| `--seed` | `42` | Python / NumPy / PyTorch |
| `--planning_prompt` | `auto` | `auto`, `structured`, or `minimal` (see above) |
| `--planning_reasoning_format` | checkpoint | Override structured sections for the planning prompt |

Outputs under `--output_dir`: `benchmark_report.json`, `benchmark_report.csv`, `benchmark_summary.csv`, `benchmark_summary.png`, and optional `clip_*.png`. Replot a saved report (or a single clip) with `python -m nureasoning.planning.visualize --report-json ... --skip-vla`.

The private **test** split has no future states, so this benchmark cannot run there. Submit trajectories through [challenge submission](submission.md) instead. Other planners are listed under [`nureasoning/planning/baselines`](../nureasoning/planning/baselines/README.md).
