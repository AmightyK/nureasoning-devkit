# Challenge submission

The private test set is scored as a combined **planning + reasoning** challenge. Released clips contain cameras, ego state through the key frame, and questions. You generate a single JSON file locally and upload that file for scoring.

This page is the end-to-end tutorial. Worked cells live in [`tutorials/nureasoning_challenge_submission.ipynb`](../tutorials/nureasoning_challenge_submission.ipynb).

## Test data

Download the test split (or point `--data-root` at a copy you already have):

```bash
python -m nureasoning.dataset.download --local-dir ./dataset --splits test
```

Layout (1,000 clips, 10 s of context through the key frame):

```
dataset/data/test/
└── <clip_name>/
    ├── metadata.json              # frames, sensors.cameras, key_frame_index=100
    ├── cameras/CAM_M_*/...        # 8-camera JPEGs
    ├── ego_state/*.pkl            # ego pose / history (no trajectory_future)
    └── reasoning_questions.json   # key-frame questions; no answers
```

Each clip has `key_frame_index = 100`. The 3,000 challenge questions are all multiple-choice (`A`–`D`; a few have only `A`/`B`) covering Spatial, Driving, and Counterfactual categories. A few clips have no questions; still submit a trajectory for every clip, with `"answers": []` when there is nothing to answer.

## Submission JSON

One object, one list of **clips**. Each clip entry holds **both** the planned trajectory and that clip's question answers:

```json
{
  "meta": {
    "coordinate_frame": "ego",
    "trajectory_dt_s": 0.1,
    "trajectory_horizon_s": 5.0,
    "trajectory_steps": 51,
    "num_clips": 1000
  },
  "clips": [
    {
      "clip": "<clip_dir_name>",
      "clip_token": "<token>",
      "log_name": "<log_name>",
      "target_frame_index": 100,
      "trajectory": [[dx, dy, dtheta], "... 51 waypoints ..."],
      "answers": [
        {"question_id": "<id>", "answer": "A"}
      ]
    }
  ]
}
```

| Field | Requirement |
| --- | --- |
| `meta.coordinate_frame` | Must be `"ego"` |
| `clips[].clip` | Directory name of the test clip |
| `clips[].target_frame_index` | Key frame used for planning (100 on the released test set) |
| `clips[].trajectory` | 51 `[dx, dy, dtheta]` waypoints, 0.1 s apart from t = 0 (key frame) to 5 s, in the key-frame ego frame |
| `clips[].answers` | One object per question on that clip; `answer` is the choice letter |

Extra fields (`answer_text`, `clip_token`, `log_name`, …) are ignored by the scorer. Duplicate clips, duplicate `question_id`s, the wrong trajectory shape, or a non-ego coordinate frame all score as zero for the affected items.

Validate a file before upload:

```bash
python -m nureasoning.submission.challenge \
  --data-root ./dataset/data/test \
  --validate-only ./challenge_submission.json
```

## Path A — one model (nuVLA trained with QA)

This is the reference solution in the devkit: the VLM adapter answers the challenge questions and the flow-matching action expert plans the trajectory, from a **single checkpoint**.

### 1. Materialize VQA from the training annotations

```bash
python -m nureasoning.vqa.generate \
  --data-root ./dataset/data/train --output ./vqa_output_train
```

### 2. Train nuVLA with VQA as the text target

`--vqa_root` uses generated multiple-choice questions as the VLM text target (same prompt/answer format the challenge uses). The action expert is still trained on every sample, so one run produces both heads. Clips without a matching VQA file keep the original structured-reasoning text.

```bash
# Single GPU
python -m nureasoning.nuvla.train \
  --data_root ./dataset/data/train \
  --test_data_root ./dataset/data/validation \
  --vqa_root ./vqa_output_train \
  --output_dir ./nureasoning_vla_qa_workspace

# Multi-GPU
torchrun --nproc_per_node=8 -m nureasoning.nuvla.train \
  --data_root ./dataset/data/train \
  --test_data_root ./dataset/data/validation \
  --vqa_root ./vqa_output_train \
  --output_dir ./nureasoning_vla_qa_workspace
```

### 3. Generate the submission

```bash
python -m nureasoning.submission.challenge \
  --data-root ./dataset/data/test \
  --provider nuvla \
  --checkpoint-dir ./nureasoning_vla_qa_workspace/final \
  --output challenge_submission.json
```

`--provider nuvla` loads the checkpoint once and, for each clip, samples a trajectory then answers every question with the same VLM. Planning uses `--planning-prompt` (default `auto`). Challenge answers always use the VQA question wording.

## Path B — two models (nuVLA planning + reasoning SFT)

Use this when you already trained a planning nuVLA **and** a separate reasoning VLM (`nureasoning.reasoning.train`). Trajectories come from the nuVLA action expert; answers come from the served reasoning model.

```bash
# Terminal 2 — vLLM env, leave running. See docs/installation.md.
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve ./reasoning_workspace/merged_qwen3.5-4b-multiframe \
  --served-model-name nureasoning-4b-sft \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.7 \
  --dtype bfloat16 \
  --mm-processor-kwargs '{"max_pixels": 200704}' \
  --enable-prefix-caching
```

```bash
python -m nureasoning.submission.challenge \
  --data-root ./dataset/data/test \
  --provider split \
  --planning-provider vla --checkpoint-dir ./nureasoning_vla_workspace/final \
  --reasoning-provider api --api-model nureasoning-4b-sft \
  --output challenge_submission.json
```

`--reasoning-provider vlm` loads a Hugging Face / nuVLA VLM in-process instead of an HTTP server. `--planning-provider constant_velocity` is the no-checkpoint sanity baseline.

## Scoring

Upload `challenge_submission.json` to the Hugging Face competition ([nureasoning-challenge-2026](https://huggingface.co/spaces/nureasoning/nureasoning-challenge-2026)). The evaluator computes:

- **Planning score**: mean target-frame-gated nuReasoning planning score (NPS) over every expected clip. Missing or invalid trajectories score zero.
- **Reasoning score**: exact multiple-choice accuracy over every expected question. Missing, duplicate, or invalid answers score zero.
- **Final score** = `0.75 * planning_score + 0.25 * reasoning_score`.

Generation writes each finished clip to ``<output>.partial.jsonl`` (same stem as ``--output``) so a crash keeps completed trajectories and answers. Re-running the same command skips clips already in that file. After the official JSON is written, the sidecar is deleted.

Generation is strict by default: if any clip's trajectory or any question fails, the official JSON is not written. `--allow-partial` is for debugging only.
