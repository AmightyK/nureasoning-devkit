# LeVJEPA + Intent JEPA + DiT: Subagent Implementation Plan

Status: M0--M2 implemented and synthetic-CPU tested. M3--M5 are prepared but unrun.
Date: 2026-09-25.
Architecture reference: [jepa_planning.md](jepa_planning.md).

## 1. Scope and Non-Negotiables

Build a planning-only pipeline with LeVJEPA visual features, an Auto-JEPA-style
future-intent predictor, and the existing nuVLA flow-matching action expert.

```text
Past/current camera clips -> frozen LeVJEPA -> token adapter -> scene tokens
                                                                 |
Ego state/history + command ---------------------> intent predictor
                                                                 |
                                                         predicted intent
                                                                 |
scene + command + predicted intent ---------------------> DiT condition
noisy waypoints + flow time + ego state/history ---------> DiT -> flow

GT future -> pretrained frozen trajectory encoder -> target intent
                                      predicted intent <-> JEPA loss
```

- No Qwen, V-JEPA 2, text tokenizer, language loss, or GT reasoning input.
- Keep intent JEPA; do not reduce the requested model to LeVJEPA + DiT alone.
- Never condition DiT on GT target intent, including during training warmup.
- Reuse nuVLA waypoint normalization, flow objective, and sampling machinery.
- Preserve existing nuVLA behavior and checkpoints as a separate baseline.
- Do not add RL, retrieval memory, scorers, safety gates, QA, or submission upload.
- Do not automatically download weights, launch full training, or change the
  environment during module development. Real-model smoke tests are explicit.
- LeVJEPA pretraining and planning JEPA are distinct. Do not add SIGReg or an
  EMA teacher to planning merely because the backbone uses the LeVJEPA name.

## 2. Integration Boundary

Create a separate package `nureasoning/jepa_planning/`. This refines the earlier
architecture draft's file map: use a new training entrypoint instead of expanding
the Qwen-specific `nureasoning/nuvla/train.py` for the MVP.

Reuse `nureasoning/nuvla/models/action_expert.py` via imports and composition.
Set its `vlm_feature_dim` to the chosen context width; despite the legacy name,
that input can carry projected visual/intent tokens. Do not fork the DiT code.
Initially use fixed-length, fully valid context tokens so no shared attention
mask API change is needed. Missing-input handling must be explicit, not padding
unmasked invalid tokens into the legacy action expert.

Reuse existing clip discovery, safe project-specific pickle loading, coordinate
and trajectory utilities where suitable. Inspect the real schema before coding.
Do not instantiate the reasoning-filtered VLA dataset for planning-only data.
Any necessary extraction of shared helpers belongs to the integration owner,
with regression tests; other agents must not independently refactor old files.

Existing workspace changes in nuVLA training, data loading, docs, and environment
files are user-owned. Do not revert or overwrite them.

## 3. Interface Contract: Freeze Before Parallel Coding

`B`: batch; `V`: configured camera count; `F`: video frames; `K`: compressed
scene-token count; `D`: context/intent width. Use Python dataclasses or TypedDicts
for these contracts, matching local style, not an additional validation framework.

### Data contract

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `video` | `[B,V,3,F,224,224]` float | ImageNet-normalized chronological clips |
| `camera_ids` | `[B,V]` integer | Stable vocabulary, not positional guesses |
| `frame_times_s` | `[B,V,F]` float | Actual times relative to anchor; all <= 0 |
| `ego_state` | `[B,4]` float | `(vx,vy,ax,ay)` in current ego axes |
| `history` | `[B,6,3]` float | Past ego-frame poses on the agreed 0.5 s grid |
| `command_id` | `[B]` integer | Stable vocabulary plus explicit unknown ID |
| `future` | `[B,10,3]` float | Training target only; raw ego-frame poses |
| `sample_id` | list of strings | Clip and anchor identity for diagnostics |

Define observation-only and training-batch types separately. `predict()` must
accept no future target. All poses use the same fixed anchor frame, not successive
displacements. Future times are 0.5, 1.0, ..., 5.0 seconds. Proposed history times
are -3.0, -2.5, ..., -0.5 seconds; verify against current data utilities.

For the MVP, require complete configured camera/history inputs. Training reports
excluded samples and reasons; evaluation reports failures in coverage rather
than silently dropping them. Do not invent future labels, repeat future frames,
or silently fill missing ego states with zeros.

### Model contract

```text
LeVJEPABackbone.forward(video) -> patch_tokens [B,V,P,1024]
SceneAdapter.forward(patch_tokens, camera_ids, frame_times_s)
    -> scene_tokens [B,K,D]
TargetTrajectoryEncoder.forward(normalized_future) -> z_target [B,10,D]
TrajectoryDecoder.forward(z_target) -> reconstructed_future [B,10,3]
IntentPredictor.forward(scene_tokens, state_token, command_token)
    -> z_hat [B,10,D]
PlanningModel.compute_losses(training_batch) -> scalar losses + diagnostics
PlanningModel.predict(observation_batch, num_steps) -> raw poses [B,10,3]
```

`P` is obtained from the audited checkpoint's patch layout; do not assume CLS
is a patch token. Candidate checkpoint metadata is recorded in the architecture
document; verify it at runtime and fail clearly on incompatible output.

Proposed DiT cross-attention memory:

```text
C_dit = concat(scene_tokens, command_token, z_hat)  # [B,K+1+10,D]
action_expert(x_1=future, vlm_features=C_dit,
              ego_state=ego_state, history_trajectory=history)
```

Reuse the action expert's `state_encoder` to produce the intent predictor's
state token. Register that module only once under the action expert, so optimizer
parameters and checkpoint keys are not duplicated. Recomputing its small MLP
inside the existing action forward is acceptable for the MVP.

### Proposed starter settings, not measured optima

- Context and intent width `D=512`; compressed scene tokens `K=64`.
- Intent predictor: four Transformer blocks, eight heads, ten learned queries.
- Video input: 16 frames spanning approximately two seconds, including the
  current anchor; timestamp-based causal sampling with a documented tolerance.
- Tiny smoke configuration: front camera only. Dataset and model APIs support
  a fixed ordered camera list; multiview is required before final comparisons.
- Keep the existing action-expert defaults unless explicitly overridden in the
  saved config. Resolve defaults in one place, not separately in train/inference.
- Explicitly configure latent-loss weights and `lambda_jepa`; do not describe
  copied Auto-JEPA weights as tuned for this hybrid.

Agent 0 must commit the exact config/schema choices before other agents rely on
them. In particular, decide the real timestamp sampling policy from the dataset
and model preprocessing, not by rounding 7.5 fps into a constant 10 Hz stride.

## 4. Work Packages and File Ownership

All paths below are relative to `nureasoning-devkit/`. Each agent owns only its
listed files and a corresponding test file. Shared changes go through Agent 0.

### Agent 0: Contracts and Integration Owner

Own: package `__init__.py`, `config.py`, `contracts.py`, `configs/smoke.yaml`,
`configs/train.yaml`, this plan, and any approved shared-file/dependency edits.
These package paths are under `nureasoning/jepa_planning/`.

Deliverables:

- Freeze field names, tensor units, camera/command vocabularies, API signatures,
  configuration serialization, and the exact frozen-module policy.
- Resolve unknown checkpoint revision/code compatibility before real-model runs.
- Provide config validation and lightweight import tests without downloading a
  backbone or importing Qwen-specific training code.
- Integrate modules, review shared-file changes, and keep ownership conflicts out
  of parallel tasks. Do not mutate global environment metadata without need.

Acceptance: every other agent can build against the same contracts using mocks;
all starter configuration values are explicit and checkpoint-serializable.

### Agent 1: Planning Dataset and Causal Preprocessing

Own: `nureasoning/jepa_planning/data.py`, `tests/test_jepa_data.py`.

- Build a planning-only dataset and collator; reasoning JSON is not required.
- Implement observation loading reusable by both training and inference, without
  requiring future metadata to build observation tensors.
- Sample each camera clip chronologically with frames no later than the anchor.
  Use consistent deterministic resize/crop across a clip; avoid untracked flips
  or rotations that invalidate waypoint targets and camera geometry.
- Extract ego state/history/future in the anchor ego frame; verify angular wrap,
  history spacing, timestamp tolerances, and vector rotation.
- Freeze vocabularies in config, apply clip-level subset selection only to train,
  and preserve train/validation separation by clip/log.

Acceptance: synthetic metadata tests for coordinate round trips, exact target
times, no future leakage, missing-input reporting, no reasoning dependency,
observation-only test data, reproducible subsets, and one/eight-camera shapes.

### Agent 2: LeVJEPA Wrapper and Scene Token Adapter

Own: `nureasoning/jepa_planning/backbone.py`, `tests/test_jepa_backbone.py`.

- Wrap the audited pretrained checkpoint using existing cache/path helpers.
  Lazy-load optional model code, accept a local path, and record immutable
  revision/code identity. Review custom code before enabling remote execution.
- Encode cameras with shared weights; allow camera chunking to control memory.
- Keep the frozen backbone in eval mode even after the parent calls `train()`.
  Frozen features must still be usable by a trainable adapter in autograd.
- Implement a fixed-budget trainable token adapter, preserving spatiotemporal
  patch identity and adding camera/time information before compression.
  If temporal layout changes, validate against checkpoint metadata.
- Use a stub backbone in unit tests; do not download weights in tests.

Acceptance: correct output contract, adapter gradients with frozen backbone,
camera identity sensitivity, correct CLS handling, parent-train freeze behavior,
and bounded compressed sequence length. Report real-GPU memory separately.

### Agent 3: Trajectory Representation and Intent JEPA

Own: `nureasoning/jepa_planning/trajectory.py`, `intent.py`, `losses.py` in that
package; `tests/test_jepa_intent.py`.

- Implement trajectory encoder/decoder for normalized `[B,10,3]` and a temporal
  intent predictor conditioned on scene/state/command tokens.
- Implement reconstruction losses with wrap-aware heading handling; compute
  motion terms using physical units and actual 0.5 s spacing.
- Implement feature, token-cosine, and configurable batch InfoNCE terms. A
  one-sample batch must not falsely claim meaningful contrastive learning.
  Cross-rank negatives may be deferred, but document local-negative semantics.
- Return named scalar losses and latent variance/alignment diagnostics.
- Frozen target weights require pretraining or a compatible checkpoint. Do not
  quietly freeze random initialization or use an EMA target without redesign.

Acceptance: tiny trajectory reconstruction overfit, finite losses/gradients,
batch-one handling, heading-wrap tests, temporal-token shapes, frozen target
tests, and state/command sensitivity. No code may read GT inside the predictor.

### Agent 4: Planner Composition and DiT Conditioning

Own: `nureasoning/jepa_planning/model.py`, `tests/test_jepa_model.py`.

- Compose Agents 2/3 with the existing action expert; own command embedding and
  use the action expert's state encoder as specified above.
- Compute `z_hat` from observations, construct `C_dit`, and pass **raw** future
  waypoints to the existing flow loss, which already normalizes them.
- Normalize separately once for target-encoder inputs. Avoid double normalization.
- Keep `z_hat` attached so flow loss can train the predictor. Compute target
  latent without gradients; never use it as conditioning.
- Implement observation-only sampling; compute context/intent once per scene,
  reuse across integration steps, and return denormalized wrapped poses.
- Support explicit `no_intent` and validation-only shuffled-intent ablations.
  The default model must always include predicted intent.

Acceptance: no-target inference, finite loss and output shapes, deterministic
seeded eval, no frozen-target gradients, flow gradients to predictor/adapter,
and unchanged existing action-expert tests. The action decoder is zero-initialized:
test upstream gradient flow after a warmup optimizer update, not only at step 0.

### Agent 5: Training Stages and Checkpoint Lifecycle

Own: `nureasoning/jepa_planning/train.py`, `checkpoint.py` in that package;
`tests/test_jepa_training.py`.

- Add one entrypoint with stages `trajectory_ae`, `intent`, and `joint`.
- Stage A: train trajectory autoencoder only; save representation/preprocessing
  metadata and encoder/decoder weights.
- Stage B: require compatible Stage A encoder; freeze it and LeVJEPA; train
  predictor, state/command encoding and scene adapter with latent losses.
- Stage C: retain Stage B weights; train DiT and predictor jointly with
  `L_flow + lambda_jepa * L_jepa`. Freeze target encoder and LeVJEPA.
- Validate trainable parameter groups per stage; avoid duplicate parameters and
  unused-parameter problems in DDP. Start with single-device smoke execution.
- Implement seeded runs, named losses, gradient norms, AMP/accumulation, proper
  partial accumulation handling, validation, and fail-fast nonfinite checks.
- Save all trainable modules, target encoder identity/weights, backbone revision,
  normalization, vocabularies, camera/time config, loss weights, stage, optimizer,
  scheduler/scaler and RNG states. Restore consistently on resume.
- Do not silently accept legacy Qwen checkpoints or missing random module weights.

Acceptance: stage-specific update/freeze tests, round-trip checkpoint prediction,
resume equivalence within stated tolerances, incompatible-config errors and a
small synthetic CPU training run. Full GPU training is not part of this task.

### Agent 6: Evaluation and Trajectory Provider

Own: `nureasoning/jepa_planning/evaluate.py`, `trajectory_provider.py` in that
package; `tests/test_jepa_evaluation.py`.

- Load the new checkpoint without instantiating Qwen, using Agent 1's observation
  preprocessing and Agent 4's `predict()`.
- Evaluate trajectories generated from pure Gaussian noise, not GT-noised inputs.
- Report ADE/FDE, wrapped heading error, coverage/failures, seed, latency, and
  existing planning metrics where the evaluator and annotations are available.
- Reuse existing benchmark callback conventions; distinguish its global-frame
  interface from current-ego model output. Reuse tested interpolation/conversion
  helpers where possible; document conversion to 51 poses if needed downstream.
- Do not modify the official benchmark or infer test routes/future states. Do not
  add upload or reasoning-answer generation.

Acceptance: observation-only fixtures with future removed, keyframe anchoring,
heading interpolation across wrap, coverage accounting, checkpoint consistency,
and explicit errors for incompatible providers or unavailable evaluator assets.

## 5. Dependencies and Parallel Work

```text
Agent 0: freeze contracts
          |
          +--> Agent 1: data -----------+
          +--> Agent 2: backbone ------+--> Agent 4: planner
          +--> Agent 3: intent --------+          |
                                                 +--> Agent 5: train/checkpoint
                                                 +--> Agent 6: eval/provider
                                                              |
                                  Agent 0: integrated acceptance
```

Agents 4-6 may prepare against mocks after contracts are frozen. Agent 6 needs
Agent 5's finalized checkpoint contract before integration. Do not merge by
reimplementing another agent's unfinished module; use its agreed interface.

## 6. Milestones and Acceptance Gates

- [x] M0: Contracts/config frozen; ownership acknowledged; no architecture drift.
- [x] M1: Data, backbone stub, trajectory and intent modules pass unit tests.
- [x] M2: Combined CPU smoke: loss/backward, seeded sampling, checkpoint reload.
- [ ] M3: Audited real LeVJEPA checkpoint smoke on available GPU; record shapes,
  causal input window, peak memory, runtime, and multiview feasibility.
- [ ] M4: Small-subset three-stage overfit; inspect trajectories and gradient flow.
- [ ] M5: Held-out comparison: LeVJEPA + DiT versus LeVJEPA + intent JEPA + DiT;
  include removed/shuffled-intent diagnostics and report failures.

Use the repository's existing `unittest` style. Proposed test commands after
implementation (not claims that these tests currently exist):

```bash
python -m unittest discover -s tests -p 'test_jepa_*.py'
python -m unittest discover -s tests -p 'test_nuvla_*.py'
```

Passing synthetic tests does not establish real checkpoint compatibility or
planning performance. Report unavailable GPU/data checks as unrun, not passed.
Do not mark M3-M5 complete without the associated evidence and artifacts.

## 8. Runnable Stage Commands

The following commands use the pinned local environment. Values in
`configs/train.yaml`, including loss weights and stage lengths, are starting
configurations rather than validated settings. Before M3, set an audited local
LeVJEPA snapshot plus immutable model/code identities in that file; the loader
does not download weights or execute Hub code automatically.

```bash
uv run --frozen python -m nureasoning.jepa_planning.train \
  --config nureasoning/jepa_planning/configs/train.yaml \
  --stage trajectory_ae \
  --checkpoint-path outputs/jepa_planning/trajectory_ae.pt

uv run --frozen python -m nureasoning.jepa_planning.train \
  --config nureasoning/jepa_planning/configs/train.yaml \
  --stage intent \
  --init-checkpoint outputs/jepa_planning/trajectory_ae.pt \
  --checkpoint-path outputs/jepa_planning/intent.pt

uv run --frozen python -m nureasoning.jepa_planning.train \
  --config nureasoning/jepa_planning/configs/train.yaml \
  --stage joint \
  --init-checkpoint outputs/jepa_planning/intent.pt \
  --checkpoint-path outputs/jepa_planning/joint.pt

uv run --frozen python -m nureasoning.jepa_planning.evaluate metrics \
  --checkpoint outputs/jepa_planning/joint.pt \
  --data-root ./dataset/data/validation \
  --device cuda \
  --seeds 42,43,44 \
  --intent-mode predicted \
  --batch-size 2 \
  --output outputs/jepa_planning/evaluation.json

uv run --frozen python -m nureasoning.jepa_planning.evaluate benchmark \
  --checkpoint outputs/jepa_planning/joint.pt \
  --data-root ./dataset/data/validation \
  --device cuda \
  --seed 42 \
  --intent-mode predicted \
  --output outputs/jepa_planning/benchmark.json
```

Use `--resume <incomplete-same-stage.pt>` for epoch-boundary continuation.
Stage transitions require a completed prerequisite checkpoint and reject legacy
nuVLA/Qwen checkpoints rather than silently accepting missing modules.

For the M5 intent-use diagnostics, rerun metrics with identical checkpoint,
data, seeds, steps, and batch size while changing only `--intent-mode`:

```bash
for mode in predicted no_intent shuffled; do
  uv run --frozen python -m nureasoning.jepa_planning.evaluate metrics \
    --checkpoint outputs/jepa_planning/joint.pt \
    --data-root ./dataset/data/validation \
    --device cuda --seeds 42,43,44 --batch-size 2 \
    --intent-mode "$mode" \
    --output "outputs/jepa_planning/evaluation_${mode}.json"
done
```

`shuffled` is intentionally metrics-only because the benchmark provider plans
one scene per callback. M3 still requires setting `backbone.local_path`,
`backbone.revision`, `backbone.code_revision`, and
`backbone.allow_audited_local_code: true` after auditing the local snapshot.
Record the wrapper-reported identity, token shapes, peak GPU memory and runtime.
M4 requires actual released training data and a deliberately small
`train_clip_fraction`; the synthetic CPU lifecycle test is not an M4 result.

## 7. Copyable Subagent Assignment

```text
Implement Agent <N> from docs/jepa_planning_implementation_plan.md.
Read docs/jepa_planning.md and the contract frozen by Agent 0 first.
Own only the files listed for your work package and its tests.
Preserve existing user changes and the original nuVLA path.
Do not introduce Qwen, replace LeVJEPA, remove intent JEPA, or use GT as condition.
Use stubs/local fixtures in tests; no automatic downloads or full training runs.
Coordinate contract changes with Agent 0 before editing another owner's files.
Return: changed files, public interfaces, tests and results, assumptions,
unrun checks, and integration blockers. Do not claim performance improvements.
```

Agent 0's final handoff should include a runnable smoke configuration, exact
commands for the three training stages and evaluation, checkpoint structure,
the tests actually run, and a concise list of remaining data/GPU requirements.
