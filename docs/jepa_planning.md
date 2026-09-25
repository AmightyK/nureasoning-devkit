# LeVJEPA + Intent JEPA + DiT Planning: Joint Design Draft

Status: M0--M2 implemented and synthetic-CPU tested; M3--M5 remain unvalidated.
Updated: 2026-09-25.

Implementation handoff: [subagent work packages and contracts](jepa_planning_implementation_plan.md).
That plan refines file ownership into a separate planning package while reusing
the existing action expert; the architecture and no-Qwen decision are unchanged.

## Agreed Direction

Keep JEPA as an active part of planning. Combine its predicted future-intent
representation with a nuVLA-style flow-matching DiT that generates waypoints.
Do not replace JEPA with a standalone DiT planner.

JEPA predicts **what future motion is appropriate** in a learned trajectory
representation. DiT converts that predicted intent, together with observation
context, into explicit future waypoints. This is a proposed hybrid, not the
original Auto-JEPA architecture or a reported result from either paper.

The selected visual backbone is **LeVJEPA**, not Qwen or V-JEPA 2. The planner
does not include a language backbone or a language reasoning-generation loss.
Checkpoint selection, conditioning details, loss weights, and training schedule
remain open. The design below is the working proposal for our next discussion.

## LeVJEPA Backbone

LeVJEPA is a video representation learner. Its visual pretraining uses view
invariance and SIGReg, without an objective-side EMA target encoder or predictor.
Our future-intent predictor and frozen trajectory target encoder are separate
downstream planning modules inspired by Auto-JEPA; they are not supplied by
LeVJEPA. The proposed planner is therefore a hybrid, not native LeVJEPA planning.
See the [official implementation](https://github.com/MLO-lab/LeVJEPA).

Candidate checkpoint: `galilai-group/LeVJEPA-VideoMix-Large`. Its
[model card](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large) specifies:

- ViT-L/16, approximately 303M parameters, feature width 1024.
- Input `[B, 3, 16, 224, 224]`, with ImageNet normalization.
- Output `[B, 3137, 1024]`: CLS plus spatiotemporal patch tokens.
- Block-causal attention; preserve the checkpoint's attention configuration.

Use pretrained patch features rather than only pooled CLS as the starting
proposal. Freeze the encoder initially and keep it in evaluation mode. Do not
automatically add SIGReg to downstream planning or replace the trajectory latent
loss with it; using a pretrained backbone does not require rerunning its original
pretraining objective. Any driving-video adaptation is a separate experiment.

For multiple cameras, initially encode each camera's chronological clip with
shared weights, then fuse/compress tokens with camera identity and timestamp
information. Do not treat different cameras as successive frames of one video.
All frames must be at or before the planning timestamp. Frame count and sampling
cadence need validation against the checkpoint's preprocessing.

Full-resolution tokens scale quickly: eight cameras at this input setting yield
25,096 tokens including CLS before compression. Choose a token budget and measure
memory/latency before fixing the fusion design; fewer backbone parameters than
Qwen do not by themselves establish lower end-to-end cost. Pin and inspect the
checkpoint's custom modeling code before execution; no weights or remote code
have been downloaded or executed for this draft.

## Training Architecture

```text
Past/current video -> LeVJEPA -> Token adapter/fusion --------> Fv
Ego history + velocity/acceleration -> State encoder ---------> Fh
Navigation command -> Command encoder -----------------------> Fc
                                                               |
                                             C = context(Fv,Fh,Fc)
                                              |                |
                                              v                |
                                      JEPA predictor           |
                                              |                |
                                              v                |
                                    Predicted intent z_hat     |
                                       |            |          |
                                       |            +-----+----+
                                       |                  |
                                       |             DiT condition
                                       |                  |
GT future Y -> Normalize -> Y_norm      |                  v
                             |         |          +----------------+
Gaussian noise eps ----------+         |          | Flow-matching  |
Flow time t -----------------+         |          | DiT            |
                             v         |          |                |
                x_t=(1-t)*eps+t*Y_norm  |          |                |
                             |         |          |                |
                   ActionEncoder(x_t,t) ---------->                |
Flow time t -> timestep embedding ---------------->                |
                                       |          +-------+--------+
                                       |                  |
                                       |             Action head
                                       |                  |
                                       |                  v
                                       |               v_pred
                                       v                  |
                                   JEPA loss          Flow loss
                                       ^                  ^
                                       |                  |
Y_norm -> Frozen target encoder -> z_target          Y_norm - eps
```

The target encoder consumes the same trajectory representation used during its
pretraining. Normalized waypoints are the proposed convention for this hybrid.

The context arrow is conceptual: inputs need not all be concatenated in one
sequence. A nuVLA-compatible implementation can keep ego state/history as a
state token, with scene, command, and predicted intent as cross-attention memory.

## Tensor Contract

| Tensor | Proposed shape | Meaning |
| --- | --- | --- |
| `Y` | `[B, 10, 3]` | Ten future `(x, y, heading)` poses over five seconds |
| `Y_norm` | `[B, 10, 3]` | Normalized target trajectory |
| `eps`, `x_t`, `v_pred` | `[B, 10, 3]` | Noise, interpolated trajectory, predicted flow |
| `t` | `[B]` | Flow time, not a future waypoint timestamp |
| `z_hat`, `z_target` | `[B, 10, D_z]` | Proposed time-aligned intent tokens |
| Action tokens | `[B, 10, D_a]` | Encoded noisy waypoints with time/position information |
| State token | `[B, 1, D_a]` | Ego dynamics and history |

Use current-ego-frame cumulative poses, not map coordinates or successive
waypoint increments. Preserve nuVLA's normalization `(50, 20, pi)` initially.
`D_z` is not yet fixed; `D_a=512` is a starting point from the existing action
expert. Use learned projections when feature widths differ, and preserve masks
for padded conditioning tokens.

Auto-JEPA's original eight 2D waypoints and `8 x 1024` latent are not a drop-in
contract for ten 3D poses. A compatible target encoder must be trained or adapted
and validated. Merely reshaping a checkpoint is not sufficient.

## Responsibilities and Conditioning

- JEPA receives observation context only, not noisy future waypoints or GT future.
- DiT receives noisy waypoint tokens, flow time, predicted intent, and context.
- `z_target` is a supervision target only; never feed it to DiT as conditioning.
- Retain direct scene conditioning as an initial proposal so DiT can access
  details not preserved in the intent representation.
- Check whether DiT actually uses intent: direct scene access could let it
  ignore `z_hat`. Compare normal, shuffled, and removed intent at validation.
- Encode categorical navigation commands with a separate learned embedding.
- LeVJEPA provides visual tokens, not text-conditioned features or QA answers.

The noisy action encoder is distinct from the frozen target trajectory encoder.
The first encodes `x_t` for generation; the second encodes clean GT for JEPA loss.

## Objectives and Gradient Flow

```text
x_t = (1 - t) * eps + t * Y_norm
v_target = Y_norm - eps
L_flow = mean_squared_error(v_pred, v_target)

L_JEPA = a * L_feature + b * L_token_cosine + c * L_InfoNCE
L_total = L_flow + lambda_JEPA * L_JEPA
```

The JEPA terms follow Auto-JEPA's feature alignment and contrastive formulation;
their weights must be evaluated for the new target representation. The target
encoder stays frozen after pretraining, with gradients stopped on `z_target`.
Batch InfoNCE depends on actual negatives, not the nominal batch obtained by
gradient accumulation. Similar trajectories may also form false negatives.

Proposed joint-training gradient policy:

| Loss | Updated modules |
| --- | --- |
| `L_flow` | Action encoder, DiT, action head, conditioning adapters, JEPA predictor, trainable context encoders |
| `L_JEPA` | JEPA predictor and trainable context encoders |
| Either loss | Never the frozen target encoder |

Do not detach `z_hat` in joint training if flow loss is intended to train JEPA.
Start with frozen LeVJEPA as a controlled experiment; any later unfreezing is a
separate decision. There is no language reasoning loss in this planner. If QA is
required, it is outside this planning design and can use a separate model.

## Proposed Training Stages

1. Train a trajectory autoencoder on training-split `Y_norm`. Verify coordinate,
   heading, endpoint, and motion reconstruction. Handle angular errors with a
   wrap-aware loss. Freeze the encoder once reconstruction is acceptable.
2. Warm up the JEPA predictor against frozen trajectory targets. Monitor latent
   alignment, variance, and sensitivity to scene/history/command changes.
3. Train the DiT using **predicted** intent, then jointly tune predictor and DiT
   with both losses. Whether a short frozen-predictor warmup helps is an ablation.

Keep the trajectory decoder for stage-one reconstruction checks; it is not
needed to generate final waypoints in the proposed DiT pipeline. No model is
trained on held-out evaluation or test futures.

## Inference

```text
Observations + ego state/history + command
                    |
          Context encoders -> JEPA -> z_hat
                    |                  |
                    +-------> DiT condition
                                     |
Gaussian noise -> repeated flow updates -> denormalize -> future waypoints
```

Compute context and intent once per scene, reuse them across integration steps.
At each step, predict `v_pred` and update `x = x + dt * v_pred`. The flow output
is a trajectory-space vector field, not physical vehicle velocity. Wrap final
headings after denormalization.

No GT trajectory or target encoder is used at inference. Retrieval and trajectory
memory are replaced by generation. A scorer or feasibility gate is outside the
initial baseline, not proven unnecessary for planning quality or safety.

## Implemented M0--M2 Package

| Area | Implementation |
| --- | --- |
| New planning package | `nureasoning/jepa_planning/`, independent of the legacy nuVLA entrypoints |
| JEPA modules | Trajectory autoencoder, intent predictor, wrap-aware reconstruction and latent objectives |
| Existing action expert | Imported and composed unchanged; raw futures enter its single internal normalization path |
| Backbone integration | Frozen LeVJEPA wrapper, audited-local loading boundary, camera/time-aware fixed-token adapter |
| Data loader | Causal timestamp sampling, command IDs, raw ego-frame targets, no reasoning-label dependency |
| Checkpoint handling | Strict staged artifacts with full config, module, representation, optimizer/scaler and RNG state |
| Evaluation/provider | Observation-only pure-noise inference, sparse metrics and 51-pose benchmark conversion |

The existing nuVLA path remains a separate baseline and its action-expert API is
unchanged. Unit tests use local stubs and synthetic fixtures. No released
LeVJEPA weights, remote model code, full dataset, or GPU training were used to
establish M0--M2.

## Verification and Experiments

- Test tensor shapes, ego-frame transforms, normalization and temporal ordering.
- Verify camera/time ordering, no post-anchor frames, frozen encoder evaluation
  mode, and token budgets on the intended camera count.
- Assert no gradients on the frozen target encoder and nonzero gradients through
  predicted intent to JEPA when joint training is enabled.
- Verify GT trajectories/reasoning answers never enter inference conditioning.
- Overfit a small training subset before larger runs.
- Test checkpoint reload and inference without future annotations.
- Evaluate from pure noise, not only one-step predictions from partially clean GT.
- Compare LeVJEPA + DiT without intent against LeVJEPA + intent JEPA + DiT using
  the same data, visual inputs and DiT settings. Include intent ablations.
- Keep original nuVLA as an external baseline, but do not attribute differences
  against Qwen-based nuVLA solely to intent JEPA: the backbone also changed.
- Track ADE/FDE, heading error, NPS components, latency and seed variability.
  Lower latent or flow loss does not establish better planning or safety.

## Decisions Still Open

1. Which LeVJEPA checkpoint revision, camera coverage, frame cadence, and token
   compression budget should be used?
2. What intent width and predictor size fit the compute budget?
3. Should DiT see intent plus full context, or a restricted context pathway?
4. When should the backbone be unfrozen, and which losses may update it?
5. Is a separate QA model needed outside the planner?
6. What loss weights, training-stage lengths, and inference step count to use?

## References

- [LeVJEPA official implementation](https://github.com/MLO-lab/LeVJEPA).
- [LeVJEPA paper](https://arxiv.org/abs/2608.27395).
- [LeVJEPA-VideoMix-Large model card](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large).
- [Auto-JEPA paper](https://arxiv.org/pdf/2607.29031): future-trajectory latent
  prediction, frozen target encoder, latent objectives, and retrieval planner.
- [nuReasoning Appendix C](https://arxiv.org/html/2605.31572v1#A3): nuVLA
  flow-matching action expert, trajectory representation, and integration.
- [Current nuVLA documentation](nuvla.md).
- [Existing action expert](../nureasoning/nuvla/models/action_expert.py).
- [Existing training loop](../nureasoning/nuvla/train.py).
