# Runbook tiếp tục M3–M5: LeVJEPA + Intent JEPA + DiT

Tài liệu này là handoff thực thi cho máy có dữ liệu và GPU. Đây không phải là
kết quả benchmark. Chỉ đánh dấu một milestone hoàn thành khi các artifact và
tiêu chí tương ứng bên dưới đã tồn tại.

## 1. Trạng thái hiện tại

| Milestone | Trạng thái | Bằng chứng hiện có |
| --- | --- | --- |
| M0 — contracts/config | Hoàn thành | Config, tensor contracts và freeze policy đã có test |
| M1 — module tests | Hoàn thành | Data, backbone stub, trajectory AE và intent JEPA đã có unit test |
| M2 — CPU integration | Hoàn thành | Loss/backward, sampling, checkpoint lifecycle và provider dùng fixture synthetic |
| M3 — LeVJEPA thật | Chưa chạy | Thiếu local audited snapshot và phép đo GPU |
| M4 — subset overfit | Chưa chạy | Thiếu train/validation data thật và ba checkpoint đã train |
| M5 — held-out comparison | Chưa chạy | Thiếu completed joint checkpoint và báo cáo valid |

Các ràng buộc không được thay đổi trong M3–M5:

- Backbone là LeVJEPA, không thay bằng Qwen hoặc V-JEPA 2.
- Target trajectory encoder phải được train ở Stage A, sau đó frozen.
- DiT chỉ nhận predicted intent hoặc ablation `no_intent`; không nhận target
  intent hoặc future ground truth làm condition.
- Future truyền vào action expert ở raw ego-frame units; action expert thực
  hiện normalization duy nhất của flow loss.
- Inference chỉ nhận observation, khởi tạo từ Gaussian noise và xuất mười pose
  `(x, y, heading)` tại 0.5–5.0 giây.
- Không dùng reasoning annotation để quyết định planning sample eligibility.

## 2. Đầu vào bắt buộc trên máy chạy

Mặc định repository dùng cấu trúc:

```text
dataset/data/train/<clip>/metadata.json
dataset/data/validation/<clip>/metadata.json
models/LeVJEPA-VideoMix-Large/config.json
models/LeVJEPA-VideoMix-Large/*.safetensors
```

Nếu dữ liệu nằm ở vị trí khác, sửa `data.train_root` và `data.val_root` trong
`nureasoning/jepa_planning/configs/train.yaml`.

Trước khi chạy:

```bash
uv sync --locked
nvidia-smi
uv run --frozen python -m unittest discover -s tests -p 'test_jepa_*.py'
uv run --frozen python -m unittest discover -s tests -p 'test_nuvla_*.py'
```

Không tiếp tục nếu test regression thất bại.

## 3. M3 — audit và smoke test LeVJEPA thật

### M3.1 Audit snapshot cục bộ

Model card yêu cầu custom model code. Không bật tải hoặc chạy code từ Hub một
cách tự động. Review các file Python/config trong snapshot và lưu manifest:

```bash
mkdir -p outputs/jepa_planning/m3
find models/LeVJEPA-VideoMix-Large -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > outputs/jepa_planning/m3/levjepa_snapshot_sha256.txt
```

Ghi lại nguồn snapshot, weight commit, code commit và người/ngày review vào
`outputs/jepa_planning/m3/audit_notes.md`. Kiểm tra đặc biệt:

- input phải là `[B, 3, 16, 224, 224]`, ImageNet normalization;
- block-causal attention không bị đổi sang full attention;
- eval mode không thực hiện token dropping;
- output chứa một CLS token cộng `16 × 14 × 14 = 3136` patch tokens;
- không có network/subprocess/file mutation không mong muốn trong model code.

Sau khi audit, sửa config:

```yaml
backbone:
  local_path: ./models/LeVJEPA-VideoMix-Large
  revision: <immutable-weight-identity>
  code_revision: <audited-code-identity>
  local_files_only: true
  trust_remote_code: false
  allow_audited_local_code: true
```

### M3.2 Audit planning data

Chạy index coverage trước training:

```bash
uv run --frozen python - <<'PY'
from nureasoning.jepa_planning.config import load_config
from nureasoning.jepa_planning.data import (
    PlanningDataset,
    assert_disjoint_planning_splits,
)

cfg = load_config("nureasoning/jepa_planning/configs/train.yaml")
train = PlanningDataset(cfg.data, split="train", observation_only=False)
valid = PlanningDataset(cfg.data, split="val", observation_only=False)
assert_disjoint_planning_splits(train, valid)
print("train", train.coverage)
print("valid", valid.coverage)
PY
```

Không train nếu train/validation trùng clip hoặc log. Review mọi exclusion reason,
đặc biệt `missing_video_frame`, `missing_history_frame`, `missing_future_frame`,
camera thiếu và timestamp tolerance. Không nới tolerance chỉ để tăng coverage
mà chưa kiểm tra frame cadence thực tế.

### M3.3 GPU smoke

Bắt đầu với `data.cameras: [front]`, batch size 1, `camera_chunk_size: 1` và AMP.
Chạy một observation qua backbone, adapter và predictor; ghi lại:

- backbone identity và audited revisions;
- input/output shapes, `PatchLayout` và dtype;
- peak allocated/reserved GPU memory;
- backbone time và end-to-end conditioning time;
- xác nhận backbone vẫn `eval()` và không có trainable parameter.

M3 chỉ pass khi có artifact log cho một camera. Sau đó thử tám camera và ghi rõ
pass/OOM. OOM tám camera không được che giấu; giữ camera chunking hoặc giảm camera
coverage trong một config thí nghiệm riêng.

Expected candidate shapes, cần xác nhận từ model thật:

```text
video             [1, V, 3, 16, 224, 224]
patch_tokens      [1, V, 3136, 1024]
scene_tokens      [1, 64, 512]
predicted_intent  [1, 10, 512]
DiT context       [1, 75, 512]  # 64 scene + 1 command + 10 intent
```

M3 artifact tối thiểu:

```text
outputs/jepa_planning/m3/
  audit_notes.md
  levjepa_snapshot_sha256.txt
  front_camera_smoke.json
  multiview_smoke.json
```

## 4. M4 — overfit subset thật qua ba stage

Tạo bản sao config, ví dụ
`nureasoning/jepa_planning/configs/m4_overfit.yaml`. Không sửa config production
trong khi một run đang chạy.

Starter settings cho GPU 8 GB:

```yaml
data:
  cameras: [front]
  train_clip_fraction: 0.01  # điều chỉnh để còn vài clip hoàn chỉnh
model:
  scene_tokens: 64
backbone:
  camera_chunk_size: 1
training:
  batch_size: 1
  val_batch_size: 1
  num_workers: 2
  amp: true
  gradient_accumulation_steps: 4
  output_dir: ./outputs/jepa_planning/m4
```

Các giá trị này chỉ là điểm bắt đầu. Kiểm tra số sample thực sau subsampling.
InfoNCE không có negative thực khi physical batch size là 1; gradient
accumulation không làm batch-negative lớn hơn. Nếu cần đánh giá InfoNCE, dùng
physical batch ≥ 2 hoặc tạm đặt `loss.info_nce: 0.0` trong một config được ghi
nhãn rõ ràng.

### M4.1 Stage A — trajectory autoencoder

```bash
uv run --frozen python -m nureasoning.jepa_planning.train \
  --config nureasoning/jepa_planning/configs/m4_overfit.yaml \
  --stage trajectory_ae --device cuda \
  --checkpoint-path outputs/jepa_planning/m4/trajectory_ae.pt \
  2>&1 | tee outputs/jepa_planning/m4/trajectory_ae.log
```

Acceptance gate:

- checkpoint có `stage=trajectory_ae`, `stage_complete=true`;
- reconstruction loss có xu hướng giảm, không non-finite;
- endpoint `(x,y)`, wrapped heading và motion reconstruction được inspect trên
  các sample thật;
- target encoder chỉ được mark pretrained sau khi stage hoàn tất.

### M4.2 Stage B — intent pretraining

```bash
uv run --frozen python -m nureasoning.jepa_planning.train \
  --config nureasoning/jepa_planning/configs/m4_overfit.yaml \
  --stage intent --device cuda \
  --init-checkpoint outputs/jepa_planning/m4/trajectory_ae.pt \
  --checkpoint-path outputs/jepa_planning/m4/intent.pt \
  2>&1 | tee outputs/jepa_planning/m4/intent.log
```

Acceptance gate:

- target encoder frozen, không có gradient hoặc weight change;
- predictor, scene adapter, command embedding và shared state encoder update;
- feature/cosine loss giảm, predicted latent variance không collapse về zero;
- intent thay đổi khi perturb scene/history/command trên dữ liệu thật;
- ghi rõ physical minibatch size và số local negatives.

### M4.3 Stage C — joint training

```bash
uv run --frozen python -m nureasoning.jepa_planning.train \
  --config nureasoning/jepa_planning/configs/m4_overfit.yaml \
  --stage joint --device cuda \
  --init-checkpoint outputs/jepa_planning/m4/intent.pt \
  --checkpoint-path outputs/jepa_planning/m4/joint.pt \
  2>&1 | tee outputs/jepa_planning/m4/joint.log
```

Acceptance gate:

- completed joint checkpoint reload được và cho seeded prediction giống nhau;
- flow loss và JEPA loss finite; target encoder/backbone vẫn frozen;
- sau zero-decoder warmup, flow gradient đến intent predictor và scene adapter;
- inference bắt đầu từ pure noise, không truyền future vào model;
- trajectory generated finite, heading wrapped và được visualize cùng GT.

Không đánh dấu M4 pass chỉ vì synthetic CPU lifecycle pass. Cần ba checkpoint,
ba log và visualization trên released data.

## 5. M5 — held-out validation và comparison

### M5.1 Đánh giá model có predicted intent

```bash
uv run --frozen python -m nureasoning.jepa_planning.evaluate metrics \
  --checkpoint outputs/jepa_planning/m4/joint.pt \
  --data-root ./dataset/data/validation \
  --device cuda --seeds 42,43,44 --batch-size 2 \
  --intent-mode predicted \
  --output outputs/jepa_planning/m5/predicted_intent.json
```

### M5.2 Intent-use diagnostics trên cùng checkpoint

```bash
for mode in no_intent shuffled; do
  uv run --frozen python -m nureasoning.jepa_planning.evaluate metrics \
    --checkpoint outputs/jepa_planning/m4/joint.pt \
    --data-root ./dataset/data/validation \
    --device cuda --seeds 42,43,44 --batch-size 2 \
    --intent-mode "$mode" \
    --output "outputs/jepa_planning/m5/${mode}.json"
done
```

`shuffled` cần batch ≥ 2 và chỉ dùng trong metrics mode. Benchmark callback xử
lý từng scene nên chỉ hỗ trợ `predicted` hoặc `no_intent`.

### M5.3 Existing planning benchmark/NPS

```bash
uv run --frozen python -m nureasoning.jepa_planning.evaluate benchmark \
  --checkpoint outputs/jepa_planning/m4/joint.pt \
  --data-root ./dataset/data/validation \
  --device cuda --seed 42 --intent-mode predicted \
  --output outputs/jepa_planning/m5/benchmark_predicted.json

uv run --frozen python -m nureasoning.jepa_planning.evaluate benchmark \
  --checkpoint outputs/jepa_planning/m4/joint.pt \
  --data-root ./dataset/data/validation \
  --device cuda --seed 42 --intent-mode no_intent \
  --output outputs/jepa_planning/m5/benchmark_no_intent.json
```

Báo cáo phải gồm ADE, FDE, wrapped heading error, NPS components, coverage,
failure reasons, latency và seed variability. So sánh trên cùng validation
split, checkpoint/data preprocessing, inference steps và seeds.

### M5.4 Khoảng trống còn phải implement trước khi claim baseline đầy đủ

`no_intent` hiện là inference ablation trên joint model đã train với intent. Nó
đo dependency của checkpoint lên intent nhưng chưa phải một LeVJEPA+DiT baseline
được train độc lập. Để hoàn tất comparison đúng nghĩa:

1. Thêm một explicit joint-training option `predicted`/`no_intent`; default vẫn
   bắt buộc là `predicted`.
2. Train một checkpoint `no_intent` từ cùng Stage-A artifact, cùng data, optimizer,
   DiT settings, step count và seeds; không feed target intent.
3. Đánh giá checkpoint baseline và intent checkpoint bằng cùng commands.
4. Không dùng `shuffled` khi training; đây chỉ là validation diagnostic.

Đây là engineering item còn dang dở, không được thay bằng việc chỉ bỏ intent ở
inference rồi gọi đó là baseline đã train.

## 6. Artifact và bảng kết quả bắt buộc

Thư mục kết quả đề nghị:

```text
outputs/jepa_planning/
  m3/
  m4/
    trajectory_ae.pt
    intent.pt
    joint.pt
    *.log
    visualizations/
  m5/
    predicted_intent.json
    no_intent.json
    shuffled.json
    benchmark_predicted.json
    benchmark_no_intent.json
    summary.md
```

`summary.md` tối thiểu phải có:

| Run | Training condition | ADE | FDE | Heading | NPS | Coverage | Latency | Seeds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Intent JEPA + DiT | predicted | TBD | TBD | TBD | TBD | TBD | TBD | 42/43/44 |
| Same checkpoint | no intent at eval | TBD | TBD | TBD | TBD | TBD | TBD | 42/43/44 |
| Same checkpoint | shuffled intent | TBD | TBD | TBD | n/a | TBD | TBD | 42/43/44 |
| Separately trained baseline | no intent train/eval | TBD | TBD | TBD | TBD | TBD | TBD | 42/43/44 |

Không chỉ báo loss. Mọi sample bị loại hoặc inference failure phải được tính vào
coverage, không silently drop.

## 7. Những việc chưa bắt buộc cho M3–M5 nhưng cần ghi backlog

- DDP và cross-rank InfoNCE negatives.
- Mid-epoch checkpoint/recovery; hiện deterministic resume được test ở epoch boundary.
- Driving-domain adaptation hoặc unfreeze LeVJEPA.
- Tuning camera coverage, token budget, frame cadence, loss weights và ODE steps.
- Feasibility/safety gate, scorer hoặc retrieval memory.
- QA/reasoning generation; nằm ngoài planning pipeline này.

Mọi thay đổi trong backlog làm thay đổi kiến trúc phải được tách thành thí nghiệm
riêng, không âm thầm trộn vào M3–M5 baseline.
