# Dataset setup

The nuReasoning dataset is hosted on Hugging Face: https://huggingface.co/datasets/nureasoning/nuReasoning

Access is subject to review and compliance requirements; data is released progressively.

## Download

Log in with your Hugging Face access token, then run the download module:

```bash
hf auth login

cd nureasoning-devkit
python -m nureasoning.dataset.download --local-dir ./dataset --max-workers 16
```

Each clip arrives as a `.tar` and is extracted in place under `data/`. The archive is deleted after a successful extract. Pass `--keep-tars` if you want to keep the archives.

The dataset repository is private. Request access on the Hugging Face page, then log in before downloading.

A full dump is large. For a first look, download one split or one part:

```bash
# One split (all parts)
python -m nureasoning.dataset.download --local-dir ./dataset --splits validation

# One part of one split (`1` and `part_1` are equivalent)
python -m nureasoning.dataset.download --local-dir ./dataset --splits train --parts part_1

# Several splits or parts
python -m nureasoning.dataset.download --local-dir ./dataset --splits train validation --parts part_1 part_2
```

`--splits` is one or more of `train`, `validation`, `test`. `--parts` without
`--splits` takes that part from every split.

Other options:

```bash
# Re-extract archives without re-downloading
python -m nureasoning.dataset.download --local-dir ./dataset --extract-only
```

## Resulting layout

Clips are self-contained directories (archives are gone unless you passed `--keep-tars`):

```
dataset/
└── data/
    └── <split>/[part_<k>/]<clip_name>/
        ├── metadata.json                   - clip-level metadata + per-frame index
        ├── map.pkl                         - static HD map around the clip
        ├── ego_state/<timestamp_us>.pkl    - ego pose, velocity, history/future trajectory
        ├── annotations/<timestamp_us>.pkl  - object annotations + traffic-light states
        ├── reasoning/<timestamp_us>.json   - Spatial / Decision / Counterfactual reasoning
        └── ... sensor assets referenced by metadata.json (8 cameras + LiDAR)
```

Train and validation clips live under `part_<k>/`. Test clips are stored
directly under `data/test/` and ship **without** ground truth: cameras, ego
state through the key frame, and `reasoning_questions.json` only. That is the
`--data-root` for [challenge submission](submission.md).

## Visualize

```bash
# Composite 8-camera + ego-state + BEV (+ LiDAR) overview PNG
python -m nureasoning.visualization.view_data \
  --input-root ./dataset/data/train/part_1 --frame-index 100

# Same layout as an MP4
python -m nureasoning.visualization.view_data \
  --input-root ./dataset/data/train/part_1 --video --max-frames 50

# Driving / Counterfactual text above left/front/right cameras at the key frame
python -m nureasoning.visualization.view_reasoning \
  --input-root ./dataset/data/train/part_1 --frame-index 100 --save-figures

# History → paused reasoning frame (with 2D boxes) → future
python -m nureasoning.visualization.view_reasoning \
  --input-root ./dataset/data/train/part_1 --video
```
