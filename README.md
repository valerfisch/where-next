# where-next

Pedestrian detection and trajectory prediction on the Forking Paths / NextX dataset.

## Overview

**where-next** predicts where people go next in surveillance video. Built in stages:

1. **Person detection** — CenterNet (anchor-free, objects as points)
2. **Trajectory prediction** — predict future positions from past tracks *(planned)*

## Architecture

CenterNet detects people as center points in a heatmap, then regresses bounding box size and offset — no anchors, no NMS.

| Component | Description |
|-----------|-------------|
| Backbone | ResNet-18 + 3 upconv layers (stride 4) |
| Heatmap head | `(1, H/4, W/4)` — Gaussian peaks at person centers |
| Size head | `(2, H/4, W/4)` — bbox width and height |
| Offset head | `(2, H/4, W/4)` — sub-pixel correction |

## Dataset

Uses the [NextX v1 dataset](https://next.cs.cmu.edu/multiverse/index.html) (Forking Paths):
- 3000 videos, 1920x1080, 30fps, 4 camera views per scene
- 7 data sources: `0000`, `0400`, `0401`, `zara01`, `eth`, `hotel`, `0500`
- Per-person bounding box annotations with track IDs
- Classes: Person, Vehicle (we filter to Person only)
- `is_x_agent` flag marks key pedestrians of interest

### Structure

```
dataset/next_x_v1_dataset/
├── bbox/           # 3000 JSON annotation files
├── rgb_videos/     # 3000 MP4 videos
└── seg_videos/     # 3000 segmentation mask videos
```

### Annotation format

Each bbox JSON contains an array of detections:
```json
{"class_name": "Person", "is_x_agent": 1, "bbox": [x, y, w, h], "frame_id": 0, "track_id": 303}
```

### Prepared dataset

After running `prepare_dataset.py`, per-video manifests are saved:
```json
{
  "video": "0000_0_303_0_1_cam1",
  "tracks": [
    {"id": 303, "bb": [[x, y, w, h, frame], ...]}
  ]
}
```
Frames are extracted at 5fps (every 6th frame) to `prepared_dataset/frames/{video_name}/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision opencv-python numpy tqdm
```

## Usage

### 1. Prepare dataset

```bash
# Extract frames (GPU-accelerated)
python scripts/prepare_dataset.py --input_dir ./dataset/next_x_v1_dataset --output_dir ./prepared_dataset

# Test with 10 videos first
python scripts/prepare_dataset.py --limit 10
```

### 2. Train CenterNet

```bash
python src/train.py  # coming soon
```

## Project structure

```
├── scripts/
│   └── prepare_dataset.py      # Frame extraction + per-video manifest generation
├── src/
│   ├── data/
│   │   └── centernet_dataset.py # CenterNet PyTorch dataset (WIP)
│   └── main.py                 # Entry point (placeholder)
├── paper/
│   └── 1904.07850v2.pdf        # CenterNet paper
├── TODO.md                     # Detailed implementation roadmap
└── README.md
```

## References

- Zhou et al., ["Objects as Points"](https://arxiv.org/abs/1904.07850) (CenterNet), arXiv:1904.07850
- Liang et al., ["The Garden of Forking Paths"](https://next.cs.cmu.edu/multiverse/index.html) (dataset)