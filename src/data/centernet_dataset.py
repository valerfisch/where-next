import json
import cv2
import torch

import torchvision.transforms as transforms

from dataclasses import dataclass
from math import floor, sqrt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset


@dataclass
class BoundingBox:
    agent_id: int
    x: int
    y: int
    width: int
    height: int


@dataclass
class CenterNetEntry:
    video_id: str
    frame: int
    has_future: bool
    bbs: list[BoundingBox]


@dataclass
class CenterNetSample:
    img: torch.Tensor
    gt_seg: torch.Tensor
    gt_heat: torch.Tensor


def parse_json(json_file):
    with open(json_file) as f:
        d = json.load(f)

    if "bbs" not in d:
        return None

    bbs = [
        BoundingBox(
            agent_id=bb["agent_id"],
            x=bb["x"],
            y=bb["y"],
            width=bb["width"],
            height=bb["height"],
        )
        for bb in d["bbs"]
    ]

    return CenterNetEntry(
        video_id=d["video_id"],
        frame=d["frame"],
        has_future=d["has_future"],
        bbs=bbs,
    )


class CenterNetDataset(Dataset):
    def __init__(self, data_dir, mode="debug"):
        self.data_dir = Path(data_dir)
        self.image_transform = transforms.ToTensor()

        json_files = sorted(
            f.name for f in (self.data_dir / "manifests").glob("*.json")
        )

        if mode == "debug":
            json_files = json_files[:50]

        data = [
            parse_json(self.data_dir / "manifests" / json_file)
            for json_file in tqdm(json_files)
        ]
        self.data = [d for d in data if d is not None]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]

        img = cv2.imread(
            str(self.data_dir / "frames" / d.video_id / f"{d.frame:06d}.jpg")
        )
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.image_transform(img_rgb)

        h, w = image.shape[1], image.shape[2]

        gt_seg = torch.zeros(h, w)
        gt_heat = torch.zeros(h, w)

        for bb in d.bbs:
            left = max(0, int(bb.x - 0.5 * bb.width))
            right = min(w, int(bb.x + 0.5 * bb.width))
            top = max(0, int(bb.y - 0.5 * bb.height))
            bottom = min(h, int(bb.y + 0.5 * bb.height))

            gt_seg[top:bottom, left:right] = bb.agent_id

            self._splat_gaussian(gt_heat, int(bb.x), int(bb.y), bb.width, bb.height)

        return CenterNetSample(gt_seg=gt_seg, gt_heat=gt_heat, img=image)

    def _gaussian_radius(self, bh, bw, iou=0.3):
        candidates = []
        d1 = (bh + bw) ** 2 - 4 * bh * bw * (1 - iou) / iou
        if d1 >= 0:
            candidates.append((bh + bw - sqrt(d1)) / 4)
        d2 = (bh + bw) ** 2 - 4 * (1 - iou) * bh * bw
        if d2 >= 0:
            candidates.append((bh + bw - sqrt(d2)) / 4)
        candidates.append(sqrt(iou * bw * bh / (4 * (1 - iou) + iou)))
        return max(1, floor(min(candidates)))

    def _splat_gaussian(self, heatmap, cx, cy, bw, bh):
        h, w = heatmap.shape
        r = self._gaussian_radius(bh, bw)
        sigma = (2 * r + 1) / 6

        g = torch.arange(-r, r + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(g, g, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        y1, y2 = max(0, cy - r), min(h, cy + r + 1)
        x1, x2 = max(0, cx - r), min(w, cx + r + 1)
        ky1, kx1 = r - (cy - y1), r - (cx - x1)

        heatmap[y1:y2, x1:x2] = torch.max(
            heatmap[y1:y2, x1:x2],
            kernel[ky1 : ky1 + y2 - y1, kx1 : kx1 + x2 - x1],
        )
