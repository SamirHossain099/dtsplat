"""D-NeRF dataset loader (transforms_{split}.json + per-frame `time` field).

Dataset: https://github.com/albertpumarola/D-NeRF (8 synthetic monocular
dynamic scenes: bouncingballs, hellwarrior, hook, jumpingjacks, lego, mutant,
standup, trex). Download into data/dnerf/<scene>/.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class DNeRFFrame:
    image_path: Path
    c2w: torch.Tensor      # (4, 4), OpenGL/Blender convention (-z forward)
    time: float            # in [0, 1]


class DNeRFDataset:
    def __init__(self, root: str | Path, split: str = "train", white_background: bool = True):
        self.root = Path(root)
        self.white_background = white_background
        meta = json.loads((self.root / f"transforms_{split}.json").read_text())
        self.camera_angle_x = float(meta["camera_angle_x"])
        self.frames: list[DNeRFFrame] = []
        for fr in meta["frames"]:
            p = self.root / (fr["file_path"].lstrip("./") + ".png")
            c2w = torch.tensor(fr["transform_matrix"], dtype=torch.float32)
            self.frames.append(DNeRFFrame(p, c2w, float(fr.get("time", 0.0))))

    def __len__(self) -> int:
        return len(self.frames)

    def intrinsics(self, width: int, height: int) -> tuple[float, float, float, float]:
        fx = 0.5 * width / math.tan(0.5 * self.camera_angle_x)
        return fx, fx, width / 2, height / 2

    def load_image(self, i: int) -> torch.Tensor:
        """(3, H, W) float in [0, 1], alpha composited onto white/black."""
        from PIL import Image

        img = np.asarray(Image.open(self.frames[i].image_path), dtype=np.float32) / 255.0
        rgb, alpha = img[..., :3], img[..., 3:4] if img.shape[-1] == 4 else np.ones_like(img[..., :1])
        bg = 1.0 if self.white_background else 0.0
        rgb = rgb * alpha + bg * (1 - alpha)
        return torch.from_numpy(rgb).permute(2, 0, 1)

    def opencv_w2c(self, i: int) -> torch.Tensor:
        """World-to-camera in OpenCV convention (+z forward, y down) for our rasterizer."""
        c2w = self.frames[i].c2w.clone()
        # Blender/OpenGL -> OpenCV: flip y and z camera axes
        c2w[:3, 1:3] *= -1
        return torch.linalg.inv(c2w)
