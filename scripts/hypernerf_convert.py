"""Convert a HyperNeRF scene to the D-NeRF/Blender transforms format TS+ reads.

HyperNeRF layout: rgb/<scale>x/<id>.png, camera/<id>.json, dataset.json
(train/val ids), metadata.json (time_id/warp_id per frame).
Camera json: orientation (world-to-camera R, row-major), position (world),
focal_length, principal_point, image_size [W, H] at scale 1x.

Output: transforms_train.json / transforms_test.json with c2w matrices
(OpenGL/Blender convention: y up, z back — matching readCamerasFromTransforms,
which flips y/z to COLMAP), per-frame `time` in [0, 1], plus rescaled images
copied into train/ and test/ dirs as PNG (no alpha).

Usage: python scripts/hypernerf_convert.py data/hypernerf/vrig_3dprinter [--scale 4]
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def main():
    ap = ArgumentParser()
    ap.add_argument("root", type=str)
    ap.add_argument("--scale", type=int, default=4, help="use rgb/<scale>x images")
    args = ap.parse_args()
    root = Path(args.root)

    dataset = json.loads((root / "dataset.json").read_text())
    meta = json.loads((root / "metadata.json").read_text())
    # normalize world into the ~unit frame TS+'s Blender init assumes
    scene_center = np.zeros(3)
    scene_scale = 1.0
    scene_json = root / "scene.json"
    if scene_json.exists():
        sj = json.loads(scene_json.read_text())
        scene_center = np.array(sj.get("center", [0, 0, 0]), dtype=np.float64)
        scene_scale = float(sj.get("scale", 1.0))
        print(f"scene.json: center={scene_center.tolist()}, scale={scene_scale}")
    all_ids = dataset["ids"]
    train_ids = set(dataset.get("train_ids", all_ids))
    val_ids = set(dataset.get("val_ids", []))

    # normalize time from metadata time_id
    time_ids = {i: meta[i].get("time_id", meta[i].get("warp_id", 0)) for i in all_ids}
    tmax = max(time_ids.values()) or 1

    rgb_dir = root / "rgb" / f"{args.scale}x"
    assert rgb_dir.exists(), f"missing {rgb_dir}"

    def frame_entry(fid: str, out_rel: str):
        cam = json.loads((root / "camera" / f"{fid}.json").read_text())
        R_w2c = np.array(cam["orientation"], dtype=np.float64)  # (3,3)
        pos = np.array(cam["position"], dtype=np.float64)       # camera center, world
        pos = (pos - scene_center) * scene_scale                 # nerfies normalization
        # c2w with OpenCV axes (x right, y down, z forward): columns = camera axes
        c2w = np.eye(4)
        c2w[:3, :3] = R_w2c.T
        c2w[:3, 3] = pos
        # OpenCV -> Blender/OpenGL (y up, z back): flip y and z columns.
        # readCamerasFromTransforms will flip them back.
        c2w[:3, 1:3] *= -1
        focal = cam["focal_length"] / args.scale
        W = cam["image_size"][0] / args.scale
        angle_x = 2 * math.atan(W / (2 * focal))
        return {
            "file_path": out_rel,
            "transform_matrix": c2w.tolist(),
            "time": time_ids[fid] / tmax,
        }, angle_x

    out = {"train": [], "test": []}
    angle_x_final = None
    for fid in all_ids:
        split = "train" if fid in train_ids else ("test" if fid in val_ids else None)
        if split is None:
            continue
        src = rgb_dir / f"{fid}.png"
        if not src.exists():
            continue
        dst_dir = root / split
        dst_dir.mkdir(exist_ok=True)
        shutil.copyfile(src, dst_dir / f"{fid}.png")
        entry, angle_x = frame_entry(fid, f"./{split}/{fid}")
        angle_x_final = angle_x
        out[split].append(entry)

    for split in ("train", "test"):
        (root / f"transforms_{split}.json").write_text(json.dumps({
            "camera_angle_x": angle_x_final,
            "frames": out[split],
        }, indent=1))
        print(f"{split}: {len(out[split])} frames")
    print(f"camera_angle_x = {angle_x_final:.4f}; images at scale 1/{args.scale}")


if __name__ == "__main__":
    main()
