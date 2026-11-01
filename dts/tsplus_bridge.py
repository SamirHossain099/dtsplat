"""Bridge to the Triangle Splatting+ codebase (external/triangle-splatting-plus).

Provides:
- import path setup for the TS+ modules and compiled extensions
- render_deformed(): TS+'s render() with a vertex override, so a deformation
  model can move vertices before rasterization (the rasterizer is already
  differentiable w.r.t. vertices — no CUDA changes needed)
- attach_times(): map D-NeRF per-frame `time` onto TS+ Camera objects
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

TS_ROOT = Path(__file__).resolve().parents[1] / "external" / "triangle-splatting-plus"


def setup_tsplus_paths() -> None:
    for p in (str(TS_ROOT), str(TS_ROOT / "triangulation")):
        if p not in sys.path:
            sys.path.insert(0, p)


setup_tsplus_paths()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from diff_triangle_rasterization import (TriangleRasterizationSettings,  # noqa: E402
                                         TriangleRasterizer)


def render_deformed(viewpoint_camera, pc, pipe, bg_color: torch.Tensor,
                    vertices_override: torch.Tensor | None = None):
    """TS+ triangle_renderer.render(), with optional deformed vertices.

    Kept in sync with external/triangle-splatting-plus/triangle_renderer/__init__.py;
    the only functional change is `vertices_override`.
    """
    triangles_indices = pc.get_triangle_indices
    vertices = pc.get_vertices if vertices_override is None else vertices_override
    vertex_weights = pc.get_vertex_weight
    scaling = torch.zeros_like(triangles_indices[:, 0], dtype=pc.get_triangles_points.dtype,
                               requires_grad=True, device="cuda").detach()

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    H_init = int(viewpoint_camera.image_height)
    W_init = int(viewpoint_camera.image_width)
    upsample = pc.scaling
    H, W = upsample * H_init, upsample * W_init

    raster_settings = TriangleRasterizationSettings(
        image_height=H, image_width=W, tanfovx=tanfovx, tanfovy=tanfovy,
        bg=bg_color, scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree, campos=viewpoint_camera.camera_center,
        prefiltered=False, debug=pipe.debug)
    rasterizer = TriangleRasterizer(raster_settings=raster_settings)

    rendered_image, radii, scaling, allmap, max_blending = rasterizer(
        vertices=vertices,
        triangles_indices=triangles_indices,
        vertex_weights=vertex_weights.squeeze(),
        sigma=pc.get_sigma,
        shs=pc.get_features,
        colors_precomp=None,
        scaling=scaling)

    def down(x):
        return F.interpolate(x.unsqueeze(0), size=(H_init, W_init), mode="area").squeeze(0)

    return {
        "render": down(rendered_image),
        "visibility_filter": radii > 0,
        "radii": radii,
        "scaling": scaling,
        "max_blending": max_blending,
        "rend_alpha": down(allmap[1:2]),
        "rend_normal": down(allmap[2:5]),
        "surf_depth": down(allmap[5:6]),
        "entropy": allmap[0:1],
    }


def attach_times(cameras, scene_path: str | os.PathLike, split_files=("transforms_train.json",
                                                                      "transforms_test.json",
                                                                      "transforms_val.json"),
                 load_alpha: bool = False) -> None:
    """Set cam.time (and optionally cam.gt_alpha) on TS+ Camera objects from
    D-NeRF transforms json, matched by image name.

    gt_alpha enables random-background training: junk triangles that hide by
    matching a fixed background color can't match a random one, so the
    photometric loss forces them transparent and pruning removes them.
    """
    name_to_frame: dict[str, tuple[float, Path]] = {}
    for tf in split_files:
        p = Path(scene_path) / tf
        if not p.exists():
            continue
        meta = json.loads(p.read_text())
        for fr in meta["frames"]:
            stem = Path(fr["file_path"]).stem
            img_path = Path(scene_path) / (fr["file_path"].lstrip("./") + ".png")
            name_to_frame[stem] = (float(fr.get("time", 0.0)), img_path)

    import numpy as np
    from PIL import Image

    missing = 0
    for cam in cameras:
        entry = name_to_frame.get(cam.image_name)
        if entry is None:
            missing += 1
            cam.time = 0.0
            continue
        cam.time = entry[0]
        if load_alpha:
            im = np.asarray(Image.open(entry[1]), dtype=np.float32) / 255.0
            a = im[..., 3:4] if im.shape[-1] == 4 else np.ones_like(im[..., :1])
            cam.gt_alpha = torch.from_numpy(a).permute(2, 0, 1)  # (1, H, W), CPU
    if missing:
        print(f"[attach_times] WARNING: {missing}/{len(cameras)} cameras had no time entry")
