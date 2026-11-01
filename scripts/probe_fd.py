"""Finite-difference check of the TS+ rasterizer's vertex-position backward.

Compares analytic directional derivative (dL/dverts . delta) against
(L(verts + h*delta) - L(verts - h*delta)) / 2h for random directions.
If they disagree materially, the CUDA backward for vertices is wrong/partial.
"""
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts.tsplus_bridge import attach_times, render_deformed  # noqa: E402

import torch  # noqa: E402
from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from scene import Scene, TriangleModel  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402

parser = ArgumentParser()
lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
model_dir = os.environ.get("DTS_MODEL", "out/dts_jumpingjacks_v16")
args = parser.parse_args(["-s", "data/data/jumpingjacks", "-m", model_dir, "--eval"])
dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

triangles = TriangleModel(dataset.sh_degree)
scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma, load_iteration=-1, shuffle=False)
cams = scene.getTrainCameras()
attach_times(cams, dataset.source_path)
cam = cams[0]
bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
gt = cam.original_image.cuda()

verts0 = triangles.get_vertices.detach()


def loss_at(v):
    with torch.no_grad():
        pkg = render_deformed(cam, triangles, pipe, bg, vertices_override=v)
        return float(l1_loss(pkg["render"], gt))


# analytic gradient
v_leaf = verts0.clone().requires_grad_(True)
pkg = render_deformed(cam, triangles, pipe, bg, vertices_override=v_leaf)
loss = l1_loss(pkg["render"], gt)
loss.backward()
g = v_leaf.grad.detach()
print(f"loss={float(loss):.6f}  |g|={float(g.norm()):.3e}")

torch.manual_seed(0)
for h in (1e-2, 1e-3):
    print(f"\nh={h}")
    for i in range(3):
        delta = torch.randn_like(verts0)
        delta = delta / delta.norm()
        analytic = float((g * delta).sum())
        fd = (loss_at(verts0 + h * delta) - loss_at(verts0 - h * delta)) / (2 * h)
        print(f"  dir{i}: analytic={analytic:+.3e}  finite-diff={fd:+.3e}  "
              f"ratio={fd / analytic if abs(analytic) > 1e-12 else float('inf'):+.2f}")
