"""Instrument the gradient path: rasterizer -> deformed verts -> deformer params.

Loads the v16 canonical, runs ONE training step per deformer type at a fixed
timestamp, and prints gradient norms at each stage. Pinpoints where the
per-vertex MLP's signal dies (v5/v6/v16: zero motion learned at any LR).
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

from dts.deform import ControlNodeDeformer, DeformationMLP  # noqa: E402

parser = ArgumentParser()
lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
model_dir = os.environ.get("DTS_MODEL", "out/dts_jumpingjacks_v16")
args = parser.parse_args(["-s", "data/data/jumpingjacks", "-m", model_dir, "--eval"])
dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

triangles = TriangleModel(dataset.sh_degree)
scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma, load_iteration=-1, shuffle=False)
cams = scene.getTrainCameras()
attach_times(cams, dataset.source_path)
cam = next(c for c in cams if abs(c.time - 0.5) < 0.05)
bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

verts = triangles.get_vertices
print(f"verts {tuple(verts.shape)}, verts.requires_grad={verts.requires_grad}")
print(f"sigma={float(triangles.get_sigma):.4f}")

# baseline: gradient w.r.t. raw vertices through the rasterizer
v_leaf = verts.detach().clone().requires_grad_(True)
pkg = render_deformed(cam, triangles, pipe, bg, vertices_override=v_leaf)
loss = l1_loss(pkg["render"], cam.original_image.cuda())
loss.backward()
g = v_leaf.grad
print(f"\n[raw verts]   loss={float(loss):.4f}  dL/dverts: norm={float(g.norm()):.3e} "
      f"absmax={float(g.abs().max()):.3e}  nonzero={float((g.abs() > 0).float().mean()):.3f}")

for name, model in [
    ("mlp", DeformationMLP(hidden=256, depth=8, x_freqs=10, t_freqs=6).cuda()),
    ("nodes", ControlNodeDeformer(verts.detach(), num_nodes=512, knn=8).cuda()),
]:
    t = torch.tensor(cam.time, device="cuda")
    deformed = model(verts.detach(), t)
    pkg = render_deformed(cam, triangles, pipe, bg, vertices_override=deformed)
    loss = l1_loss(pkg["render"], cam.original_image.cuda())
    loss.backward()
    stats = {}
    for pn, p in model.named_parameters():
        if p.grad is not None:
            stats[pn] = float(p.grad.norm())
    top = sorted(stats.items(), key=lambda kv: -kv[1])[:3]
    total = sum(stats.values())
    last_layer = [v for k, v in stats.items() if "net" in k and (".6." in k or "-1" in k)]
    print(f"\n[{name}] loss={float(loss):.4f}  total param-grad norm={total:.3e}")
    for k, v in top:
        print(f"    {k}: {v:.3e}")
