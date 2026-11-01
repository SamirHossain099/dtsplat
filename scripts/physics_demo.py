"""Stage-3 physics demo: simulate the reconstruction DIRECTLY — zero conversion.

Loads a trained DTS checkpoint, treats its triangle vertices as particles and
its true mesh edges as springs (rest lengths from the reconstruction), drops
the subject onto a ground plane with a GPU mass-spring integrator, and renders
every simulated frame through the same TS+ rasterizer.

Contrast: mesh-Gaussian hybrids (arXiv 2606.00444) must extract a proxy mesh
and synchronize two representations; here the reconstruction IS the sim mesh.

Run:  python scripts/physics_demo.py   (env DTS_MODEL to pick checkpoint)
"""
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts.tsplus_bridge import attach_times, render_deformed  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from scene import Scene, TriangleModel  # noqa: E402

from dts.mesh import TriMesh  # noqa: E402

# ---- load reconstruction ----
parser = ArgumentParser()
lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
model_dir = os.environ.get("DTS_MODEL", "out/dts_jumpingjacks_v17")
scene_dir = os.environ.get("DTS_SCENE", "data/data/jumpingjacks")
args = parser.parse_args(["-s", scene_dir, "-m", model_dir, "--eval"])
dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

triangles = TriangleModel(dataset.sh_degree)
scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma, load_iteration=-1, shuffle=False)
cams = scene.getTestCameras()
attach_times(cams, dataset.source_path)
cam = cams[0]
bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

x = triangles.get_vertices.detach().clone()          # particles = reconstruction vertices
faces = triangles.get_triangle_indices.detach()
mesh = TriMesh(x.clone(), faces)
edges = mesh.edges()                                  # true mesh springs
rest = (x[edges[:, 0]] - x[edges[:, 1]]).norm(dim=1).clamp(min=1e-6)
print(f"sim mesh: {x.shape[0]} particles, {edges.shape[0]} springs, {faces.shape[0]} triangles"
      " (taken from the reconstruction as-is)")

# ---- world setup: detect up axis (largest subject extent) and floor ----
extent = x.max(0).values - x.min(0).values
up_axis = int(extent.argmax())
floor = float(x[:, up_axis].min())
print(f"up axis = {'xyz'[up_axis]}, floor at {floor:.3f}")

# ---- shape-matching soft body (Mueller et al. 2005) ----
# Pure edge-springs collapse a surface-shell triangle soup; shape matching
# blends particles toward the best-fit rigid transform of the rest pose each
# step — soft but structurally stable, and robust to soup connectivity.
rest_x = x.clone()
cm_rest = rest_x.mean(0)
q_rest = rest_x - cm_rest
v = torch.zeros_like(x)
x[:, up_axis] += 0.45                       # lift, then drop
v[:, (up_axis + 1) % 3] = 0.35              # slight sideways toss for visual interest
gravity = torch.zeros(3, device="cuda")
gravity[up_axis] = -4.0
STIFF, AIR = 0.55, 0.35                     # shape-matching stiffness / air drag
RESTITUTION, FRICTION = 0.25, 0.80
DT, SUBSTEPS, FRAMES = 1.0 / 240.0, 6, 110

out_dir = Path(model_dir) / "physics_demo"
out_dir.mkdir(exist_ok=True)

t0 = time.time()
frames = []
for f in range(FRAMES):
    for _ in range(SUBSTEPS):
        v = v + DT * gravity - DT * AIR * v
        x_pred = x + DT * v
        # best-fit rotation of rest shape onto predicted positions
        cm = x_pred.mean(0)
        p = x_pred - cm
        cov = q_rest.T @ p                                    # (3,3)
        U, _, Vh = torch.linalg.svd(cov)
        R = (Vh.transpose(0, 1) @ U.transpose(0, 1))
        if torch.linalg.det(R) < 0:
            Vh2 = Vh.clone()
            Vh2[2] *= -1
            R = (Vh2.transpose(0, 1) @ U.transpose(0, 1))
        goal = q_rest @ R.T + cm
        x_new = x_pred + STIFF * (goal - x_pred)
        v = (x_new - x) / DT
        x = x_new
        below = x[:, up_axis] < floor                          # ground collision
        if below.any():
            x[below, up_axis] = floor
            vn = v[below, up_axis]
            v[below, up_axis] = -RESTITUTION * vn.clamp(max=0)
            for a in [a for a in range(3) if a != up_axis]:
                v[below, a] *= FRICTION
    with torch.no_grad():
        img = render_deformed(cam, triangles, pipe, bg, vertices_override=x)["render"]
    frames.append((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

elapsed = time.time() - t0
print(f"{FRAMES} frames ({FRAMES * SUBSTEPS} sim steps + renders) in {elapsed:.1f}s "
      f"= {FRAMES / elapsed:.1f} fps end-to-end on one RTX 5080")

imgs = [Image.fromarray(fr) for fr in frames]
imgs[0].save(out_dir / "drop.gif", save_all=True, append_images=imgs[1:],
             duration=33, loop=0)
for i in (0, FRAMES // 3, 2 * FRAMES // 3, FRAMES - 1):
    Image.fromarray(frames[i]).save(out_dir / f"frame_{i:03d}.png")
print(f"saved {out_dir}\\drop.gif and keyframes")
