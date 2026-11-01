"""Minimal backward sanity test: can the TS+ rasterizer recover a translation?

Small triangle (their forward culls screen-large ones), TS+-native camera
matrices, activated weights in [0,1]. Target = same triangle shifted. If Adam
on vertex positions cannot recover the shift, the CUDA backward w.r.t.
positions is broken (upstream's core contribution is exactly this gradient).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts.tsplus_bridge import setup_tsplus_paths  # noqa: E402

setup_tsplus_paths()

import math  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from diff_triangle_rasterization import (TriangleRasterizationSettings,  # noqa: E402
                                         TriangleRasterizer)
from utils.graphics_utils import getProjectionMatrix, getWorld2View2  # noqa: E402

H = W = 256
fov = math.radians(45.0)
R = np.eye(3)
T = np.zeros(3)
world_view = torch.tensor(getWorld2View2(R, T, np.zeros(3), 1.0)).transpose(0, 1).cuda()
proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov, fovY=fov).transpose(0, 1).cuda()
full_proj = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
campos = world_view.inverse()[3, :3]

settings = TriangleRasterizationSettings(
    image_height=H, image_width=W,
    tanfovx=math.tan(fov / 2), tanfovy=math.tan(fov / 2),
    bg=torch.zeros(3, device="cuda"),
    scale_modifier=1.0, viewmatrix=world_view, projmatrix=full_proj,
    sh_degree=0, campos=campos, prefiltered=False, debug=False)
rast = TriangleRasterizer(raster_settings=settings)

# ~30 px triangle at z=3 (their forward culls triangles with edge^2 > 1600 px)
tri = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
verts_gt = torch.tensor([[-0.12, -0.10, 3.0], [0.14, -0.08, 3.0], [0.0, 0.16, 3.0]],
                        device="cuda")
weights = torch.full((3,), 0.999, device="cuda")   # activated opacity in [0,1]
# distinct per-vertex colors activate the barycentric-interpolation gradient
# path in the backward (identical colors make it cancel exactly)
vcolors = torch.tensor([[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]], device="cuda")


def render(v, sigma):
    scaling = torch.zeros(1, device="cuda")
    img, radii, scal, allmap, maxb = rast(
        vertices=v, triangles_indices=tri, vertex_weights=weights,
        sigma=sigma, shs=None, colors_precomp=vcolors, scaling=scaling)
    return img


for sigma in (0.3, 0.05, 0.005):
    with torch.no_grad():
        target = render(verts_gt, sigma)
    cover = float((target.sum(0) > 0.01).float().mean())
    print(f"\n=== sigma={sigma}  target: max={float(target.max()):.3f} coverage={cover:.4f}")
    if float(target.max()) < 0.01:
        print("  triangle not rendered — check culling/conventions")
        continue

    # finite-difference validation of the analytic gradient at the offset pose
    v0 = (verts_gt + torch.tensor([0.05, 0.03, 0.0], device="cuda")).clone()
    v_leaf = v0.clone().requires_grad_(True)
    l0 = (render(v_leaf, sigma) - target).abs().mean()
    l0.backward()
    g = v_leaf.grad.detach()
    torch.manual_seed(0)
    for i in range(3):
        d = torch.randn_like(v0)
        d = d / d.norm()
        h = 1e-3
        with torch.no_grad():
            lp = (render(v0 + h * d, sigma) - target).abs().mean()
            lm = (render(v0 - h * d, sigma) - target).abs().mean()
        fd = float((lp - lm) / (2 * h))
        an = float((g * d).sum())
        print(f"  fd-check dir{i}: analytic={an:+.3e}  fd={fd:+.3e}  "
              f"ratio={fd / an if abs(an) > 1e-15 else float('inf'):+.2f}")

    # gradient direction quality: cos(g, v - v_gt) should be strongly positive
    true_dir = (v0 - verts_gt)
    cos = float((g * true_dir).sum() / (g.norm() * true_dir.norm() + 1e-12))
    gz = float(g[:, 2].abs().sum() / (g.abs().sum() + 1e-12))
    print(f"  grad quality: cos(g, offset-dir)={cos:+.3f}   z-fraction of |g|={gz:.3f}")

    for zfree, tag in ((True, "xy-only"), (False, "xyz")):
        verts = v0.clone().requires_grad_(True)
        opt = torch.optim.Adam([verts], lr=2e-3)
        init_err = float((verts - verts_gt).norm().detach())
        traj = []
        for it in range(1, 1501):
            img = render(verts, sigma)
            loss = (img - target).abs().mean()
            opt.zero_grad()
            loss.backward()
            if zfree:
                verts.grad[:, 2] = 0
            opt.step()
            if it % 300 == 0:
                traj.append(float((verts - verts_gt).norm().detach()))
        print(f"  [{tag}] err {init_err:.4f} -> " + " -> ".join(f"{e:.4f}" for e in traj)
              + ("   RECOVERED" if traj[-1] < 0.2 * init_err else "   FAILED"))
