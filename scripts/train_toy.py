"""End-to-end toy experiment: recover a known deformation from images only.

A checkered sheet waves over time (GT deformation). We render target images
from a few cameras at several timesteps, then optimize a deformation model
(per-vertex MLP or control nodes) from photometric loss + ARAP + edge
regularization — never seeing the GT geometry. Success = low vertex error at
held-out times, proving the Stage-2 optimization loop works before touching
real data or the TS+ CUDA rasterizer.

Run:  python scripts/train_toy.py --model nodes --iters 1500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dts.arap import arap_energy
from dts.deform import ControlNodeDeformer, DeformationMLP
from dts.losses import edge_length_loss, photometric_loss, psnr
from dts.mesh import TriMesh
from dts.softras import Camera, render


def gt_deform(verts: torch.Tensor, t: float) -> torch.Tensor:
    """Waving-flag deformation: out-of-plane sine wave traveling over time."""
    x, y = verts[:, 0], verts[:, 1]
    z = 0.18 * torch.sin(2 * torch.pi * (x + float(t))) * (0.5 + 0.5 * (y + 0.5))
    return torch.stack([x, y, z], dim=1)


def make_cameras(im: int, device) -> list[Camera]:
    eyes = [(0.0, 0.0, 2.2), (1.2, 0.6, 1.9), (-1.2, -0.4, 1.9)]
    return [Camera.look_at(e, (0, 0, 0), (0, 1, 0), fov_deg=45, height=im, width=im, device=device)
            for e in eyes]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "nodes"], default="nodes")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--im", type=int, default=64)
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--times", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--w-arap", type=float, default=0.2)
    ap.add_argument("--w-edge", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out or f"out/toy_{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    mesh = TriMesh.make_grid(args.grid, args.grid, size=1.4, device=device)
    cams = make_cameras(args.im, device)
    train_times = torch.linspace(0, 1, args.times, device=device)
    held_out = torch.tensor([0.24, 0.61, 0.87], device=device)  # not in train set

    # render GT target images (no grad)
    with torch.no_grad():
        targets = {}
        for ti, t in enumerate(train_times):
            dv = gt_deform(mesh.verts, float(t))
            for ci, cam in enumerate(cams):
                targets[(ti, ci)] = render(dv, mesh.faces, mesh.colors, cam)

    if args.model == "mlp":
        model = DeformationMLP().to(device)
    else:
        model = ControlNodeDeformer(mesh.verts, num_nodes=48, knn=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=args.lr * 0.02)

    t0 = time.time()
    for it in range(1, args.iters + 1):
        ti = int(torch.randint(args.times, (1,)))
        t = train_times[ti]

        deformed = model(mesh.verts, t)
        # all cameras per step: 3x render cost, far lower gradient variance
        loss = torch.stack([
            photometric_loss(render(deformed, mesh.faces, mesh.colors, cam), targets[(ti, ci)])
            for ci, cam in enumerate(cams)
        ]).mean()
        loss = loss + args.w_arap * arap_energy(mesh, deformed)
        loss = loss + args.w_edge * edge_length_loss(mesh, deformed)


        opt.zero_grad(set_to_none=True)
        loss.backward()
        # edge-on triangles can emit huge barycentric gradients; one unclipped
        # step throws the sheet off-screen and ARAP (rigid-invariant) can't recover it
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if it % 200 == 0 or it == 1:
            with torch.no_grad():
                verr = torch.stack([
                    (model(mesh.verts, t) - gt_deform(mesh.verts, float(t))).norm(dim=1).mean()
                    for t in held_out
                ]).mean()
                p = psnr(render(deformed.detach(), mesh.faces, mesh.colors, cams[0]), targets[(ti, 0)])
            print(f"iter {it:5d}  loss {float(loss.detach()):.4f}  train-PSNR {p:5.2f} dB  "
                  f"held-out vert err {float(verr):.4f}  ({time.time() - t0:.0f}s)")

    # final report + qualitative dump
    from PIL import Image
    import numpy as np

    with torch.no_grad():
        final_err, final_psnr = [], []
        for k, t in enumerate(held_out):
            dv_pred = model(mesh.verts, t)
            dv_gt = gt_deform(mesh.verts, float(t))
            final_err.append(float((dv_pred - dv_gt).norm(dim=1).mean()))
            img_p = render(dv_pred, mesh.faces, mesh.colors, cams[0])
            img_g = render(dv_gt, mesh.faces, mesh.colors, cams[0])
            final_psnr.append(psnr(img_p, img_g))
            side = torch.cat([img_g, img_p], dim=2).permute(1, 2, 0).cpu().numpy()
            Image.fromarray((side * 255).astype(np.uint8)).save(out_dir / f"heldout_{k}_gt_vs_pred.png")

    diag = float(mesh.verts.max(0).values.sub(mesh.verts.min(0).values).norm())
    err = sum(final_err) / len(final_err)
    print("\n== held-out times (never trained on) ==")
    print(f"mean vertex error: {err:.4f}  ({100 * err / diag:.2f}% of mesh diagonal)")
    print(f"mean render PSNR vs GT: {sum(final_psnr) / len(final_psnr):.2f} dB")
    print(f"comparisons saved to {out_dir}")


if __name__ == "__main__":
    main()
