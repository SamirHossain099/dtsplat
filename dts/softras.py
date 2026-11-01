"""Minimal differentiable soft triangle rasterizer (SoftRas-style aggregation).

Pure PyTorch, O(F·H·W) — intended for toy-scale validation of the deformation
stage, NOT for real scenes. Real scenes plug into the Triangle Splatting+
CUDA rasterizer (external/); this module exists so the deformation/ARAP
pipeline can be developed and unit-tested end-to-end without it.

Convention: OpenCV pinhole — camera looks down +z, x right, y down.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Camera:
    w2c: torch.Tensor  # (4, 4)
    fx: float
    fy: float
    cx: float
    cy: float
    height: int
    width: int

    @staticmethod
    def look_at(eye, target, up, fov_deg: float, height: int, width: int, device="cpu") -> "Camera":
        eye = torch.as_tensor(eye, dtype=torch.float32, device=device)
        target = torch.as_tensor(target, dtype=torch.float32, device=device)
        up = torch.as_tensor(up, dtype=torch.float32, device=device)

        fwd = torch.nn.functional.normalize(target - eye, dim=0)   # +z
        right = torch.nn.functional.normalize(torch.cross(fwd, up, dim=0), dim=0)  # +x
        down = torch.cross(fwd, right, dim=0)                      # +y (OpenCV y-down)

        R = torch.stack([right, down, fwd], dim=0)  # world -> cam rotation
        t = -R @ eye
        w2c = torch.eye(4, device=device)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        f = 0.5 * width / torch.tan(torch.deg2rad(torch.tensor(fov_deg / 2)))
        return Camera(w2c=w2c, fx=float(f), fy=float(f), cx=width / 2, cy=height / 2,
                      height=height, width=width)


def render(verts: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor, cam: Camera,
           sigma: float = 2e-2, background: float = 1.0, near: float = 0.05) -> torch.Tensor:
    """Render (3, H, W) image in [0, 1]. Differentiable w.r.t. verts and colors.

    sigma: edge softness in barycentric units. Faces are combined by
    depth-sorted alpha compositing (soft coverage acts as alpha), matching
    opaque-triangle splatting semantics — numerically robust, no exponentials.
    """
    device = verts.device
    H, W = cam.height, cam.width

    # world -> camera -> screen
    v_cam = (cam.w2c[:3, :3] @ verts.T).T + cam.w2c[:3, 3]
    z = v_cam[:, 2].clamp(min=near)
    x = cam.fx * v_cam[:, 0] / z + cam.cx
    y = cam.fy * v_cam[:, 1] / z + cam.cy
    v_scr = torch.stack([x, y], dim=1)  # (V, 2)

    fv = v_scr[faces]                   # (F, 3, 2)
    fz = z[faces]                       # (F, 3)
    fc = colors[faces]                  # (F, 3, 3)

    ys = torch.arange(H, device=device, dtype=verts.dtype) + 0.5
    xs = torch.arange(W, device=device, dtype=verts.dtype) + 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pix = torch.stack([gx, gy], dim=-1).reshape(-1, 2)  # (P, 2)

    # barycentrics of every pixel in every face: (F, P, 3)
    a, b, c = fv[:, 0], fv[:, 1], fv[:, 2]
    v0, v1 = b - a, c - a                       # (F, 2)
    vp = pix[None, :, :] - a[:, None, :]        # (F, P, 2)
    d00 = (v0 * v0).sum(-1)[:, None]
    d01 = (v0 * v1).sum(-1)[:, None]
    d11 = (v1 * v1).sum(-1)[:, None]
    d20 = (vp * v0[:, None, :]).sum(-1)
    d21 = (vp * v1[:, None, :]).sum(-1)
    denom = (d00 * d11 - d01 * d01).clamp(min=1e-12)
    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    w0 = 1.0 - w1 - w2
    bary = torch.stack([w0, w1, w2], dim=-1)    # (F, P, 3)

    # soft coverage from signed "insideness" (min barycentric, negative outside)
    inside = bary.min(dim=-1).values            # (F, P)
    cov = torch.sigmoid(inside / sigma)

    # interpolated color / depth with clamped barycentrics
    bary_cl = bary.clamp(0, 1)
    bary_cl = bary_cl / bary_cl.sum(-1, keepdim=True).clamp(min=1e-8)
    col = (bary_cl[..., None] * fc[:, None, :, :]).sum(dim=2)   # (F, P, 3)
    depth = (bary_cl * fz[:, None, :]).sum(-1)                  # (F, P)

    # depth-sorted alpha compositing (sort order carries no gradient, as in 3DGS)
    order = depth.argsort(dim=0)                                 # (F, P) near -> far
    alpha = cov.gather(0, order)                                 # (F, P)
    col_s = col.gather(0, order[..., None].expand(-1, -1, 3))    # (F, P, 3)
    trans = torch.cumprod(
        torch.cat([torch.ones(1, alpha.shape[1], device=device, dtype=alpha.dtype),
                   (1 - alpha).clamp(0, 1)], dim=0), dim=0)      # (F+1, P)
    weight = alpha * trans[:-1]                                  # (F, P)
    img = (weight[..., None] * col_s).sum(dim=0) + trans[-1][..., None] * background

    return img.reshape(H, W, 3).permute(2, 0, 1).clamp(0, 1)
