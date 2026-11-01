"""Training losses: photometric (L1 + D-SSIM, 3DGS convention) and mesh regularizers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .mesh import TriMesh


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] @ g[None, :]


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM for (C, H, W) or (B, C, H, W) images in [0, 1]."""
    if img1.ndim == 3:
        img1, img2 = img1[None], img2[None]
    b, c, h, w = img1.shape
    win = _gaussian_window(window_size, sigma, img1.device, img1.dtype).expand(c, 1, window_size, window_size)
    pad = window_size // 2

    mu1 = F.conv2d(img1, win, padding=pad, groups=c)
    mu2 = F.conv2d(img2, win, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(img1 * img1, win, padding=pad, groups=c) - mu1_sq
    s2 = F.conv2d(img2 * img2, win, padding=pad, groups=c) - mu2_sq
    s12 = F.conv2d(img1 * img2, win, padding=pad, groups=c) - mu12

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean()


def photometric_loss(pred: torch.Tensor, target: torch.Tensor, lambda_dssim: float = 0.2) -> torch.Tensor:
    """(1 - λ) * L1 + λ * (1 - SSIM) / 2 on (C, H, W) images."""
    l1 = (pred - target).abs().mean()
    return (1 - lambda_dssim) * l1 + lambda_dssim * (1 - ssim(pred, target)) / 2


def edge_length_loss(mesh: TriMesh, deformed: torch.Tensor) -> torch.Tensor:
    """Relative squared change of edge lengths — cheap stretch penalty."""
    rest_len = mesh.edge_lengths()
    def_len = mesh.edge_lengths(deformed)
    return ((def_len - rest_len) / rest_len.clamp(min=1e-8)).pow(2).mean()


def edge_hinge_loss(mesh: TriMesh, deformed: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
    """Penalize only extreme stretch (needle triangles): zero cost below
    `threshold`x rest length, quadratic beyond. Articulation stretches edges
    <2x; LBS-boundary needles stretch 10-100x — this separates them, unlike
    the full edge loss which taxes all motion (~1.1 dB, v9/v13/v14)."""
    rest_len = mesh.edge_lengths()
    def_len = mesh.edge_lengths(deformed)
    ratio = def_len / rest_len.clamp(min=1e-8)
    return torch.relu(ratio - threshold).pow(2).mean()


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = (pred - target).pow(2).mean().clamp(min=1e-12)
    return float(10 * torch.log10(1.0 / mse))
