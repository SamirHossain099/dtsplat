"""Exact As-Rigid-As-Possible energy (Sorkine & Alexa 2007) on the true triangle graph.

This is the core differentiator vs. Gaussian methods like SC-GS, which can only
approximate rigidity over an unstructured point soup: here the one-ring
neighborhoods come from real shared-vertex connectivity.

Local step (per-vertex rotation fit via SVD) is done under no_grad — the
standard local/global treatment — so gradients flow through the deformed
positions only, which keeps the SVD out of the backward pass and the
optimization stable.
"""

from __future__ import annotations

import torch

from .mesh import TriMesh


def fit_rotations(rest: torch.Tensor, deformed: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                  weights: torch.Tensor) -> torch.Tensor:
    """Best-fit rotation per vertex from its one-ring (V, 3, 3), computed without grad."""
    with torch.no_grad():
        e_rest = rest[src] - rest[dst]          # (E2, 3)
        e_def = deformed[src] - deformed[dst]   # (E2, 3)
        # covariance S_i = sum_j w_ij * e_rest_ij ⊗ e_def_ij
        outer = weights[:, None, None] * (e_rest[:, :, None] @ e_def[:, None, :])  # (E2, 3, 3)
        cov = torch.zeros(rest.shape[0], 3, 3, device=rest.device, dtype=rest.dtype)
        cov.index_add_(0, src, outer)

        U, _, Vh = torch.linalg.svd(cov)
        R = (Vh.transpose(1, 2) @ U.transpose(1, 2))
        # reflection fix: flip the axis of least singular value where det < 0
        det = torch.linalg.det(R)
        Vh_fixed = Vh.clone()
        Vh_fixed[det < 0, 2, :] *= -1
        R = (Vh_fixed.transpose(1, 2) @ U.transpose(1, 2))
    return R


def arap_energy(mesh: TriMesh, deformed: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Mean ARAP energy of `deformed` (V, 3) w.r.t. the mesh rest pose. Differentiable in `deformed`."""
    de = mesh.directed_edges()
    src, dst = de[:, 0], de[:, 1]
    if weights is None:
        weights = torch.ones(de.shape[0], device=deformed.device, dtype=deformed.dtype)

    rest = mesh.verts
    R = fit_rotations(rest, deformed, src, dst, weights)  # (V, 3, 3), no grad

    e_rest = rest[src] - rest[dst]
    e_def = deformed[src] - deformed[dst]
    e_rot = (R[src] @ e_rest[:, :, None]).squeeze(-1)
    return (weights * (e_def - e_rot).pow(2).sum(dim=1)).mean()
