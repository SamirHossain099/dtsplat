"""Deformation models mapping (rest vertex, time) -> deformed vertex.

Option A: DeformationMLP — per-vertex displacement field (D-NeRF style baseline).
Option B: ControlNodeDeformer — sparse control nodes with blended rigid
transforms (LBS). This is the preferred model: rigidity lives at the nodes and
the ARAP term operates on the true triangle graph.

Both are zero-initialized so training starts exactly at the identity
deformation (canonical pose), which keeps early photometric gradients sane.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.include_input = include_input
        freqs = 2.0 ** torch.arange(num_freqs) * torch.pi
        self.register_buffer("freqs", freqs)

    def out_dim(self, in_dim: int) -> int:
        d = 2 * in_dim * self.freqs.shape[0]
        return d + in_dim if self.include_input else d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = x[..., None, :] * self.freqs[:, None]  # (..., L, D)
        enc = torch.cat([xb.sin(), xb.cos()], dim=-1).flatten(-2)
        return torch.cat([x, enc], dim=-1) if self.include_input else enc


def _mlp(in_dim: int, hidden: int, depth: int, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
    last = nn.Linear(hidden, out_dim)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Zhou et al. continuous 6D rotation parameterization -> (…, 3, 3)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


class DeformationMLP(nn.Module):
    """Option A: x, t -> x + MLP(PE(x), PE(t))."""

    def __init__(self, hidden: int = 128, depth: int = 6, x_freqs: int = 6, t_freqs: int = 4):
        super().__init__()
        self.pe_x = PositionalEncoding(x_freqs)
        self.pe_t = PositionalEncoding(t_freqs)
        in_dim = self.pe_x.out_dim(3) + self.pe_t.out_dim(1)
        self.net = _mlp(in_dim, hidden, depth, 3)

    def forward(self, verts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """verts (V, 3), t scalar tensor or (1,) -> deformed verts (V, 3)."""
        t = t.reshape(1, 1).expand(verts.shape[0], 1)
        h = torch.cat([self.pe_x(verts), self.pe_t(t)], dim=-1)
        return verts + self.net(h)


def farthest_point_sampling(pts: torch.Tensor, k: int) -> torch.Tensor:
    """Indices (k,) of an FPS subset of pts (V, 3). Deterministic (starts at point 0)."""
    v = pts.shape[0]
    k = min(k, v)
    idx = torch.zeros(k, dtype=torch.long, device=pts.device)
    dist = torch.full((v,), torch.inf, device=pts.device)
    cur = 0
    for i in range(k):
        idx[i] = cur
        d = (pts - pts[cur]).pow(2).sum(dim=1)
        dist = torch.minimum(dist, d)
        cur = int(dist.argmax())
    return idx


class ControlNodeDeformer(nn.Module):
    """Option B: K control nodes, each with a time-conditioned rigid transform,
    skinned to vertices with k-NN Gaussian RBF weights precomputed on the rest pose.
    """

    def __init__(self, rest_verts: torch.Tensor, num_nodes: int = 32, knn: int = 4,
                 hidden: int = 128, depth: int = 4, t_freqs: int = 4, node_feat_dim: int = 32,
                 learn_skinning: bool = False):
        super().__init__()
        node_idx = farthest_point_sampling(rest_verts, num_nodes)
        nodes = rest_verts[node_idx].clone()
        self.register_buffer("nodes", nodes)  # (K, 3)
        K = nodes.shape[0]
        self.knn = min(knn, K)
        self.learn_skinning = learn_skinning
        self.register_buffer("skin_idx", torch.empty(0, dtype=torch.long))
        self.register_buffer("skin_w", torch.empty(0))
        if learn_skinning:
            # per-vertex logits over the k-NN candidate nodes; lets stretch
            # concentrate at joints instead of smearing over blind RBF falloff
            self.skin_logits = nn.Parameter(torch.empty(0))
        self.rebuild_skinning(rest_verts)

        self.node_feats = nn.Parameter(torch.randn(K, node_feat_dim) * 0.01)
        self.pe_t = PositionalEncoding(t_freqs)
        in_dim = node_feat_dim + self.pe_t.out_dim(1)
        self.net = _mlp(in_dim, hidden, depth, 9)  # 6D rotation + 3 translation, zero-init

    @torch.no_grad()
    def rebuild_skinning(self, rest_verts: torch.Tensor) -> None:
        """Recompute vertex->node k-NN candidates and RBF weights. Call after
        the vertex set changes (pruning/densification): nodes and the learned
        per-node transform MLP are untouched, so the deformation survives.
        With learn_skinning, logits re-init from RBF for the new vertex set —
        recreate the optimizer afterwards so it tracks the new parameter."""
        d2 = torch.cdist(rest_verts, self.nodes).pow(2)          # (V, K)
        nn_d2, nn_idx = d2.topk(self.knn, dim=1, largest=False)  # (V, k)
        sigma2 = nn_d2[:, -1].mean().clamp(min=1e-8)
        w = torch.exp(-nn_d2 / sigma2)
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        self.skin_idx = nn_idx.to(rest_verts.device)
        self.skin_w = w.to(rest_verts.device)
        if self.learn_skinning:
            self.skin_logits = nn.Parameter(torch.log(self.skin_w + 1e-8))

    def skinning_weights(self, detach: bool = False) -> torch.Tensor:
        if self.learn_skinning:
            w = torch.softmax(self.skin_logits, dim=1)
            return w.detach() if detach else w
        return self.skin_w

    def node_transforms(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-node (R (K,3,3), trans (K,3)) at time t."""
        K = self.nodes.shape[0]
        tt = t.reshape(1, 1).expand(K, 1)
        h = torch.cat([self.node_feats, self.pe_t(tt)], dim=-1)
        out = self.net(h)
        d6 = out[:, :6]
        # zero-init MLP -> d6 == 0, which is degenerate for 6D rotations; bias to identity
        d6 = d6 + torch.tensor([1.0, 0, 0, 0, 1.0, 0], device=out.device, dtype=out.dtype)
        return rotation_6d_to_matrix(d6), out[:, 6:]

    def forward(self, verts: torch.Tensor, t: torch.Tensor,
                detach_skinning: bool = False) -> torch.Tensor:
        """detach_skinning: use for regularizer branches — rigidity priors
        otherwise learn to suppress motion via the skinning lever instead of
        constraining the motion itself."""
        R, trans = self.node_transforms(t)                       # (K,3,3), (K,3)
        g = self.nodes                                           # (K, 3)
        idx = self.skin_idx                                      # (V,k)
        w = self.skinning_weights(detach=detach_skinning)        # (V,k)
        Ri, gi, ti = R[idx], g[idx], trans[idx]                  # (V,k,3,3), (V,k,3), (V,k,3)
        local = verts[:, None, :] - gi                           # (V,k,3)
        rotated = (Ri @ local[..., None]).squeeze(-1)            # (V,k,3)
        per_node = rotated + gi + ti                             # (V,k,3)
        return (w[..., None] * per_node).sum(dim=1)
