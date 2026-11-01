"""Triangle mesh container with the topology utilities the deformation stage needs.

The whole point of triangles over Gaussians is the shared-vertex graph, so this
class exposes exact edges / neighborhoods rather than approximating anything
over a point soup.
"""

from __future__ import annotations

import torch


class TriMesh:
    def __init__(self, verts: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor | None = None):
        assert verts.ndim == 2 and verts.shape[1] == 3, "verts must be (V, 3)"
        assert faces.ndim == 2 and faces.shape[1] == 3, "faces must be (F, 3)"
        self.verts = verts
        self.faces = faces.long()
        # Per-vertex RGB in [0, 1]; defaults to mid-gray.
        self.colors = colors if colors is not None else torch.full_like(verts, 0.5)
        self._edges: torch.Tensor | None = None
        self._directed_edges: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.verts.device

    @property
    def num_verts(self) -> int:
        return self.verts.shape[0]

    @property
    def num_faces(self) -> int:
        return self.faces.shape[0]

    def to(self, device) -> "TriMesh":
        m = TriMesh(self.verts.to(device), self.faces.to(device), self.colors.to(device))
        return m

    def edges(self) -> torch.Tensor:
        """Unique undirected edges (E, 2) with e[:, 0] < e[:, 1]."""
        if self._edges is None:
            f = self.faces
            e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=0)
            e = torch.sort(e, dim=1).values
            self._edges = torch.unique(e, dim=0)
        return self._edges

    def directed_edges(self) -> torch.Tensor:
        """Both directions of every unique edge, (2E, 2) as (src, dst)."""
        if self._directed_edges is None:
            e = self.edges()
            self._directed_edges = torch.cat([e, e.flip(1)], dim=0)
        return self._directed_edges

    def face_normals(self, verts: torch.Tensor | None = None, eps: float = 1e-9) -> torch.Tensor:
        """Unit face normals (F, 3). Pass deformed verts to evaluate a deformed pose."""
        v = self.verts if verts is None else verts
        p0, p1, p2 = v[self.faces[:, 0]], v[self.faces[:, 1]], v[self.faces[:, 2]]
        n = torch.cross(p1 - p0, p2 - p0, dim=1)
        return n / (n.norm(dim=1, keepdim=True) + eps)

    def face_areas(self, verts: torch.Tensor | None = None) -> torch.Tensor:
        v = self.verts if verts is None else verts
        p0, p1, p2 = v[self.faces[:, 0]], v[self.faces[:, 1]], v[self.faces[:, 2]]
        return 0.5 * torch.cross(p1 - p0, p2 - p0, dim=1).norm(dim=1)

    def edge_lengths(self, verts: torch.Tensor | None = None) -> torch.Tensor:
        v = self.verts if verts is None else verts
        e = self.edges()
        return (v[e[:, 0]] - v[e[:, 1]]).norm(dim=1)

    @staticmethod
    def make_grid(nx: int, ny: int, size: float = 1.0, device="cpu", checker: bool = True) -> "TriMesh":
        """Planar sheet in the XY plane, centered at origin — the toy-experiment mesh."""
        xs = torch.linspace(-size / 2, size / 2, nx, device=device)
        ys = torch.linspace(-size / 2, size / 2, ny, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        verts = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.zeros(nx * ny, device=device)], dim=1)

        idx = torch.arange(nx * ny, device=device).reshape(ny, nx)
        a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
        tri1 = torch.stack([a.reshape(-1), b.reshape(-1), c.reshape(-1)], dim=1)
        tri2 = torch.stack([b.reshape(-1), d.reshape(-1), c.reshape(-1)], dim=1)
        faces = torch.cat([tri1, tri2], dim=0)

        if checker:
            ci = (gx > 0).long() ^ (gy > 0).long()
            base = torch.tensor([[0.9, 0.2, 0.2], [0.2, 0.4, 0.9]], device=device)
            colors = base[ci.reshape(-1)]
            # add a gradient so photometric loss sees texture within squares
            colors = colors * (0.6 + 0.4 * (gx.reshape(-1, 1) + size / 2) / size)
        else:
            colors = None
        return TriMesh(verts, faces, colors)
