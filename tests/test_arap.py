import torch

from dts.arap import arap_energy
from dts.deform import rotation_6d_to_matrix
from dts.mesh import TriMesh


def _random_rotation(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return rotation_6d_to_matrix(torch.randn(6, generator=g))


def test_rigid_motion_has_zero_energy():
    m = TriMesh.make_grid(8, 8)
    R = _random_rotation(3)
    t = torch.tensor([0.3, -1.2, 0.7])
    deformed = m.verts @ R.T + t
    e = arap_energy(m, deformed)
    assert float(e) < 1e-10


def test_identity_has_zero_energy():
    m = TriMesh.make_grid(6, 6)
    assert float(arap_energy(m, m.verts.clone())) < 1e-12


def test_stretch_has_positive_energy():
    m = TriMesh.make_grid(8, 8)
    stretched = m.verts * torch.tensor([2.0, 1.0, 1.0])
    assert float(arap_energy(m, stretched)) > 1e-3


def test_bend_energy_below_stretch():
    # smooth isometric-ish bend should cost less than a hard stretch
    m = TriMesh.make_grid(10, 10)
    x, y = m.verts[:, 0], m.verts[:, 1]
    bent = torch.stack([x, y, 0.1 * torch.sin(3 * x)], dim=1)
    stretched = m.verts * torch.tensor([2.0, 1.0, 1.0])
    assert float(arap_energy(m, bent)) < float(arap_energy(m, stretched))


def test_gradients_flow_to_deformed():
    m = TriMesh.make_grid(6, 6)
    deformed = (m.verts + 0.05).clone().requires_grad_(True)
    e = arap_energy(m, deformed * torch.tensor([1.3, 1.0, 1.0]))
    e.backward()
    assert deformed.grad is not None
    assert torch.isfinite(deformed.grad).all()
