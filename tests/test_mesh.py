import torch

from dts.mesh import TriMesh


def test_grid_topology():
    m = TriMesh.make_grid(4, 3)
    assert m.num_verts == 12
    assert m.num_faces == 2 * 3 * 2  # 2 tris per cell, 3x2 cells
    e = m.edges()
    # undirected, unique, sorted pairs
    assert (e[:, 0] < e[:, 1]).all()
    assert e.shape[0] == len({(int(a), int(b)) for a, b in e})


def test_directed_edges_symmetric():
    m = TriMesh.make_grid(5, 5)
    de = m.directed_edges()
    assert de.shape[0] == 2 * m.edges().shape[0]
    pairs = {(int(a), int(b)) for a, b in de}
    assert all((b, a) in pairs for a, b in pairs)


def test_face_normals_unit_and_planar():
    m = TriMesh.make_grid(6, 6)
    n = m.face_normals()
    assert torch.allclose(n.norm(dim=1), torch.ones(m.num_faces), atol=1e-5)
    # planar grid in XY -> normals along ±z
    assert torch.allclose(n[:, :2].abs().max(), torch.tensor(0.0), atol=1e-6)


def test_edge_lengths_positive():
    m = TriMesh.make_grid(5, 4)
    assert (m.edge_lengths() > 0).all()
