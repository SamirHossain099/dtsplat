import torch

from dts.deform import (ControlNodeDeformer, DeformationMLP, farthest_point_sampling,
                        rotation_6d_to_matrix)
from dts.mesh import TriMesh


def test_mlp_zero_init_is_identity():
    torch.manual_seed(0)
    m = TriMesh.make_grid(6, 6)
    model = DeformationMLP()
    out = model(m.verts, torch.tensor(0.37))
    assert torch.allclose(out, m.verts, atol=1e-6)


def test_nodes_zero_init_is_identity():
    torch.manual_seed(0)
    m = TriMesh.make_grid(8, 8)
    model = ControlNodeDeformer(m.verts, num_nodes=16)
    out = model(m.verts, torch.tensor(0.5))
    assert torch.allclose(out, m.verts, atol=1e-5)


def test_skinning_weights_normalized():
    m = TriMesh.make_grid(8, 8)
    model = ControlNodeDeformer(m.verts, num_nodes=16, knn=4)
    s = model.skin_w.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


def test_rotation_6d_identity():
    d6 = torch.tensor([1.0, 0, 0, 0, 1.0, 0])
    R = rotation_6d_to_matrix(d6)
    assert torch.allclose(R, torch.eye(3), atol=1e-6)


def test_rotation_6d_is_rotation():
    g = torch.Generator().manual_seed(1)
    R = rotation_6d_to_matrix(torch.randn(10, 6, generator=g))
    eye = torch.eye(3).expand(10, 3, 3)
    assert torch.allclose(R @ R.transpose(1, 2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(10), atol=1e-5)


def test_fps_distinct_and_covers():
    m = TriMesh.make_grid(10, 10)
    idx = farthest_point_sampling(m.verts, 20)
    assert len(set(idx.tolist())) == 20


def test_learned_skinning_normalized_and_trainable():
    torch.manual_seed(0)
    m = TriMesh.make_grid(8, 8)
    model = ControlNodeDeformer(m.verts, num_nodes=16, knn=4, learn_skinning=True)
    w = model.skinning_weights()
    assert torch.allclose(w.sum(dim=1), torch.ones(m.num_verts), atol=1e-5)
    # starts near the RBF init
    assert torch.allclose(w, model.skin_w, atol=1e-4)
    out = model(m.verts, torch.tensor(0.3))
    (out - (m.verts + 0.1)).pow(2).mean().backward()
    assert model.skin_logits.grad is not None
    # zero-init transforms are identity => output invariant to skinning => zero grad;
    # after perturbing transforms the skinning gradient must be nonzero
    model.zero_grad()
    with torch.no_grad():
        model.net[-1].weight.normal_(0, 0.1)
    out = model(m.verts, torch.tensor(0.3))
    (out - (m.verts + 0.1)).pow(2).mean().backward()
    assert float(model.skin_logits.grad.abs().sum()) > 0


def test_rebuild_skinning_replaces_logits():
    torch.manual_seed(0)
    m = TriMesh.make_grid(8, 8)
    model = ControlNodeDeformer(m.verts, num_nodes=16, knn=4, learn_skinning=True)
    v2 = m.verts[: m.num_verts // 2]
    model.rebuild_skinning(v2)
    assert model.skin_logits.shape[0] == v2.shape[0]
    assert model(v2, torch.tensor(0.5)).shape == v2.shape


def test_deformers_are_trainable():
    torch.manual_seed(0)
    m = TriMesh.make_grid(6, 6)
    for model in (DeformationMLP(hidden=32, depth=2), ControlNodeDeformer(m.verts, num_nodes=8)):
        out = model(m.verts, torch.tensor(0.2))
        loss = (out - (m.verts + 0.1)).pow(2).mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)
