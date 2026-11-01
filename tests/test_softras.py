import torch

from dts.losses import photometric_loss, psnr, ssim
from dts.mesh import TriMesh
from dts.softras import Camera, render


def _scene(im=32):
    m = TriMesh.make_grid(6, 6, size=1.0)
    cam = Camera.look_at((0, 0, 2.0), (0, 0, 0), (0, 1, 0), fov_deg=45, height=im, width=im)
    return m, cam


def test_render_shape_and_range():
    m, cam = _scene()
    img = render(m.verts, m.faces, m.colors, cam)
    assert img.shape == (3, 32, 32)
    assert float(img.min()) >= 0 and float(img.max()) <= 1


def test_mesh_visible_against_background():
    m, cam = _scene()
    img = render(m.verts, m.faces, m.colors, cam, background=1.0)
    # center pixels should not be pure white background
    center = img[:, 12:20, 12:20]
    assert float((center - 1.0).abs().mean()) > 0.05


def test_gradients_flow_to_vertices():
    m, cam = _scene()
    verts = m.verts.clone().requires_grad_(True)
    img = render(verts, m.faces, m.colors, cam)
    img.mean().backward()
    assert verts.grad is not None
    assert float(verts.grad.abs().sum()) > 0


def test_moving_mesh_changes_image():
    m, cam = _scene()
    img0 = render(m.verts, m.faces, m.colors, cam)
    img1 = render(m.verts + torch.tensor([0.3, 0.0, 0.0]), m.faces, m.colors, cam)
    assert float((img0 - img1).abs().mean()) > 1e-3


def test_ssim_and_psnr_sanity():
    m, cam = _scene()
    img = render(m.verts, m.faces, m.colors, cam)
    assert abs(float(ssim(img, img)) - 1.0) < 1e-5
    assert psnr(img, img.clone()) > 60
    assert float(photometric_loss(img, img)) < 1e-6
