"""Diagnose a trained DTS checkpoint: NaNs, weight stats, visible-triangle count."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts.tsplus_bridge import attach_times, render_deformed  # noqa: E402

import torch  # noqa: E402
from argparse import ArgumentParser  # noqa: E402
from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from scene import Scene, TriangleModel  # noqa: E402

parser = ArgumentParser()
lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
import os  # noqa: E402
model_dir = os.environ.get("DTS_MODEL", "out/dts_jumpingjacks")
scene_dir = os.environ.get("DTS_SCENE", "data/data/jumpingjacks")
args = parser.parse_args(["-s", scene_dir, "-m", model_dir, "--eval"])
dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

triangles = TriangleModel(dataset.sh_degree)
scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma, load_iteration=-1, shuffle=False)

v = triangles.get_vertices
w = triangles.get_vertex_weight
print(f"verts {tuple(v.shape)} nan={bool(v.isnan().any())} absmax={float(v.abs().max()):.3f}")
print(f"tris  {tuple(triangles.get_triangle_indices.shape)}")
print(f"weights min={float(w.min()):.4f} max={float(w.max()):.4f} mean={float(w.mean()):.4f}")
print(f"sigma = {float(triangles.get_sigma):.6f}")

cams = scene.getTestCameras()
attach_times(cams, dataset.source_path)
bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

# load deformer if present
deformer = None
dp = Path(dataset.model_path) / "deformer.pt"
if dp.exists():
    from dts.deform import ControlNodeDeformer, DeformationMLP
    ck = torch.load(dp, weights_only=False)
    if ck["kind"] == "mlp":
        deformer = DeformationMLP(hidden=256, depth=8, x_freqs=10, t_freqs=6).cuda()
    else:
        deformer = ControlNodeDeformer(triangles.get_vertices.detach(),
                                       num_nodes=ck["num_nodes"],
                                       knn=ck["state"]["skin_idx"].shape[1],
                                       learn_skinning="skin_logits" in ck["state"]).cuda()
    deformer.load_state_dict(ck["state"])
    print(f"deformer loaded ({ck['kind']})")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

out_dir = Path(dataset.model_path) / "diag"
out_dir.mkdir(exist_ok=True)
with torch.no_grad():
    for i in [0, len(cams) // 2, len(cams) - 1]:
        cam = cams[i]
        dv = deformer(triangles.get_vertices, torch.tensor(cam.time, device="cuda")) if deformer else None
        pkg = render_deformed(cam, triangles, pipe, bg, vertices_override=dv)
        img = pkg["render"].clamp(0, 1)
        gt = cam.original_image.cuda()
        side = torch.cat([gt, img], dim=2).permute(1, 2, 0).cpu().numpy()
        Image.fromarray((side * 255).astype(np.uint8)).save(out_dir / f"test{i}_t{cam.time:.2f}_gt_vs_pred.png")
        if i == 0:
            print(f"render: min={float(img.min()):.3f} max={float(img.max()):.3f} mean={float(img.mean()):.3f}")
            print(f"visible triangles: {int(pkg['visibility_filter'].sum())} / {triangles.get_triangle_indices.shape[0]}")
print(f"comparisons saved to {out_dir}")
