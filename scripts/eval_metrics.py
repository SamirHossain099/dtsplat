"""Full-metric evaluation over best checkpoints: PSNR / SSIM / LPIPS (alex+vgg),
at native 800x800 and area-downsampled 400x400 (resolution-fair vs the original
D-NeRF paper, which evaluated at 400x400).

Usage:
  python scripts/eval_metrics.py                      # all scenes, best table below
  DTS_ONLY=jumpingjacks:out/dts_jumpingjacks_v29 ...  # single override pair
Writes out/metrics.json and prints a markdown table.
"""
from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts.tsplus_bridge import attach_times, render_deformed  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from scene import Scene, TriangleModel  # noqa: E402

from dts.deform import ControlNodeDeformer, DeformationMLP  # noqa: E402
from dts.losses import psnr, ssim  # noqa: E402

# scene -> best model dir (per-scene winners from RESEARCH.md final table)
BEST = {
    "jumpingjacks": "out/dts_jumpingjacks_v29",
    "trex": "out/dts_trex_v3",
    "bouncingballs": "out/dts_bouncingballs_v1",
    "hook": "out/dts_hook_v1",
    "mutant": "out/dts_mutant_v2",
    "standup": "out/dts_standup_v2",
    "hellwarrior": "out/dts_hellwarrior_v1",
    "lego": "out/dts_lego_v1",
}


def load_model(scene_name: str, model_dir: str):
    parser = ArgumentParser()
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    args = parser.parse_args(["-s", f"data/data/{scene_name}", "-m", model_dir, "--eval"])
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

    best_path = Path(model_dir) / "deformer_best.pt"
    ck = torch.load(best_path, weights_only=False)
    it = ck.get("iteration", -1)

    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma,
                  load_iteration=it, shuffle=False)
    if ck["kind"] == "mlp":
        deformer = DeformationMLP(hidden=256, depth=8, x_freqs=10, t_freqs=6).cuda()
    else:
        deformer = ControlNodeDeformer(
            triangles.get_vertices.detach(), num_nodes=ck["num_nodes"],
            knn=ck["state"]["skin_idx"].shape[1],
            learn_skinning="skin_logits" in ck["state"]).cuda()
    deformer.load_state_dict(ck["state"])
    cams = scene.getTestCameras()
    attach_times(cams, dataset.source_path)
    return triangles, deformer, cams, pipe, it


@torch.no_grad()
def evaluate_scene(scene_name: str, model_dir: str, lpips_alex, lpips_vgg) -> dict:
    triangles, deformer, cams, pipe, it = load_model(scene_name, model_dir)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    acc = {k: 0.0 for k in ("psnr800", "ssim800", "lpA800", "lpV800",
                            "psnr400", "ssim400", "lpA400", "lpV400")}
    n = 0
    for cam in cams:
        t = torch.tensor(cam.time, device="cuda")
        dv = deformer(triangles.get_vertices, t)
        img = render_deformed(cam, triangles, pipe, bg, vertices_override=dv)["render"].clamp(0, 1)
        gt = cam.original_image.cuda().clamp(0, 1)

        def metrics(a, b, suffix):
            acc["psnr" + suffix] += psnr(a, b)
            acc["ssim" + suffix] += float(ssim(a, b))
            a2, b2 = a[None] * 2 - 1, b[None] * 2 - 1  # lpips expects [-1,1]
            acc["lpA" + suffix] += float(lpips_alex(a2, b2))
            acc["lpV" + suffix] += float(lpips_vgg(a2, b2))

        metrics(img, gt, "800")
        img4 = F.interpolate(img[None], size=(400, 400), mode="area")[0]
        gt4 = F.interpolate(gt[None], size=(400, 400), mode="area")[0]
        metrics(img4, gt4, "400")
        n += 1
    out = {k: v / n for k, v in acc.items()}
    out["iteration"] = it
    out["views"] = n
    out["tris"] = int(triangles.get_triangle_indices.shape[0])
    return out


def main():
    import lpips
    lpips_alex = lpips.LPIPS(net="alex").cuda()
    lpips_vgg = lpips.LPIPS(net="vgg").cuda()

    pairs = BEST
    out_name = "metrics.json"
    if os.environ.get("DTS_ONLY"):
        name, mdir = os.environ["DTS_ONLY"].split(":", 1)
        pairs = {name: mdir}
        # single-config evals must never clobber the benchmark table
        out_name = f"metrics_{Path(mdir).name}.json"

    results = {}
    for scene_name, mdir in pairs.items():
        print(f"== {scene_name}  ({mdir})", flush=True)
        try:
            results[scene_name] = evaluate_scene(scene_name, mdir, lpips_alex, lpips_vgg)
            r = results[scene_name]
            print(f"   800: {r['psnr800']:.2f} / {r['ssim800']:.4f} / {r['lpA800']:.4f}a {r['lpV800']:.4f}v"
                  f"   400: {r['psnr400']:.2f} / {r['ssim400']:.4f} / {r['lpA400']:.4f}a", flush=True)
        except Exception as e:  # keep going, report at end
            print(f"   FAILED: {e}", flush=True)
            results[scene_name] = {"error": str(e)}

    ok = [r for r in results.values() if "error" not in r]
    if ok:
        avg = {k: sum(r[k] for r in ok) / len(ok)
               for k in ok[0] if k not in ("iteration", "views", "tris")}
        results["_average"] = avg

    out_path = Path("out") / out_name
    out_path.write_text(json.dumps(results, indent=2))

    print("\n| scene | PSNR800 | SSIM800 | LPIPS-a | LPIPS-v | PSNR400 | SSIM400 |")
    print("|---|---|---|---|---|---|---|")
    for s, r in results.items():
        if s == "_average" or "error" in r:
            continue
        print(f"| {s} | {r['psnr800']:.2f} | {r['ssim800']:.4f} | {r['lpA800']:.4f} | "
              f"{r['lpV800']:.4f} | {r['psnr400']:.2f} | {r['ssim400']:.4f} |")
    if ok:
        print(f"| **avg** | **{avg['psnr800']:.2f}** | **{avg['ssim800']:.4f}** | **{avg['lpA800']:.4f}** | "
              f"**{avg['lpV800']:.4f}** | **{avg['psnr400']:.2f}** | **{avg['ssim400']:.4f}** |")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
