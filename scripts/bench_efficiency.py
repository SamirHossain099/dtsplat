"""Efficiency benchmark over best checkpoints: render FPS (full pipeline and
rasterizer-only), primitive counts, checkpoint sizes, deformer parameter counts.

Run AFTER eval_metrics.py (needs exclusive GPU for clean timings).
Writes out/efficiency.json and prints a markdown table.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts.tsplus_bridge import render_deformed  # noqa: E402

import torch  # noqa: E402

from eval_metrics import BEST, load_model  # noqa: E402  (same scripts dir)


@torch.no_grad()
def bench_scene(scene_name: str, model_dir: str, n_iters: int = 100) -> dict:
    triangles, deformer, cams, pipe, it = load_model(scene_name, model_dir)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    cam = cams[0]
    t = torch.tensor(cam.time, device="cuda")

    # warmup
    for _ in range(10):
        dv = deformer(triangles.get_vertices, t)
        render_deformed(cam, triangles, pipe, bg, vertices_override=dv)
    torch.cuda.synchronize()

    # full pipeline: deformation + rasterization
    t0 = time.perf_counter()
    for _ in range(n_iters):
        dv = deformer(triangles.get_vertices, t)
        render_deformed(cam, triangles, pipe, bg, vertices_override=dv)
    torch.cuda.synchronize()
    full_fps = n_iters / (time.perf_counter() - t0)

    # rasterizer-only: fixed deformed vertices
    dv = deformer(triangles.get_vertices, t)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        render_deformed(cam, triangles, pipe, bg, vertices_override=dv)
    torch.cuda.synchronize()
    raster_fps = n_iters / (time.perf_counter() - t0)

    pc_dir = Path(model_dir) / "point_cloud" / f"iteration_{it}"
    size_mb = sum(f.stat().st_size for f in pc_dir.glob("**/*") if f.is_file()) / 1e6 \
        if pc_dir.exists() else -1
    deformer_params = sum(p.numel() for p in deformer.parameters())

    return {
        "full_fps": round(full_fps, 1),
        "raster_fps": round(raster_fps, 1),
        "triangles": int(triangles.get_triangle_indices.shape[0]),
        "vertices": int(triangles.get_vertices.shape[0]),
        "checkpoint_MB": round(size_mb, 1),
        "deformer_params_K": round(deformer_params / 1e3, 1),
        "iteration": it,
    }


def main():
    results = {}
    for scene_name, mdir in BEST.items():
        print(f"== {scene_name}", flush=True)
        try:
            results[scene_name] = bench_scene(scene_name, mdir)
            r = results[scene_name]
            print(f"   {r['full_fps']} fps full / {r['raster_fps']} fps raster, "
                  f"{r['triangles']} tris, {r['checkpoint_MB']} MB", flush=True)
        except Exception as e:
            print(f"   FAILED: {e}", flush=True)
            results[scene_name] = {"error": str(e)}

    Path("out/efficiency.json").write_text(json.dumps(results, indent=2))
    print("\n| scene | FPS (full) | FPS (raster) | triangles | vertices | ckpt MB | deformer KB |")
    print("|---|---|---|---|---|---|---|")
    for s, r in results.items():
        if "error" in r:
            continue
        print(f"| {s} | {r['full_fps']} | {r['raster_fps']} | {r['triangles']} | "
              f"{r['vertices']} | {r['checkpoint_MB']} | {r['deformer_params_K']} |")
    print("\nsaved out/efficiency.json")


if __name__ == "__main__":
    main()
