"""Figure-grade renders per scene: RGB, normal map, depth, alpha at 3 timestamps.

Evidences the mesh-native claim (well-defined normals/geometry from triangles).
Writes out/figures/<scene>/{rgb,normal,depth,alpha}_t{..}.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts.tsplus_bridge import render_deformed  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from eval_metrics import BEST, load_model  # noqa: E402


def save(t: torch.Tensor, path: Path):
    a = t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    Image.fromarray((a * 255).astype(np.uint8)).save(path)


@torch.no_grad()
def main():
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    for scene_name, mdir in BEST.items():
        out = Path("out/figures") / scene_name
        out.mkdir(parents=True, exist_ok=True)
        try:
            triangles, deformer, cams, pipe, _ = load_model(scene_name, mdir)
        except Exception as e:
            print(f"{scene_name} FAILED: {e}", flush=True)
            continue
        for cam in (cams[0], cams[len(cams) // 2], cams[-1]):
            t = torch.tensor(cam.time, device="cuda")
            dv = deformer(triangles.get_vertices, t)
            pkg = render_deformed(cam, triangles, pipe, bg, vertices_override=dv)
            tag = f"t{cam.time:.2f}"
            save(pkg["render"], out / f"rgb_{tag}.png")
            save(pkg["rend_normal"] * 0.5 + 0.5, out / f"normal_{tag}.png")
            d = pkg["surf_depth"]
            dmax = float(d.max()) if float(d.max()) > 0 else 1.0
            save(d / dmax, out / f"depth_{tag}.png")
            save(pkg["rend_alpha"], out / f"alpha_{tag}.png")
            save(cam.original_image.cuda(), out / f"gt_{tag}.png")
        print(f"{scene_name}: figures saved to {out}", flush=True)


if __name__ == "__main__":
    main()
