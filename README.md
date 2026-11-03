# dtsplat

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23875183.svg)](https://doi.org/10.5281/zenodo.23875183)

Dynamic triangle splatting: reconstruction of continuously deforming scenes with
triangle splatting primitives, together with the finite-difference audit and the
repair of the position gradients in the released Triangle Splatting+ rasterizer.

Companion code for the manuscript "Dynamic triangle splatting with repaired
position gradients" (Samir Hossain, Texas Tech University).

## What is here

- `dts/` -- the library: triangle mesh container with exact edge topology,
  as-rigid-as-possible energy on the triangle graph, control-node deformation
  with learned blend-skinning weights, losses (photometric, edge hinge, mask),
  a reference soft rasterizer for unit testing, a D-NeRF loader, and the bridge
  that renders deformed vertices through the Triangle Splatting+ CUDA rasterizer.
- `patches/` -- the gradient repair and Windows build fixes, as patches against
  the upstream Triangle Splatting+ repository and its rasterization submodule.
  The patched code remains governed by the upstream licenses; see
  `patches/README.md`.
- `scripts/` -- training (`train_dnerf.py`), evaluation (`eval_metrics.py`,
  `bench_efficiency.py`, `render_figures.py`, `diag_dnerf.py`), the physics
  demonstration (`physics_demo.py`), the gradient audits (`probe_grads.py`,
  `probe_fd.py`, `probe_single_triangle.py`), the HyperNeRF converter, and a
  CPU toy experiment (`train_toy.py`).
- `tests/` -- unit tests for the library (CPU only, no dataset needed).

## Setup

Tested on Windows 11 with Python 3.12, CUDA 12.8, and an RTX 5080 (16 GB);
the code is standard PyTorch and is not Windows-specific.

```
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest tests
```

Clone Triangle Splatting+ (github.com/trianglesplatting2/triangle-splatting2,
with submodules) into `external/triangle-splatting-plus` at the audited commits
listed below, apply the patches in `patches/`, and build its rasterizer and
triangulation modules following its README. The D-NeRF scenes come from the D-NeRF release; HyperNeRF scenes from
the HyperNeRF release, converted with `scripts/hypernerf_convert.py`.

## Audited upstream code

The gradient audit and the patches refer to this exact upstream state, cloned
on 2026-07-05:

| Repository | Commit | Date |
|---|---|---|
| trianglesplatting2/triangle-splatting2 | `fcde03dd0c3953ce1d47b210485a5e8a9e14b016` | 2025-10-09 |
| trianglesplatting2/diff-triangle2-rasterization (submodule) | `42e62aaf03f87835105ed84dd8ed513efd458716` | 2025-10-08 |

SHA-256 of the unmodified `cuda_rasterizer/backward.cu` at the rasterizer
commit, the file in which the defects were found:

```
8ceaf7c0199dd76b914b6269428cde76598e93e6371228a7c1f4008a948860aa
```

Check a clone against it, before applying the patches:

```
git -C external/triangle-splatting-plus/submodules/diff-triangle2-rasterization show 42e62aa:cuda_rasterizer/backward.cu | sha256sum
```

As of 2026-11-02 neither repository had a later commit. The upstream issue cited
in the manuscript as consistent with the finding is triangle-splatting2 issue 25
(opened 2026-03-24, open without a reply as of 2026-11-02). The digest is
recorded here rather than a copy of the file because the upstream rasterizer is
partly governed by the Inria Gaussian-Splatting license.

## Reproducing the paper

Training, per scene:

```
python scripts/train_dnerf.py -s data/data/jumpingjacks -m out/jj --deform-model nodes --num-nodes 512 --w-arap 0.05 --w-edge-hinge 0.1 --w-mask 0.3 --learn-skinning --z-damp 0.9 --warmup 3000 --total 20000 --eval
```

`scripts/eval_metrics.py` regenerates the benchmark table over the released
checkpoints (per-scene zips attached to the v0.1.0 GitHub release; unpack them
into `out/`), `scripts/bench_efficiency.py` the efficiency table, and
`scripts/probe_single_triangle.py` the gradient audit. Per-scene configurations
match the manuscript; depth damping (`--z-damp`) is disclosed per scene there.

## License

MIT for the code in this repository. The patches apply to third-party code
governed by its own licenses (Triangle Splatting+ and the Inria 3D Gaussian
Splatting rasterizer license); applying them does not relicense that code.
