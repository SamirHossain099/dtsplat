"""Dynamic Triangle Splatting — Stage 2 trainer on D-NeRF scenes.

Joint optimization (monocular input forces it, as in Deformable-3DGS):
  Phase 1 (warmup):  TS+ static training over ALL frames ignoring time —
                     canonical triangles form as a motion-blurred average.
                     Densification/pruning active (topology may change).
  Phase 2 (dynamic): topology frozen; a deformation model (control nodes or
                     per-vertex MLP) moves vertices per timestamp before
                     rasterization; exact ARAP on the triangle graph
                     regularizes; canonical + deformer train jointly.

Run:  python scripts/train_dnerf.py -s data/data/jumpingjacks -m out/dts_jumpingjacks
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path
from random import randint

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dts.tsplus_bridge import TS_ROOT, attach_times, render_deformed  # noqa: E402  (sets up TS+ paths)

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402  (TS+ modules)
from scene import Scene, TriangleModel  # noqa: E402
from utils.loss_utils import l1_loss, ssim  # noqa: E402
from utils.image_utils import psnr  # noqa: E402
from tqdm import tqdm  # noqa: E402

from dts.arap import arap_energy  # noqa: E402
from dts.deform import ControlNodeDeformer, DeformationMLP  # noqa: E402
from dts.losses import edge_hinge_loss, edge_length_loss  # noqa: E402
from dts.mesh import TriMesh  # noqa: E402


def build_deformer(kind: str, verts: torch.Tensor, num_nodes: int,
                   learn_skinning: bool = False) -> torch.nn.Module:
    if kind == "mlp":
        # Deformable-3DGS capacity: articulated motion needs more than 6x128
        return DeformationMLP(hidden=256, depth=8, x_freqs=10, t_freqs=6).to(verts.device)
    # knn=8 with many nodes: smoother blending across node boundaries
    return ControlNodeDeformer(verts.detach(), num_nodes=num_nodes, knn=8,
                               learn_skinning=learn_skinning).to(verts.device)


def evaluate(cams, triangles, pipe, background, deformer=None, max_views: int = 10) -> float:
    psnrs = []
    with torch.no_grad():
        for cam in cams[:max_views]:
            dv = None
            if deformer is not None:
                t = torch.tensor(cam.time, device="cuda")
                dv = deformer(triangles.get_vertices, t)
            img = render_deformed(cam, triangles, pipe, background, vertices_override=dv)["render"]
            psnrs.append(psnr(img, cam.original_image.cuda()).mean().item())
    return sum(psnrs) / len(psnrs)


def main() -> None:
    parser = ArgumentParser(description="Dynamic Triangle Splatting on D-NeRF")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--warmup", type=int, default=3000)
    parser.add_argument("--total", type=int, default=20000)
    parser.add_argument("--deform-model", choices=["nodes", "mlp"], default="nodes")
    parser.add_argument("--num-nodes", type=int, default=64)
    parser.add_argument("--learn-skinning", action="store_true")
    parser.add_argument("--w-arap", type=float, default=0.2)
    parser.add_argument("--w-edge", type=float, default=0.0)  # off unless requested
    parser.add_argument("--edge-start", type=int, default=-1,
                        help="if >0, ramp edge loss 0->w-edge from this iter to total "
                             "(edge loss suppresses motion while it is being learned; "
                             "apply it after motion converges to clean geometry)")
    parser.add_argument("--w-edge-hinge", type=float, default=0.0,
                        help="hinge stretch penalty: zero below --hinge-threshold x rest "
                             "length, quadratic beyond — kills needles, spares motion")
    parser.add_argument("--hinge-threshold", type=float, default=2.0)
    parser.add_argument("--warmup-tmax", type=float, default=-1.0,
                        help="if >0, warmup trains only on frames with time <= tmax: the "
                             "canonical becomes a real pose instead of a motion-blurred "
                             "mean (which amputates fast thin geometry like arms)")
    parser.add_argument("--t-ramp", type=int, default=0,
                        help="if >0 with --warmup-tmax: expand the training time window "
                             "from tmax to 1.0 over this many iters after warmup. A t0 "
                             "canonical is perfect at t=0 and wrong at t=1; instant full-"
                             "range training lets pruning execute geometry the deformer "
                             "has not yet learned to move (v22 collapse to 0 triangles)")
    parser.add_argument("--w-mask", type=float, default=0.0,
                        help="L1 loss between rendered alpha and GT alpha mask. Missing "
                             "geometry (amputated fast limbs) produces a maximal per-pixel "
                             "signal here, vs the diffuse RGB signal thin structures evade")
    parser.add_argument("--z-damp", type=float, default=0.0,
                        help="damp the camera-ray (depth) component of vertex gradients "
                             "by this factor (0..1). Per-view depth is weakly constrained; "
                             "Adam amplifies its noisy gradients into depth drift "
                             "(single-triangle probe: xy-only recovers, xyz diverges)")
    parser.add_argument("--prune-cap", type=float, default=0.15,
                        help="cap the TS+ prune-threshold ratchet (+0.01/500it). The "
                             "uncapped ratchet (~0.6 by 30k) is export-time minimalism "
                             "for static scenes; it destroys deforming geometry once "
                             "densification stops (v19 collapse 27->16 dB)")
    parser.add_argument("--lr-deform", type=float, default=1e-3)
    parser.add_argument("--test-interval", type=int, default=2000)
    parser.add_argument("--override-start-pruning", type=int, default=-1,
                        help="delay pruning so warmup-blurred thin geometry (arms) "
                             "survives until the deformer can rescue it")
    parser.add_argument("--override-densify-until", type=int, default=-1,
                        help="extend densification so geometry keeps growing while "
                             "motion refines")
    args = parser.parse_args(sys.argv[1:])
    args.iterations = args.total  # keep TS+ LR schedules consistent with our length
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

    # TS+ schedules are tuned for 30k iterations — scale them to our horizon.
    # sigma_until deliberately NOT scaled: fast sigma-hardening bakes in wrong
    # geometry while motion is still being learned (v4 declined after 10k;
    # slow-sigma v3 did not). Sharpness can come from a final polish phase.
    scale = args.total / 30000
    for attr in ("final_opacity_iter", "densify_until_iter",
                 "start_opacity_floor", "start_pruning", "densify_from_iter"):
        setattr(opt, attr, int(getattr(opt, attr) * scale))
    if args.override_start_pruning > 0:
        opt.start_pruning = args.override_start_pruning
    if args.override_densify_until > 0:
        opt.densify_until_iter = args.override_densify_until

    Path(dataset.model_path).mkdir(parents=True, exist_ok=True)
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma)
    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)
    triangles.add_percentage = opt.add_percentage
    triangles.size_probs_zero = opt.size_probs_zero
    triangles.size_probs_zero_image_space = opt.size_probs_zero_image_space

    train_cams = scene.getTrainCameras().copy()
    test_cams = scene.getTestCameras().copy()
    attach_times(train_cams, dataset.source_path, load_alpha=True)
    attach_times(test_cams, dataset.source_path)
    print(f"{len(train_cams)} train / {len(test_cams)} test cameras, "
          f"t in [{min(c.time for c in train_cams):.2f}, {max(c.time for c in train_cams):.2f}]")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # TS+ schedules (train.py) — sigma stays soft while deformation learns
    initial_sigma, final_sigma = opt.set_sigma, 0.0001
    init_opacity, final_opacity = 0.1, 0.9999
    prune_threshold = opt.prune_triangles_threshold

    deformer = None
    deform_opt = None
    skin_opt = None
    arap_mesh = None
    best_psnr = -1.0

    viewpoint_stack = train_cams.copy()
    progress = tqdm(range(1, args.total + 1), desc="DTS training")
    for iteration in progress:
        triangles.update_learning_rate(iteration)

        if iteration < opt.sigma_start:
            current_sigma = initial_sigma
        else:
            p = min((iteration - opt.sigma_start) / max(1, opt.sigma_until - opt.sigma_start), 1.0)
            current_sigma = initial_sigma - (initial_sigma - final_sigma) * p
        triangles.set_sigma(current_sigma)
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        # ---- phase switch: freeze topology, spawn deformer ----
        if iteration == args.warmup:
            verts = triangles.get_vertices
            faces = triangles.get_triangle_indices
            deformer = build_deformer(args.deform_model, verts, args.num_nodes,
                                      learn_skinning=args.learn_skinning)
            # skin_logits get their own optimizer: the parameter is replaced on
            # every topology rebuild, and recreating the MAIN optimizer there
            # wipes MLP momentum every 500 iters (v10 regression)
            base_params = [p for n, p in deformer.named_parameters() if n != "skin_logits"]
            deform_opt = torch.optim.Adam(base_params, lr=args.lr_deform)
            skin_opt = torch.optim.Adam([deformer.skin_logits], lr=args.lr_deform) \
                if args.learn_skinning else None
            arap_mesh = TriMesh(verts.detach().clone(), faces.detach().clone())
            print(f"\n[warmup done] {verts.shape[0]} vertices, {faces.shape[0]} triangles; "
                  f"deformer={args.deform_model}")

        if not viewpoint_stack:
            if args.warmup_tmax > 0:
                if deformer is None:
                    tmax_now = args.warmup_tmax
                elif args.t_ramp > 0:
                    p = min(1.0, (iteration - args.warmup) / max(1, args.t_ramp))
                    tmax_now = args.warmup_tmax + (1.0 - args.warmup_tmax) * p
                else:
                    tmax_now = 1.0
                viewpoint_stack = [c for c in train_cams if c.time <= tmax_now]
            else:
                viewpoint_stack = train_cams.copy()
        cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        deformed = None
        if deformer is not None:
            t = torch.tensor(cam.time, device="cuda")
            deformed = deformer(triangles.get_vertices, t)
            if args.z_damp > 0:
                rays = torch.nn.functional.normalize(
                    deformed.detach() - cam.camera_center.cuda(), dim=1)
                k = args.z_damp
                deformed.register_hook(
                    lambda g, rays=rays, k=k: g - k * (g * rays).sum(1, keepdim=True) * rays)

        # random background: GT is black-composited (rgb*alpha), so
        # recompositing is gt + bg*(1-alpha). Junk can't hide on every color.
        bg = torch.rand(3, device="cuda")
        render_pkg = render_deformed(cam, triangles, pipe, bg, vertices_override=deformed)
        image = render_pkg["render"]
        alpha = cam.gt_alpha.cuda()
        gt = cam.original_image.cuda() + bg[:, None, None] * (1 - alpha)

        loss = (1.0 - opt.lambda_dssim) * l1_loss(image, gt) \
            + opt.lambda_dssim * (1.0 - ssim(image, gt))
        if args.w_mask > 0:
            loss = loss + args.w_mask * l1_loss(render_pkg["rend_alpha"], alpha)

        if iteration < opt.start_opacity_floor:
            loss = loss + triangles.get_vertex_weight[triangles._triangle_indices].mean() * opt.lambda_weight

        if deformer is not None:
            arap_mesh.verts = triangles.get_vertices.detach()
            # regularizers see a skinning-detached copy (same values): with a
            # gradient path into the logits, ARAP learns to erase motion via
            # uniform skinning instead of constraining it (v10/v11 regression)
            if args.learn_skinning and args.deform_model == "nodes":
                deformed_reg = deformer(triangles.get_vertices, t, detach_skinning=True)
            else:
                deformed_reg = deformed
            loss = loss + args.w_arap * arap_energy(arap_mesh, deformed_reg)
            # kills LBS boundary stretch (v7 spike artifacts): squared relative
            # edge-length change on the true triangle edges
            if args.edge_start > 0:
                ramp = max(0.0, min(1.0, (iteration - args.edge_start)
                                    / max(1, args.total - args.edge_start)))
                w_edge_now = args.w_edge * ramp
            else:
                w_edge_now = args.w_edge
            if w_edge_now > 0:
                loss = loss + w_edge_now * edge_length_loss(arap_mesh, deformed_reg)
            if args.w_edge_hinge > 0:
                loss = loss + args.w_edge_hinge * edge_hinge_loss(
                    arap_mesh, deformed_reg, args.hinge_threshold)

        loss.backward()

        with torch.no_grad():
            # importance/size stats feed warmup pruning
            image_size = render_pkg["scaling"].detach()
            m = image_size > triangles.image_size
            triangles.image_size[m] = image_size[m]
            imp = render_pkg["max_blending"].detach()
            m = imp > triangles.importance_score
            triangles.importance_score[m] = imp[m]

            if iteration % 50 == 0:
                progress.set_postfix({"loss": f"{loss.item():.4f}",
                                      "phase": "dyn" if deformer is not None else "warm"})

            # ---- periodic maintenance (mirrors TS+ train.py, every 500 iters) ----
            # Pruning must stay active alongside the opacity-floor ratchet:
            # opacifying junk without pruning it walls off the scene (v1 bug).
            if iteration % 500 == 0:
                n_verts_before = triangles.vertices.shape[0]
                w = triangles.opacity_activation(triangles.vertex_weight[triangles._triangle_indices])
                min_w = w.min(dim=1).values
                delete = (min_w <= prune_threshold).squeeze() \
                    | (triangles.importance_score <= prune_threshold).squeeze() \
                    | (triangles.image_size > opt.prune_size).squeeze()
                # floor guard: an over-eager prune to ~0 crashes the rasterizer
                # backward and is never recoverable anyway
                if iteration > opt.start_pruning and int((~delete).sum()) >= 2000:
                    triangles.prune_triangles(~delete)
                used = torch.zeros(triangles.vertices.shape[0], dtype=torch.bool, device="cuda")
                if triangles._triangle_indices.numel() > 0:
                    used[triangles._triangle_indices.flatten()] = True
                keep_v = (triangles.get_vertex_weight.squeeze() >= prune_threshold) | used
                triangles._prune_vertices(keep_v)

                # densification runs in BOTH phases: TS+ pruning assumes the
                # add_new_gs replenish cycle, prune-only collapses the scene
                # (v2 bug: 1.9M -> 1.7k triangles). New vertices are skinned
                # by the rebuild below.
                # hard guard: unchecked growth overflows 16GB VRAM into shared-
                # memory fallback (~100x slowdown, v24: 154 s/it)
                if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter \
                        and iteration % opt.densification_interval == 0 \
                        and triangles.get_triangle_indices.shape[0] < 2_500_000:
                    if iteration < opt.start_opacity_floor:
                        probs_opacity = True
                    else:
                        probs_opacity = iteration % 1000 == 0
                    triangles.add_new_gs(iteration, cap_max=opt.max_points,
                                         splitt_large_triangles=opt.splitt_large_triangles,
                                         probs_opacity=probs_opacity)

                if iteration > opt.start_opacity_floor:
                    a = min(1.0, max(0.0, (iteration - opt.start_opacity_floor)
                                     / max(1, opt.final_opacity_iter - opt.start_opacity_floor)))
                    triangles.update_min_weight(min(init_opacity + (final_opacity - init_opacity) * a,
                                                    final_opacity))
                    # TS+ ratchet: opacify AND cull harder together — capped for dynamics
                    prune_threshold = min(prune_threshold + 0.01, args.prune_cap)

                # topology changed -> refresh skinning + ARAP graph (learned
                # node transforms survive; only vertex assignments recompute)
                if deformer is not None and triangles.vertices.shape[0] != n_verts_before:
                    verts_now = triangles.get_vertices.detach()
                    if hasattr(deformer, "rebuild_skinning"):
                        deformer.rebuild_skinning(verts_now)
                        if args.learn_skinning:
                            skin_opt = torch.optim.Adam([deformer.skin_logits],
                                                        lr=args.lr_deform)
                    arap_mesh = TriMesh(verts_now.clone(),
                                        triangles.get_triangle_indices.detach().clone())

        if deform_opt is not None:
            # per-vertex MLP has ~500k params; a 1.0 norm clip strangles it
            torch.nn.utils.clip_grad_norm_(deformer.parameters(),
                                           5.0 if args.deform_model == "mlp" else 1.0)
            deform_opt.step()
            deform_opt.zero_grad(set_to_none=True)
            if skin_opt is not None:
                skin_opt.step()
                skin_opt.zero_grad(set_to_none=True)
        triangles.optimizer.step()
        triangles.optimizer.zero_grad(set_to_none=True)

        if iteration % args.test_interval == 0 or iteration == args.total:
            test_psnr = evaluate(test_cams, triangles, pipe, background, deformer)
            n_tri = triangles.get_triangle_indices.shape[0]
            tqdm.write(f"[iter {iteration}] test PSNR {test_psnr:.3f} dB, {n_tri} tris "
                       f"({'dynamic' if deformer is not None else 'static warmup'})")
            if deformer is not None and test_psnr > best_psnr:
                best_psnr = test_psnr
                scene.save(iteration)
                torch.save({"kind": args.deform_model, "state": deformer.state_dict(),
                            "num_nodes": args.num_nodes, "iteration": iteration,
                            "test_psnr": test_psnr},
                           Path(dataset.model_path) / "deformer_best.pt")
                tqdm.write(f"  ^ new best, checkpoint saved (iteration_{iteration})")

    scene.save(args.total)
    if deformer is not None:
        out = Path(dataset.model_path) / "deformer.pt"
        torch.save({"kind": args.deform_model, "state": deformer.state_dict(),
                    "num_nodes": args.num_nodes}, out)
        print(f"deformer saved to {out}")


if __name__ == "__main__":
    main()
