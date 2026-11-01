# Patches against upstream Triangle Splatting+

`tsplus_windows.patch` applies to the Triangle Splatting+ repository
(github.com/trianglesplatting2/triangle-splatting2) and
`tsplus_submodules_windows.patch` to its rasterization and simple-knn
submodules. They contain two kinds of change.

1. The position-gradient repair in the rasterization backward: the interior
   barycentric gradient buffer, computed but never read upstream, is converted
   through the projection Jacobian into the vertex gradients, and the missing
   device-coordinate-to-pixel scale is restored on the window pathway. Forward
   rendering is unchanged and images are bit-identical.
2. Portability fixes for MSVC and CUDA 12.8 builds (type and operator
   portability, header guards for torch 2.9 and later, build-system flags),
   plus removal of an unused pytorch3d import and a non-interactive flag for
   the depth-prior download.

The upstream code these patches modify is governed by its own licenses (the
Triangle Splatting+ license and the Inria/MPII 3D Gaussian Splatting rasterizer
license for the rasterization submodule). The patches are provided for research
reproducibility and do not relicense any upstream code.
