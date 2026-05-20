# agfb-generators

`agfb-generators` provides batched PyTorch image generators for the Analytic Gradient Filter Benchmark. Each generator renders a synthetic intensity field together with its analytic horizontal and vertical gradients, so gradient filters can be checked against known references without finite-difference approximations.

The public Python package is `agfb_generators`.

**Installation**

Install the package from a local checkout with `uv`.

```bash
uv sync
```

After the repository is published, install it directly from GitHub.

```bash
uv pip install "agfb-generators @ git+https://github.com/j-vaught/agfb-generators.git"
```

**Quickstart**

```python
import torch

from agfb_generators import gaussian_blob

height = 128
width = 128

frame = gaussian_blob(height, width, scale_sigma=10.0, center_x=-8.0, center_y=6.0)

print(frame.I.shape)
print(frame.g.shape)

gx = frame.gx
gy = frame.gy
gmag = torch.sqrt(gx**2 + gy**2)
```

`frame.I` has shape `(B, H, W)` and stores image intensity. `frame.g` has shape `(B, 2, H, W)`. The gradient channel order is fixed. `frame.g[:, 0]` is $g_x$ and `frame.g[:, 1]` is $g_y$. The aliases `frame.gx` and `frame.gy` return the same channels as `(B, H, W)` tensors.

The quickstart uses scalar parameters, so `B` is `1`. The returned shapes are `torch.Size([1, 128, 128])` for `frame.I` and `torch.Size([1, 2, 128, 128])` for `frame.g`.

**Scalar And Batched Parameters**

Every public generator accepts Python scalars or one-dimensional tensors for numeric parameters. Scalars are broadcast across the batch. One-dimensional tensor parameters set the batch size and must agree on length.

```python
import math

import torch

from agfb_generators import smoothed_step

angles = torch.tensor([0.0, math.pi / 6.0, math.pi / 3.0])
edge_sigma = 4.0

frames = smoothed_step(96, 96, angle_rad=angles, edge_sigma=edge_sigma)

assert frames.I.shape == (3, 96, 96)
assert frames.g.shape == (3, 2, 96, 96)
```

If `device` is omitted and a tensor parameter is passed, the generator uses that tensor's device. If only scalar parameters are passed, CPU is used unless `device` is provided.

**Truth Helpers**

The truth helpers return masks and maps for localized scoring regions. Scalar inputs return unbatched `(H, W)` tensors. One-dimensional tensor inputs preserve the batch axis and return `(B, H, W)` tensors.

```python
import torch

from agfb_generators import junction_mask, vessel_crossing_truth

single_mask = junction_mask(96, 96, radius_px=10.0)
batched_mask = junction_mask(96, 96, radius_px=torch.tensor([8.0, 12.0]))

truth = vessel_crossing_truth(96, 96)

assert single_mask.shape == (96, 96)
assert batched_mask.shape == (2, 96, 96)
assert truth["centerline_mask"].shape == (96, 96)
```

`vessel_crossing_truth` and `vessel_bifurcation_truth` return dictionaries with `centerline_mask`, `branch_label`, `junction_mask`, and `radius_map`.

**Generator Gallery**

The notebook `agfb_generator_visual_check.ipynb` is a visual smoke test and gallery. It shows each generator, the returned gradient channels, the `gx` and `gy` aliases, and the truth-helper masks using the package notebook helpers.

**Generators**

| Function | Field type |
| --- | --- |
| `gaussian_blob` | Isotropic Gaussian peak. |
| `anisotropic_blob` | Rotated Gaussian peak with independent axis scales. |
| `gaussian_ridge` | One-dimensional Gaussian ridge. |
| `asymmetric_ridge` | Gaussian ridge with different widths on each side. |
| `curved_ridge` | Gaussian ridge bent by a parabolic centerline. |
| `curved_arc` | Smoothed circular boundary rendered as a curved edge. |
| `smoothed_step` | Gaussian-smoothed straight edge. |
| `finite_ramp` | Finite-width linear intensity ramp. |
| `smoothed_ramp` | Gaussian-smoothed finite-width ramp. |
| `mach_band` | Smoothed ramp with paired Mach-band shoulders. |
| `roof_profile` | Triangular roof intensity profile. |
| `smoothed_bar` | Soft bar from two opposite smoothed edges. |
| `sinusoid` | Single-frequency sinusoidal grating. |
| `chirp` | Oriented sinusoid with linearly changing frequency. |
| `gabor_packet` | Sinusoid inside a rotated Gaussian envelope. |
| `polynomial` | Two-dimensional polynomial surface. |
| `smoothed_l_junction` and `hard_l_junction` | L-shaped junction fields. |
| `smoothed_t_junction` and `hard_t_junction` | T-shaped junction fields. |
| `smoothed_x_junction` and `hard_x_junction` | X-shaped junction fields. |
| `smoothed_y_junction` and `hard_y_junction` | Y-shaped junction fields. |
| `vessel_crossing` | Two-vessel Gaussian-ridge crossing. |
| `vessel_bifurcation` | Smooth three-branch vessel bifurcation. |
| `junction_mask` | Circular junction-local truth mask. |
| `vessel_crossing_truth` | Geometric truth maps for a two-branch crossing. |
| `vessel_bifurcation_truth` | Geometric truth maps for a three-branch bifurcation. |

**License**

This project is released under the MIT License.
