# agfb-noise

**Overview**

`agfb-noise` provides fast batched PyTorch noise models for images in the Analytic Gradient Filter Benchmark (AGFB) project family. AGFB packages use synthetic and experimental image tensors to evaluate gradient filters, gradient-field metrics, and related image-processing workflows. This package focuses on controlled image corruption, so the same clean AGFB input can be evaluated under camera-like, impulse, coherent-imaging, and quantization noise.

Every direct model accepts a floating-point `torch.Tensor`, keeps random generation on the input tensor device, and returns a tensor with the same shape and dtype. The common AGFB convention is `(B, H, W)`, where `$B$` is batch size, `$H$` is image height, and `$W$` is image width. The functions also work on other image-shaped floating tensors when the requested parameter broadcasting is valid.

Scalar parameters apply to every pixel. One-dimensional tensor parameters broadcast across the leading batch dimension, which makes it possible to apply different noise strengths to each image in one call. Parameters that represent a full image, such as a local variance map, may also match the input tensor shape exactly. Output clipping is opt-in through `clamp=(low, high)`, so the caller decides whether simulated noise should be allowed to leave the nominal image range.

**Installation**

Install directly from the repository with `python -m pip install git+https://github.com/j-vaught/agfb-noise.git` or `uv add git+https://github.com/j-vaught/agfb-noise.git`.

PyTorch is the core runtime dependency. The project declares `torch>=2.4,<2.7` and `numpy>=2.0`. On Linux, the included `uv` configuration uses the PyTorch CUDA 12.4 wheel index for `torch`; on macOS and other non-Linux platforms it follows the normal PyTorch package resolution. If your environment needs a different CUDA runtime or a CPU-only PyTorch build, install the appropriate PyTorch wheel first and then install this package into the same environment.

**Basic Use**

Direct functions are the simplest entry point when the model is known at coding time.

```python
import torch

from agfb_noise import add_gaussian, add_poisson

image = torch.zeros(4, 128, 128)
sigma = torch.tensor([0.005, 0.01, 0.02, 0.04])

noisy = add_gaussian(image, sigma=sigma, seed=0, clamp=(0.0, 1.0))
shot = add_poisson(image.clamp_min(0.0), peak=4096.0, seed=1, clamp=(0.0, 1.0))
```

Name-based dispatch is useful when experiments are configured from files, notebooks, or parameter sweeps.

```python
from agfb_noise import NoiseSpec, add_noise, apply_noise_sequence

single = add_noise(image, "awgn", sigma=0.02, seed=5, clamp=(0.0, 1.0))

specs = (
    NoiseSpec("poisson_gaussian", {"peak": 4096.0, "read_sigma": 0.002}),
    NoiseSpec("quantization", {"levels": 4096}),
)

corrupted = apply_noise_sequence(image, specs, seed=10, clamp=(0.0, 1.0))
```

Randomness is local to the call when `seed` is provided. The helper creates a `torch.Generator` on the input device and does not reset the global PyTorch seed. Pass an explicit `generator` instead of `seed` when a larger experiment needs to own random-state progression. Passing both `seed` and `generator` is rejected.

**Noise Models**

| Model | Registered names | Typical use |
| --- | --- | --- |
| Additive Gaussian | `gaussian`, `normal`, `awgn` | Signal-independent read noise, electronic noise, and baseline denoising experiments. |
| Local-variance Gaussian | `local_variance`, `localvar`, `local_variance_gaussian` | Spatially varying or signal-dependent variance maps. |
| Additive uniform | `uniform` | Bounded perturbations, simple dither, and controlled robustness tests. |
| Poisson shot noise | `poisson`, `shot`, `shot_noise` | Photon-counting and electron-counting uncertainty. |
| Poisson-Gaussian | `poisson_gaussian`, `poisson-gaussian`, `pg` | Raw camera noise with shot noise plus read noise. |
| Dark-current noise | `dark_current` | Long-exposure and warm-sensor dark signal with optional read noise. |
| Salt impulse noise | `salt` | Bright fixed-valued impulse replacement. |
| Pepper impulse noise | `pepper` | Dark fixed-valued impulse replacement. |
| Salt-and-pepper impulse noise | `salt_pepper`, `s&p`, `salt-and-pepper` | Mixed bright and dark fixed-valued impulse replacement. |
| Random-valued impulse noise | `random_impulse`, `random_valued_impulse`, `rvin` | Sparse replacement by random intensities. |
| Dead and hot pixels | `dead_pixel`, `dead_pixels`, `defect_pixel` | Sensor defects and bad-pixel maps. |
| Gaussian speckle | `speckle` | Multiplicative speckle approximation. |
| Gamma speckle | `gamma_speckle`, `multilook_speckle` | Multilook coherent-imaging intensity speckle. |
| Rician magnitude noise | `rician`, `rice` | Magnitude magnetic resonance imaging and quadrature-channel amplitude data. |
| Rayleigh positive noise | `rayleigh` | Positive magnitude noise and no-signal magnitude regions. |
| Quantization | `quantization`, `quantize` | Finite-level digitization and low-bit-depth image effects. |
| Fixed-pattern noise | `fixed_pattern`, `fpn` | Pixelwise offset and gain nonuniformity. |
| Stripe noise | `stripe`, `banding`, `row_column` | Row- and column-correlated sensor artifacts. |

The fixed-valued impulse family shares one implementation. `add_salt_pepper(..., salt=False)` gives pepper-only replacement, `add_salt_pepper(..., pepper=False)` gives salt-only replacement, and the `add_salt` and `add_pepper` functions are convenience wrappers around those settings.

**Tensor And Parameter Rules**

Every model first validates that the image is a floating-point `torch.Tensor`. A scalar model parameter can be a Python number or tensor scalar. A one-dimensional parameter must match `image.shape[0]` and is reshaped to broadcast over the remaining dimensions. A full-image parameter must match the input tensor shape exactly.

The `clamp` argument accepts either `None` or a pair `(low, high)`, where either bound may be `None`. Leaving `clamp=None` preserves the physically simulated value, even if it falls outside `[0, 1]`. Passing `clamp=(0.0, 1.0)` is appropriate when the output should remain in a normalized display or model-input range.

**Notebooks**

The `notebooks/` directory contains one source-only notebook per shipped noise model. Each notebook starts from the same synthetic `1024 x 1024` AGFB image, explains the noise type, links to a relevant paper or reference page, applies one model, reports compact tensor summaries, previews the clean image, noisy image, and residual, and times one hot-path call on the selected device.

The notebooks target the `agfb-noise (.venv)` kernel. After `uv sync`, register that kernel with this command.

```sh
uv run python -m ipykernel install --sys-prefix --name agfb-noise --display-name "agfb-noise (.venv)"
```

When working in Visual Studio Code or JupyterLab, open the repository root and select the `agfb-noise (.venv)` kernel. The notebooks are intentionally saved without outputs so they stay small and easy to review.

**Development**

This repository uses `uv` for environment and dependency management. A normal local check is.

```sh
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run ty check .
```

Build the package artifacts with `uv build`. The wheel contains the importable Python package. The repository also includes notebooks for visual inspection and documentation, but notebook outputs are not committed.

**License**

`agfb-noise` is distributed under the MIT license.
