# agfb-noise

**Overview**

`agfb-noise` provides fast batched PyTorch noise models for AGFB image tensors. The package mirrors the other AGFB repositories. Each noise model lives in its own module file, shared tensor utilities live in `agfb_noise/base.py`, and name-based execution lives in `agfb_noise/runtime/`.

All direct model functions keep random generation on the input tensor device. Scalar parameters broadcast over the full image. One-dimensional tensor parameters broadcast across the first axis, so a single call can apply different noise levels to each image in a batch.

**Basic Use**

```python
import torch

from agfb_noise import add_gaussian, add_noise

image = torch.zeros(4, 128, 128)
sigma = torch.tensor([0.005, 0.01, 0.02, 0.04])

noisy = add_gaussian(image, sigma=sigma, seed=0)
shot = add_noise(image.clamp_min(0.0), "poisson", peak=4096.0, seed=1)
```

Configured calls use `NoiseSpec`.

```python
from agfb_noise import NoiseSpec, apply_noise_sequence

specs = (
    NoiseSpec("poisson_gaussian", {"peak": 4096.0, "read_sigma": 0.002}),
    NoiseSpec("quantization", {"levels": 4096}),
)

corrupted = apply_noise_sequence(image, specs, seed=10, clamp=(0.0, 1.0))
```

**Noise Models**

| Module | Registered names | Model |
| --- | --- | --- |
| `gaussian.py` | `gaussian`, `normal`, `awgn` | Additive independent Gaussian noise. |
| `local_variance.py` | `local_variance`, `localvar` | Gaussian noise with scalar, batched, or per-pixel variance. |
| `uniform.py` | `uniform` | Additive uniform noise over a centered interval. |
| `poisson.py` | `poisson`, `shot`, `shot_noise` | Poisson shot noise generated from nonnegative intensity. |
| `poisson_gaussian.py` | `poisson_gaussian`, `poisson-gaussian`, `pg` | Poisson shot noise plus Gaussian read noise. |
| `dark_current.py` | `dark_current` | Poisson dark-current background plus optional read noise. |
| `salt.py` | `salt` | Random high-valued impulse replacement. |
| `pepper.py` | `pepper` | Random low-valued impulse replacement. |
| `salt_pepper.py` | `salt_pepper`, `s&p` | Random low- and high-valued impulse replacement. |
| `random_impulse.py` | `random_impulse`, `rvin` | Random-valued impulse replacement. |
| `dead_pixel.py` | `dead_pixel`, `dead_pixels` | Dead- and hot-pixel defects. |
| `speckle.py` | `speckle` | Multiplicative Gaussian speckle. |
| `gamma_speckle.py` | `gamma_speckle` | Unit-mean gamma speckle for integer-look simulation. |
| `rician.py` | `rician`, `rice` | Magnitude image noise from two Gaussian channels. |
| `rayleigh.py` | `rayleigh` | Positive Rayleigh-distributed additive noise. |
| `quantization.py` | `quantization`, `quantize` | Uniform scalar quantization. |
| `fixed_pattern.py` | `fixed_pattern`, `fpn` | Pixelwise offset and gain nonuniformity. |
| `stripe.py` | `stripe`, `banding` | Row and column correlated offsets. |

**Notebooks**

Single-model notebooks are in `notebooks/noise`. Each notebook starts with the same synthetic 1024 x 1024 image, applies one noise model, reports compact tensor statistics, previews the clean image, noisy image, and residual, and times one hot-path call on the selected device.

The notebooks target the uv-managed `agfb-noise (.venv)` kernel. Register it once after `uv sync`.

```sh
uv run python -m ipykernel install --sys-prefix --name agfb-noise --display-name "agfb-noise (.venv)"
```

In Visual Studio Code, open this repository folder directly so `.vscode/settings.json` can point the Python and Jupyter extensions at `.venv/bin/python`. If the kernel list is already open, reload the window after running the command above.

**Literature Basis**

The shipped set starts with the practical image-noise modes used by common denoising tools. The `random_noise` API in scikit-image covers additive Gaussian noise, local-variance Gaussian noise, Poisson noise, salt, pepper, salt-and-pepper, and multiplicative speckle, which is a useful baseline vocabulary for image-processing experiments. The impulse models are also supported by image-denoising surveys that distinguish fixed-valued impulse noise from random-valued impulse noise.

For camera-like raw data, the package includes Poisson, Poisson-Gaussian, dark-current, quantization, fixed-pattern, stripe, and defect-pixel models. Foi et al. model raw sensor data with a signal-dependent Poissonian component and a signal-independent Gaussian component. EMVA 1288 describes image-sensor characterization around photon transfer, temporal dark noise, dark current, quantization noise, spatial nonuniformity, row and column effects, and defect pixels.

For coherent and medical imaging, the package includes multiplicative speckle, gamma speckle, Rician, and Rayleigh models. Goodman describes coherent speckle as an interference-driven granular irradiance pattern. Gudbjartsson and Patz show that magnitude magnetic resonance images are governed by a Rician distribution, with Rayleigh behavior in pure-noise magnitude regions.

The bibliography is centralized in `references.bib`.

**Development**

```sh
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run ty check .
uv run pytest
```
