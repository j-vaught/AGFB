# agfb-noise

**Overview**

`agfb-noise` provides fast batched PyTorch noise models for AGFB image tensors. The package mirrors the other AGFB repositories. Noise model definitions live in `agfb_noise/definitions/`, shared tensor and dispatch helpers live in `agfb_noise/helpers/`, and the package root exposes the public API from `agfb_noise/__init__.py`.

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
| `definitions/gaussian.py` | `gaussian`, `normal`, `awgn` | Additive independent Gaussian noise. |
| `definitions/local_variance.py` | `local_variance`, `localvar` | Gaussian noise with scalar, batched, or per-pixel variance. |
| `definitions/uniform.py` | `uniform` | Additive uniform noise over a centered interval. |
| `definitions/poisson.py` | `poisson`, `shot`, `shot_noise` | Poisson shot noise generated from nonnegative intensity. |
| `definitions/poisson_gaussian.py` | `poisson_gaussian`, `poisson-gaussian`, `pg` | Poisson shot noise plus Gaussian read noise. |
| `definitions/dark_current.py` | `dark_current` | Poisson dark-current background plus optional read noise. |
| `definitions/salt.py` | `salt` | Random high-valued impulse replacement. |
| `definitions/pepper.py` | `pepper` | Random low-valued impulse replacement. |
| `definitions/salt_pepper.py` | `salt_pepper`, `s&p` | Random low- and high-valued impulse replacement. |
| `definitions/random_impulse.py` | `random_impulse`, `rvin` | Random-valued impulse replacement. |
| `definitions/dead_pixel.py` | `dead_pixel`, `dead_pixels` | Dead- and hot-pixel defects. |
| `definitions/speckle.py` | `speckle` | Multiplicative Gaussian speckle. |
| `definitions/gamma_speckle.py` | `gamma_speckle` | Unit-mean gamma speckle for integer-look simulation. |
| `definitions/rician.py` | `rician`, `rice` | Magnitude image noise from two Gaussian channels. |
| `definitions/rayleigh.py` | `rayleigh` | Positive Rayleigh-distributed additive noise. |
| `definitions/quantization.py` | `quantization`, `quantize` | Uniform scalar quantization. |
| `definitions/fixed_pattern.py` | `fixed_pattern`, `fpn` | Pixelwise offset and gain nonuniformity. |
| `definitions/stripe.py` | `stripe`, `banding` | Row and column correlated offsets. |

**Notebooks**

Single-model notebooks are in `notebooks/`. Each notebook starts with the same synthetic 1024 x 1024 image, applies one noise model, reports compact tensor statistics, previews the clean image, noisy image, and residual, and times one hot-path call on the selected device.



**Development**

```sh
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run ty check .
uv run pytest
```
