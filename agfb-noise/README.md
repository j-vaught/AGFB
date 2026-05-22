# agfb-noise

**Overview**

`agfb-noise` provides fast batched PyTorch noise models for AGFB image tensors. The package keeps noise generation on the input tensor device and supports scalar parameters or one-dimensional per-image parameter tensors with the batch axis in the first dimension.

**Basic Use**

The public functions accept floating-point tensors with any shape. When a parameter is a one-dimensional tensor, its length must match the image batch size.

```python
import torch

from agfb_noise import add_gaussian, add_noise

image = torch.zeros(4, 128, 128)
sigma = torch.tensor([0.005, 0.01, 0.02, 0.04])

noisy = add_gaussian(image, sigma=sigma, seed=0)
shot = add_noise(image.clamp_min(0.0), "poisson", peak=4096.0, seed=1)
```

The dispatcher supports `none`, `gaussian`, `uniform`, `salt_pepper`, `poisson`, `speckle`, and `rician`. Direct functions are available when call sites want to avoid dispatch overhead inside tight loops.

**Development**

Use the local workflow from this directory.

```sh
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run ty check .
uv run pytest
```
