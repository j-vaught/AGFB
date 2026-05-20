"""Roberts cross (2x2, half-pixel offset).

The classical Roberts cross approximates the diagonal derivatives
`diagonal_down` and `diagonal_up`, evaluated at pixel center
`(row + 0.5, column + 0.5)`. Projecting onto image axes gives the horizontal
and vertical gradients.

The half-pixel offset is intrinsic to Roberts and part of what the benchmark
measures. Output is `(batch, height, width)` via replicate padding on the right
and bottom.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from agfb_filters.base import check_input

_KERNEL_X = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]) / 2.0
_KERNEL_Y = torch.tensor([[-1.0, -1.0], [1.0, 1.0]]) / 2.0


def roberts(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    image = check_input(image)
    image_channels = image.unsqueeze(1)
    # replicate on the right and bottom so the 2x2 conv yields the original shape.
    padded_image = F.pad(image_channels, (0, 1, 0, 1), mode="replicate")
    kernel_x = _KERNEL_X.to(device=image.device, dtype=image.dtype).view(1, 1, 2, 2)
    kernel_y = _KERNEL_Y.to(device=image.device, dtype=image.dtype).view(1, 1, 2, 2)
    gradient_x = F.conv2d(padded_image, kernel_x).squeeze(1)
    gradient_y = F.conv2d(padded_image, kernel_y).squeeze(1)
    return gradient_x, gradient_y
