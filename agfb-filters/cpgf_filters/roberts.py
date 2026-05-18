"""Roberts cross (2x2, half-pixel offset).

The classical Roberts cross approximates the diagonal derivatives
`d1 = I[i+1, j+1] - I[i, j]` and `d2 = I[i+1, j] - I[i, j+1]`, evaluated at
pixel center `(i + 0.5, j + 0.5)`. Projecting onto `(x, y)`:
    g_x = (d1 - d2) / 2 = ([[-1, 1], [-1, 1]] / 2) ⊙ I[i:i+2, j:j+2]
    g_y = (d1 + d2) / 2 = ([[-1, -1], [1, 1]] / 2) ⊙ I[i:i+2, j:j+2]

The half-pixel offset is intrinsic to Roberts and part of what the benchmark
measures. Output is `(B, H, W)` via replicate padding on the right and bottom.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cpgf_filters.base import check_input

_KX = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]) / 2.0
_KY = torch.tensor([[-1.0, -1.0], [1.0, 1.0]]) / 2.0


def roberts(I: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    I = check_input(I)
    x = I.unsqueeze(1)
    # replicate on the right and bottom so the 2x2 conv yields the original (H, W)
    x = F.pad(x, (0, 1, 0, 1), mode="replicate")
    kx = _KX.to(device=I.device, dtype=I.dtype).view(1, 1, 2, 2)
    ky = _KY.to(device=I.device, dtype=I.dtype).view(1, 1, 2, 2)
    gx = F.conv2d(x, kx).squeeze(1)
    gy = F.conv2d(x, ky).squeeze(1)
    return gx, gy
