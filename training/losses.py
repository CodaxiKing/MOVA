"""Flow-matching objective of Wan2.1 with region weighting for face and hands.

Wan uses rectified flow: x_t = (1 - s) * x0 + s * noise, the network predicts the velocity v = noise - x0, and
sampling shifts s with `flow_shift` (s' = shift * s / (1 + (shift - 1) * s)). Timestep given to the DiT = s * 1000.

Regional weighting: faces and hands are small in the frame, so an unweighted MSE barely penalises them. A weight
map on the latent grid (face / hand boxes from the keypoints, max-combined) multiplies the per-element error; it is
normalised to mean 1 per sample so the loss scale does not depend on how big the regions are.
"""

from __future__ import annotations

import torch

NUM_TRAIN_TIMESTEPS = 1000


def shift_sigmas(u: torch.Tensor, shift: float = 3.0) -> torch.Tensor:
    return shift * u / (1 + (shift - 1) * u)


def sample_sigmas(batch: int, shift: float = 3.0, generator: torch.Generator | None = None,
                  eps: float = 1e-3) -> torch.Tensor:
    u = torch.rand(batch, generator=generator) * (1 - 2 * eps) + eps
    return shift_sigmas(u, shift)


def flow_matching_pair(x0: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(x_t, target velocity) for latents (B, C, F, H, W) and sigma (B,)."""
    s = sigma.view(-1, *([1] * (x0.dim() - 1))).to(x0.dtype)
    return (1 - s) * x0 + s * noise, noise - x0


def keypoint_region_mask(points: torch.Tensor, valid: torch.Tensor, latent_hw: tuple[int, int],
                         num_latent_frames: int, pad: float = 0.02) -> torch.Tensor:
    """Box around the valid points of each frame, rasterised on the latent grid and OR-pooled per latent frame.

    points (B, T, P, 2) normalised [0, 1] image coordinates; valid (B, T, P) bool; T = 4k+1 video frames.
    Returns (B, F, H, W) float mask in {0, 1}.
    """
    b, t, _, _ = points.shape
    h, w = latent_hw
    ys = (torch.arange(h, dtype=points.dtype) + 0.5) / h
    xs = (torch.arange(w, dtype=points.dtype) + 0.5) / w
    big = torch.tensor(1e6, dtype=points.dtype)
    x = torch.where(valid, points[..., 0], big)
    y = torch.where(valid, points[..., 1], big)
    x0, y0 = x.min(-1).values - pad, y.min(-1).values - pad
    x1 = torch.where(valid, points[..., 0], -big).max(-1).values + pad
    y1 = torch.where(valid, points[..., 1], -big).max(-1).values + pad
    any_valid = valid.any(-1)
    inside = ((xs[None, None, None, :] >= x0[..., None, None]) & (xs[None, None, None, :] <= x1[..., None, None]) &
              (ys[None, None, :, None] >= y0[..., None, None]) & (ys[None, None, :, None] <= y1[..., None, None]))
    inside = inside & any_valid[..., None, None]                                        # (B, T, H, W)
    if (t - 1) % 4:
        raise ValueError("T must be 4k+1")
    first = inside[:, :1]
    rest = inside[:, 1:].reshape(b, (t - 1) // 4, 4, h, w).any(2)
    out = torch.cat([first, rest], 1).float()
    if out.shape[1] != num_latent_frames:
        raise ValueError(f"{out.shape[1]} latent frames from T={t}, expected {num_latent_frames}")
    return out


def region_weights(shape: tuple[int, ...], regions: list[tuple[torch.Tensor, float]]) -> torch.Tensor:
    """Weight map (B, 1, F, H, W) = max over regions of weight*mask (1 elsewhere), mean-normalised per sample."""
    b, _, f, h, w = shape
    wmap = torch.ones(b, f, h, w)
    for mask, weight in regions:
        wmap = torch.maximum(wmap, 1 + (weight - 1) * mask)
    wmap = wmap / wmap.mean(dim=(1, 2, 3), keepdim=True)
    return wmap[:, None]


def flow_matching_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    err = (pred.float() - target.float()) ** 2
    if weights is not None:
        err = err * weights.to(err.dtype)
    return err.mean()
