"""Precompute what the frozen parts produce, so training never loads them (ADR-004):
VAE latents of each training clip and UMT5 prompt embeddings. Latents are normalised exactly like
diffusers' WanVACEPipeline: (z - latents_mean) / latents_std.
"""

from __future__ import annotations

import numpy as np
import torch


def video_to_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    """RGB uint8 frames (T, H, W, 3) -> (1, 3, T, H, W) in [-1, 1]."""
    x = torch.from_numpy(np.stack(frames)).float() / 127.5 - 1
    return x.permute(3, 0, 1, 2)[None]


@torch.no_grad()
def encode_latents(vae, frames: list[np.ndarray]) -> torch.Tensor:
    """(16, F, H/8, W/8) normalised latents with the deterministic posterior mode (as the pipeline does)."""
    if (len(frames) - 1) % 4:
        raise ValueError("clip length must be 4k+1 frames")
    x = video_to_tensor(frames).to(vae.dtype)
    z = vae.encode(x).latent_dist.mode().float()
    c = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, c, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, c, 1, 1, 1)
    return ((z - mean) / std)[0]


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Inverse of encode_latents, for checks: (16, F, h, w) -> (3, T, H, W) in [-1, 1]."""
    c = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, c, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, c, 1, 1, 1)
    return vae.decode((latents[None] * std + mean).to(vae.dtype)).sample[0].float()
