"""Motion encoders: body / face / hands streams -> motion tokens aligned with Wan latent frames.

Pipeline (docs/architecture.md, "Modelo treinável planejado"):

    per-frame stream encoders (MLP)  ->  LatentFramePool (4k+1 video frames -> k+1 latent frames, Wan VAE rule)
    -> projection to the condition width  ->  temporal self-attention per token slot  ->  ConditionTokens

Each stream keeps its own tokens (separate by region, as in the public Kling principle): body K_b, face K_f and
one token per hand, per latent frame. A token is masked when its stream was absent in all frames of its group.
Shapes are checked by tests on CPU; nothing here has been trained.
"""

from __future__ import annotations

import torch
from torch import nn

from models.fusion.condition import ConditionTokens

from .features import BODY_DIM, FACE_DIM, HAND_DIM

VAE_TEMPORAL = 4


def latent_frames(num_frames: int) -> int:
    if num_frames < 1 or (num_frames - 1) % VAE_TEMPORAL:
        raise ValueError(f"num_frames must be 4k+1, got {num_frames}")
    return (num_frames - 1) // VAE_TEMPORAL + 1


class LatentFramePool(nn.Module):
    """(B, T=4k+1, K, D) -> (B, k+1, K, D): frame 0 alone, then groups of 4 (like the Wan causal VAE)."""

    def __init__(self, dim: int):
        super().__init__()
        self.first = nn.Linear(dim, dim)
        self.group = nn.Linear(VAE_TEMPORAL * dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, k, d = x.shape
        f = latent_frames(t)
        head = self.first(x[:, :1])
        rest = x[:, 1:].reshape(b, f - 1, VAE_TEMPORAL, k, d).permute(0, 1, 3, 2, 4).reshape(b, f - 1, k, -1)
        pooled = torch.cat([head, self.group(rest)], 1)
        m = torch.cat([mask[:, :1], mask[:, 1:].reshape(b, f - 1, VAE_TEMPORAL, *mask.shape[2:]).any(2)], 1)
        return pooled, m


def _mlp(i: int, o: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, o))


class BodyEncoder(nn.Module):
    def __init__(self, dim: int, tokens: int = 4, hidden: int = 256):
        super().__init__()
        self.tokens, self.dim = tokens, dim
        self.net = _mlp(33 * BODY_DIM, tokens * dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, 33, 6) -> (B, T, K, D)
        b, t = x.shape[:2]
        return self.net(x.reshape(b, t, -1)).reshape(b, t, self.tokens, self.dim)


class FaceEncoder(nn.Module):
    def __init__(self, dim: int, tokens: int = 2, hidden: int = 128):
        super().__init__()
        self.tokens, self.dim = tokens, dim
        self.net = _mlp(FACE_DIM, tokens * dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, 58) -> (B, T, K, D)
        b, t = x.shape[:2]
        return self.net(x).reshape(b, t, self.tokens, self.dim)


class HandEncoder(nn.Module):
    """Shared weights for both hands + a learned side embedding; one token per hand."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = _mlp(21 * HAND_DIM, dim, hidden)
        self.side = nn.Parameter(torch.zeros(2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, 2, 21, 4) -> (B, T, 2, D)
        b, t = x.shape[:2]
        return self.net(x.reshape(b, t, 2, -1)) + self.side


class TemporalAttention(nn.Module):
    """Self-attention over latent frames, independently per token slot (keeps streams separate)."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # (B, F, K, D), (B, F, K)
        b, f, k, d = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b * k, f, d)
        m = mask.permute(0, 2, 1).reshape(b * k, f)
        pad = ~m
        empty = pad.all(1)  # slot absent in every frame: attention would be undefined, skip it
        pad = pad & ~empty[:, None]
        h = self.norm(seq)
        out, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        out = torch.where(empty[:, None, None], torch.zeros_like(out), out)
        return x + out.reshape(b, k, f, d).permute(0, 2, 1, 3)


class MotionEncoder(nn.Module):
    """{body, face, hands} per video frame -> ConditionTokens per latent frame (width `dim`)."""

    STREAMS = ("body", "face", "hands")

    def __init__(self, dim: int = 256, body_tokens: int = 4, face_tokens: int = 2, heads: int = 4,
                 streams: tuple[str, ...] = STREAMS):
        super().__init__()
        unknown = set(streams) - set(self.STREAMS)
        if unknown or not streams:
            raise ValueError(f"streams must be a non-empty subset of {self.STREAMS}")
        self.dim, self.streams = dim, tuple(streams)
        self.encoders = nn.ModuleDict()
        self.pools = nn.ModuleDict()
        if "body" in streams:
            self.encoders["body"] = BodyEncoder(dim, body_tokens)
        if "face" in streams:
            self.encoders["face"] = FaceEncoder(dim, face_tokens)
        if "hands" in streams:
            self.encoders["hands"] = HandEncoder(dim)
        for s in self.streams:
            self.pools[s] = LatentFramePool(dim)
        self.stream_embed = nn.Parameter(torch.zeros(len(self.streams), dim))
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.temporal = TemporalAttention(dim, heads)

    def forward(self, inputs: dict[str, torch.Tensor]) -> ConditionTokens:
        tokens, masks = [], []
        for i, s in enumerate(self.streams):
            x = self.encoders[s](inputs[s])                          # (B, T, K, D)
            m = inputs[f"{s}_mask"]
            m = m if m.dim() == 3 else m[..., None].expand(*x.shape[:3])   # body/face: per frame -> per token
            pooled, pm = self.pools[s](x, m)
            tokens.append(pooled + self.stream_embed[i])
            masks.append(pm)
        x = self.temporal(self.proj(torch.cat(tokens, 2)), torch.cat(masks, 2))
        b, f, k, d = x.shape
        frame_index = torch.arange(f, device=x.device).repeat_interleave(k)
        return ConditionTokens(x.reshape(b, f * k, d), frame_index, torch.cat(masks, 2).reshape(b, f * k))
