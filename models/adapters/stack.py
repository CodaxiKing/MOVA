"""MovaConditioning: every trainable MOVA module around a frozen Wan DiT, built from one config.

    motion inputs (models/motion/features.py) ─► MotionEncoder ─┐
    reference views (B, N, 3, H, W)          ─► IdentityEncoder ─┼─► ConditionFusion ─► MotionAdapter (hooks)
    pose video (optional, dense path)        ─► PoseGridEncoder ─────────────────────► MotionAdapter (dense)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from models.fusion.condition import ConditionFusion, ConditionTokens
from models.identity.encoder import IdentityEncoder
from models.motion.encoders import MotionEncoder

from .motion_adapter import MotionAdapter, PoseGridEncoder, latent_grid


@dataclass
class ConditioningConfig:
    cond_dim: int = 256
    adapter_dim: int = 128
    heads: int = 4
    body_tokens: int = 4
    face_tokens: int = 2
    identity_tokens: int = 8
    identity_patch: int = 16
    streams: tuple[str, ...] = ("body", "face", "hands")
    use_identity: bool = True
    use_dense_pose: bool = False
    blocks: list[int] | None = None  # None = every DiT block

    def to_dict(self) -> dict:
        return asdict(self)


class MovaConditioning(nn.Module):
    def __init__(self, hidden_dim: int, num_blocks: int, cfg: ConditioningConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ConditioningConfig()
        self.motion = MotionEncoder(cfg.cond_dim, cfg.body_tokens, cfg.face_tokens, cfg.heads, tuple(cfg.streams))
        self.identity = (IdentityEncoder(cfg.cond_dim, cfg.identity_tokens, cfg.identity_patch, cfg.heads)
                         if cfg.use_identity else None)
        self.fusion = ConditionFusion(cfg.cond_dim)
        self.pose = PoseGridEncoder(cfg.adapter_dim) if cfg.use_dense_pose else None
        self.adapter = MotionAdapter(hidden_dim, num_blocks, cfg.cond_dim, blocks=cfg.blocks, dim=cfg.adapter_dim,
                                     heads=cfg.heads, dense_dim=cfg.adapter_dim if cfg.use_dense_pose else None)

    @classmethod
    def for_transformer(cls, transformer: nn.Module, cfg: ConditioningConfig | None = None) -> "MovaConditioning":
        c = transformer.config
        return cls(c.num_attention_heads * c.attention_head_dim, len(transformer.blocks), cfg)

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[ConditionTokens, torch.Tensor | None]:
        motion = self.motion(batch)
        identity = None
        if self.identity is not None and "reference" in batch:
            identity = self.identity(batch["reference"], batch.get("reference_mask"))
        dense = self.pose(batch["pose_video"]) if self.pose is not None and "pose_video" in batch else None
        return self.fusion(motion=motion, identity=identity), dense

    def conditioned(self, batch: dict[str, torch.Tensor], num_frames: int, height: int, width: int,
                    reference_frames: int = 0):
        """reference_frames: latent frames the backbone prepends (WanVACEPipeline: one per reference image)."""
        cond, dense = self.encode(batch)
        grid = latent_grid(num_frames, height, width, reference_frames=reference_frames)
        return self.adapter.conditioned(cond, grid, dense, reference_frames)
