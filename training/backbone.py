"""Frozen backbones for adapter training. Never downloads (local_files_only); missing weights -> structured error.

- tiny          : tiny random WanTransformer3DModel (CPU smoke; no weights)
- tiny-vace     : tiny random WanVACETransformer3DModel with a null VACE control
- wan2.1-vace-1.3b : the transformer of the pinned Wan2.1-VACE-1.3B snapshot (ADR-008). UNVERIFIED: needs the
                     19 GB snapshot and a GPU. The VACE branch gets a zero control at scale 0, i.e. the adapter
                     trains on the plain Wan2.1 DiT path (VACE's forward requires control inputs; found by test).
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from common.errors import ModelWeightsMissingError

BACKBONES = ("tiny", "tiny-vace", "wan2.1-vace-1.3b")


def _null_vace_control(transformer) -> Callable[[torch.Tensor], dict[str, Any]]:
    c = transformer.config

    def extra(xt: torch.Tensor) -> dict[str, Any]:
        b, _, f, h, w = xt.shape
        return {"control_hidden_states": xt.new_zeros(b, c.vace_in_channels, f, h, w),
                "control_hidden_states_scale": xt.new_zeros(len(c.vace_layers))}
    return extra


def load_backbone(name: str, dtype: torch.dtype = torch.float32) -> tuple[torch.nn.Module, Callable | None]:
    if name == "tiny":
        from .smoke import tiny_transformer

        return tiny_transformer().to(dtype), None
    if name == "tiny-vace":
        from diffusers import WanVACETransformer3DModel

        torch.manual_seed(0)
        t = WanVACETransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
                                      in_channels=16, out_channels=16, text_dim=32, freq_dim=32, ffn_dim=32,
                                      num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
                                      rope_max_seq_len=32, vace_layers=[0], vace_in_channels=96).to(dtype)
        return t, _null_vace_control(t)
    if name == "wan2.1-vace-1.3b":
        from diffusers import WanVACETransformer3DModel

        from models.backbones.wan_vace import VACE_REPO, VACE_REVISION

        try:
            t = WanVACETransformer3DModel.from_pretrained(VACE_REPO, subfolder="transformer", revision=VACE_REVISION,
                                                          local_files_only=True, torch_dtype=dtype)
        except Exception as e:  # not cached: never download from the trainer
            raise ModelWeightsMissingError(f"{VACE_REPO}@{VACE_REVISION[:12]} transformer not in the local cache",
                                           hint="Download it first with `mova infer --model wan --allow-download` "
                                                "(19.04 GB, needs authorisation).") from e
        return t, _null_vace_control(t)
    raise ValueError(f"Unknown backbone {name!r}; known: {BACKBONES}")
