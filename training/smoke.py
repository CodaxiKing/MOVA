"""CPU smoke training: tiny random Wan DiT + synthetic manifest. Proves the loop, not a model.

Used by `mova train --smoke` and tests. The synthetic clip's latents are a smooth function of the motion so
there is something learnable; the loss must go down on a fixed batch (overfit check).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def tiny_transformer(seed: int = 0):
    from diffusers import WanTransformer3DModel

    torch.manual_seed(seed)
    return WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=16,
                                 out_channels=16, text_dim=32, freq_dim=32, ffn_dim=32, num_layers=2,
                                 cross_attn_norm=True, qk_norm="rms_norm_across_heads", rope_max_seq_len=32)


def _tracks(t: int, rng: np.random.Generator) -> dict:
    kp = np.zeros((t, 33, 4), np.float32)
    kp[..., :2] = 0.5 + rng.normal(0, 0.1, (33, 2)) + np.linspace(0, 0.1, t)[:, None, None]
    kp[..., 3] = 1.0
    meta = {"format_version": 1, "fps": 16.0, "width": 32, "height": 32, "num_frames": t, "source": "synthetic"}
    face_lm = np.clip(0.5 + rng.normal(0, 0.03, (t, 478, 3)), 0, 1).astype(np.float32)
    hands = np.clip(0.5 + rng.normal(0, 0.05, (t, 2, 21, 3)), 0, 1).astype(np.float32)
    return {
        "body": {"meta": meta, "kp2d": kp, "kp3d_world": rng.normal(0, 0.3, (t, 33, 3)).astype(np.float32),
                 "present": np.ones(t, bool)},
        "face": {"meta": meta, "landmarks": face_lm, "blendshapes": rng.random((t, 52)).astype(np.float32),
                 "head_transform": np.repeat(np.eye(4, dtype=np.float32)[None], t, 0), "present": np.ones(t, bool)},
        "hands": {"meta": meta, "kp2d": hands, "kp3d_world": rng.normal(0, 0.05, (t, 2, 21, 3)).astype(np.float32),
                  "present": np.ones((t, 2), bool)},
    }


def make_synthetic_dataset(root: str | Path, samples: int = 2, num_frames: int = 9, size: int = 32,
                           seed: int = 0) -> Path:
    from PIL import Image

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    f = (num_frames - 1) // 4 + 1
    entries = []
    for i in range(samples):
        d = root / f"clip{i}"
        (d / "motion").mkdir(parents=True, exist_ok=True)
        tr = _tracks(num_frames, rng)
        for k, name in (("body", "body_motion.pt"), ("face", "face_motion.pt"), ("hands", "hand_motion.pt")):
            torch.save({kk: (torch.from_numpy(vv) if isinstance(vv, np.ndarray) else vv) for kk, vv in tr[k].items()},
                       d / "motion" / name)
        base = torch.linspace(-1, 1, f)[None, :, None, None] * torch.randn(16, 1, 1, 1, generator=torch.Generator().manual_seed(i))
        torch.save(base.expand(16, f, size // 8, size // 8).clone(), d / "latents.pt")
        torch.save(torch.randn(8, 32, generator=torch.Generator().manual_seed(100 + i)), d / "prompt.pt")
        Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(d / "ref.png")
        entries.append({"id": f"clip{i}", "split": "train", "identity_id": f"synthetic-{i}",
                        "latents": f"clip{i}/latents.pt", "prompt_embeds": f"clip{i}/prompt.pt",
                        "motion_dir": f"clip{i}/motion", "start_frame": 0, "references": [f"clip{i}/ref.png"]})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "num_frames": num_frames, "height": size, "width": size,
                                    "samples": entries}, indent=2), encoding="utf-8")
    return manifest
