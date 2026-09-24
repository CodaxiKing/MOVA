"""Training clips: precomputed VAE latents + prompt embeddings + motion tracks + reference views.

Manifest (JSON): {"schema_version": 1, "num_frames": 17, "height": 256, "width": 256, "samples": [
  {"id": "...", "split": "train" | "val", "identity_id": "...", "latents": "<16,F,H/8,W/8 .pt>",
   "prompt_embeds": "<S,D .pt>", "motion_dir": "<dir with body/face/hand_motion.pt>", "start_frame": 0,
   "references": ["img.png", ...]}]}
Paths are relative to the manifest. Validation happens at construction so a broken sample fails before training,
and benchmark/test identities must never appear in a training manifest (`forbid_identities`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from models.motion.features import tracks_to_inputs

REQUIRED = ("id", "split", "identity_id", "latents", "prompt_embeds", "motion_dir", "references")


def _load_tracks(folder: Path) -> dict[str, dict[str, Any]]:
    names = {"body": "body_motion.pt", "face": "face_motion.pt", "hands": "hand_motion.pt"}
    out = {}
    for k, f in names.items():
        t = torch.load(folder / f, map_location="cpu", weights_only=True)
        out[k] = {kk: (vv.numpy() if torch.is_tensor(vv) else vv) for kk, vv in t.items()}
    return out


def region_points(tracks: dict[str, dict[str, Any]], start: int, num_frames: int) -> dict[str, np.ndarray]:
    """Normalised 2D face (every 8th of 478 landmarks) and hand points + masks, for the loss region weights."""
    sl = slice(start, start + num_frames)
    face = np.asarray(tracks["face"]["landmarks"], np.float32)[sl, ::8, :2]
    fmask = np.repeat(np.asarray(tracks["face"]["present"], bool)[sl, None], face.shape[1], 1)
    hands = np.asarray(tracks["hands"]["kp2d"], np.float32)[sl, :, :, :2].reshape(num_frames, 42, 2)
    hmask = np.repeat(np.asarray(tracks["hands"]["present"], bool)[sl], 21, 1)
    return {"face_xy": face, "face_xy_mask": fmask, "hands_xy": hands, "hands_xy_mask": hmask}


def _image(path: Path, size: int) -> torch.Tensor:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BICUBIC)
        return torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255


class MotionClipDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str = "train", *, max_references: int = 4,
                 reference_size: int = 224, forbid_identities: set[str] | None = None):
        self.path = Path(manifest)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("manifest schema_version must be 1")
        self.num_frames, self.height, self.width = data["num_frames"], data["height"], data["width"]
        if (self.num_frames - 1) % 4:
            raise ValueError("num_frames must be 4k+1")
        self.max_references, self.reference_size = max_references, reference_size
        root = self.path.parent
        f_lat = (self.num_frames - 1) // 4 + 1
        self.samples = []
        problems = []
        for s in data["samples"]:
            missing = [k for k in REQUIRED if k not in s]
            if missing:
                problems.append(f"{s.get('id', '?')}: missing {missing}")
                continue
            if s["split"] != split:
                continue
            if forbid_identities and s["identity_id"] in forbid_identities:
                problems.append(f"{s['id']}: identity {s['identity_id']} is reserved for evaluation")
                continue
            files = [root / s["latents"], root / s["prompt_embeds"], *(root / r for r in s["references"])]
            files += [root / s["motion_dir"] / n for n in ("body_motion.pt", "face_motion.pt", "hand_motion.pt")]
            absent = [str(f) for f in files if not f.is_file()]
            if absent:
                problems.append(f"{s['id']}: files not found {absent}")
                continue
            if not s["references"]:
                problems.append(f"{s['id']}: needs at least one reference image")
                continue
            shape = tuple(torch.load(root / s["latents"], map_location="cpu", weights_only=True).shape)
            want = (16, f_lat, self.height // 8, self.width // 8)
            if shape != want:
                problems.append(f"{s['id']}: latents {shape}, expected {want}")
                continue
            self.samples.append(s)
        if problems:
            raise ValueError("Invalid training manifest:\n  " + "\n  ".join(problems))
        if not self.samples:
            raise ValueError(f"No '{split}' samples in {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        s, root = self.samples[i], self.path.parent
        tracks = _load_tracks(root / s["motion_dir"])
        start = int(s.get("start_frame", 0))
        motion = tracks_to_inputs(tracks, start, self.num_frames)
        motion.update(region_points(tracks, start, self.num_frames))
        refs = s["references"][: self.max_references]
        images = torch.zeros(self.max_references, 3, self.reference_size, self.reference_size)
        mask = torch.zeros(self.max_references, dtype=torch.bool)
        for j, r in enumerate(refs):
            images[j], mask[j] = _image(root / r, self.reference_size), True
        item = {k: torch.from_numpy(v) for k, v in motion.items()}
        item.update(latents=torch.load(root / s["latents"], map_location="cpu", weights_only=True).float(),
                    prompt_embeds=torch.load(root / s["prompt_embeds"], map_location="cpu", weights_only=True).float(),
                    reference=images, reference_mask=mask)
        return item


def collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}
