"""Can the MOVA adapter train on the real Wan2.1-VACE-1.3B on this GPU? Measures VRAM, time per step and loss.

Synthetic data only ("own" source): smooth latents at the real Wan latent shape, random motion tracks and the real
cached UMT5 prompt embeddings. It answers memory/speed/"does the loss move", never quality. The checkpoint is
discarded.

Usage:
  python scripts/train_feasibility.py --height 256 --width 256 --frames 17 --steps 6
  python scripts/train_feasibility.py --height 832 --width 464 --frames 17 --steps 6 --checkpointing
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402


def build_manifest(root: Path, prompt_file: Path, samples: int, frames: int, height: int, width: int) -> Path:
    from PIL import Image

    from training.smoke import _tracks

    rng = np.random.default_rng(0)
    pe = torch.load(prompt_file, map_location="cpu", weights_only=False)["prompt_embeds"][0].float()  # (512, 4096)
    f = (frames - 1) // 4 + 1
    entries = []
    for i in range(samples):
        d = root / f"clip{i}"
        (d / "motion").mkdir(parents=True, exist_ok=True)
        tr = _tracks(frames, rng)
        for k, name in (("body", "body_motion.pt"), ("face", "face_motion.pt"), ("hands", "hand_motion.pt")):
            torch.save({kk: (torch.from_numpy(vv) if isinstance(vv, np.ndarray) else vv) for kk, vv in tr[k].items()},
                       d / "motion" / name)
        g = torch.Generator().manual_seed(i)
        base = torch.linspace(-1, 1, f)[None, :, None, None] * torch.randn(16, 1, 1, 1, generator=g)
        torch.save(base.expand(16, f, height // 8, width // 8).clone(), d / "latents.pt")
        torch.save(pe, d / "prompt.pt")
        Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)).save(d / "ref.png")
        entries.append({"id": f"clip{i}", "split": "train", "identity_id": f"synthetic-{i}",
                        "latents": f"clip{i}/latents.pt", "prompt_embeds": f"clip{i}/prompt.pt",
                        "motion_dir": f"clip{i}/motion", "start_frame": 0, "references": [f"clip{i}/ref.png"]})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "num_frames": frames, "height": height, "width": width,
                                    "samples": entries}, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    from core.train import TrainRequest, run_training

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--checkpointing", action="store_true")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--prompt", default=None, help="cached prompt embeddings .pt (default: newest in checkpoints/embeds)")
    args = ap.parse_args()

    from common.config import PROJECT_ROOT

    prompt = Path(args.prompt) if args.prompt else max((PROJECT_ROOT / "checkpoints/embeds").glob("prompt_*.pt"),
                                                        key=lambda p: p.stat().st_mtime)
    with tempfile.TemporaryDirectory(prefix="mova-feasibility-") as tmp:
        root = Path(tmp)
        manifest = build_manifest(root, prompt, args.samples, args.frames, args.height, args.width)
        base = yaml.safe_load((PROJECT_ROOT / "configs/train_adapter.yaml").read_text(encoding="utf-8"))
        base.update(dataset=str(manifest), output_dir=str(root / "ckpt"), steps=args.steps, save_every=0,
                    forbid_identities_from=None)
        base["train"] = {**base["train"], "grad_accum": 1, "gradient_checkpointing": args.checkpointing}
        cfg = root / "train.yaml"
        cfg.write_text(yaml.safe_dump(base), encoding="utf-8")
        res = run_training(TrainRequest(config=str(cfg), precision=args.precision))
        res["checkpoint"] = "(discarded)"
    print(json.dumps({"height": args.height, "width": args.width, "frames": args.frames,
                      "checkpointing": args.checkpointing, **res}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
