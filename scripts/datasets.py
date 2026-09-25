"""Training data tools. Never downloads anything.

  python scripts/datasets.py list                                   # registry with license status
  python scripts/datasets.py validate --sources my_sources.yaml      # pairs, licenses, split leakage, benchmark overlap
  python scripts/datasets.py plan --source humanvid --use commercial # what using a source would require (no download)
  python scripts/datasets.py build --sources my_sources.yaml --out datasets/built/v1 --frames 17 --size 256 --vae tiny
      (--vae wan: the pinned Wan VAE from the local cache only; --prompt-embeds <.pt> to reuse cached UMT5 embeddings)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import resolve_path  # noqa: E402
from training.data_sources import ACCEPTED, load_registry, validate_sources  # noqa: E402


def _benchmark_ids(path):
    from core.train import benchmark_identities

    return benchmark_identities(path) if path else set()


def _vae(name):
    import torch

    if name == "tiny":
        from diffusers import AutoencoderKLWan

        torch.manual_seed(0)
        return AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                                temperal_downsample=[False, True, True]).eval()
    from diffusers import AutoencoderKLWan

    from models.backbones.wan_vace import VACE_REPO, VACE_REVISION

    return AutoencoderKLWan.from_pretrained(VACE_REPO, subfolder="vae", revision=VACE_REVISION, local_files_only=True,
                                            torch_dtype=torch.float32).eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("validate")
    p.add_argument("--sources", required=True)
    p.add_argument("--min-frames", type=int, default=17)
    p.add_argument("--benchmark", default="benchmark/v1.draft.yaml")
    p = sub.add_parser("plan")
    p.add_argument("--source", required=True)
    p.add_argument("--use", choices=sorted(ACCEPTED), required=True)
    p = sub.add_parser("build")
    p.add_argument("--sources", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=17)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--height", type=int, default=None, help="overrides --size (with --width)")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--min-body-detection", type=float, default=0.0, help="skip clips below this body detection rate")
    p.add_argument("--fps", type=float, default=16.0)
    p.add_argument("--vae", choices=["tiny", "wan"], default="wan")
    p.add_argument("--prompt-embeds", default=None, help=".pt with cached prompt embeddings (UMT5 is not run here)")
    p.add_argument("--benchmark", default="benchmark/v1.draft.yaml")
    args = ap.parse_args()

    if args.cmd == "list":
        for name, s in load_registry().items():
            print(f"{name:20s} {s['kind']:8s} {s['status']:14s} {s['license']}")
        return 0
    if args.cmd == "validate":
        rep = validate_sources(args.sources, min_frames=args.min_frames, forbid_identities=_benchmark_ids(args.benchmark))
        print(json.dumps(rep, indent=2))
        return 0 if rep["ok"] else 2
    if args.cmd == "plan":
        s = load_registry().get(args.source)
        if s is None:
            print(f"Unknown source {args.source!r}", file=sys.stderr)
            return 2
        ok = s["status"] in ACCEPTED[args.use]
        print(json.dumps({"source": args.source, "use": args.use, "allowed": ok, **s,
                          "next": "Download manually after reading the terms; record source_url and license per clip."
                          if ok else "Not usable for this purpose."}, indent=2))
        return 0 if ok else 2
    import torch

    from training.data_sources import build_training_manifest

    prompt = "a person moving"
    if args.prompt_embeds:
        embeds = torch.load(resolve_path(args.prompt_embeds), map_location="cpu", weights_only=True)
        if isinstance(embeds, dict):
            prompt = embeds.get("prompt") or prompt  # record the text the embeddings really encode
            embeds = embeds["prompt_embeds"][0]
    elif args.vae == "tiny":
        embeds = torch.zeros(8, 32)
    else:
        print("--prompt-embeds is required with --vae wan (encode once with the baseline's UMT5 cache)", file=sys.stderr)
        return 2
    path = build_training_manifest(args.sources, resolve_path(args.out), num_frames=args.frames, size=args.size,
                                   height=args.height, width=args.width, vae=_vae(args.vae),
                                   encode_prompt=lambda _: embeds, prompt=prompt, fps=args.fps,
                                   forbid_identities=_benchmark_ids(args.benchmark),
                                   min_body_detection=args.min_body_detection, log=lambda m: print(m, flush=True))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
