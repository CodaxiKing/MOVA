"""Training data intake: validate raw (reference, motion, identity) pairs, then build the training manifest.

Source manifest (YAML):
  schema_version: 1
  intended_use: research | commercial      # decides which registry statuses are acceptable
  clips:
  - id: clip-001
    identity_id: person-007                # same person in reference and motion video
    split: train | val
    motion: path/to/video.mp4             # the driving video; also the reconstruction target (self-supervision)
    references: [path/to/ref_front.png, ...]  # images of the SAME identity, not frames of this clip ideally
    source: humanvid                       # key in datasets/registry.yaml, or "own" for self-recorded material
    license: CC-BY-4.0                     # license of THIS item
    source_url: https://...               # where it came from (required unless source == own)

Checks (all reported together, nothing is fixed silently): schema, unique ids, files exist and decode, enough
frames, registry status vs intended use, per-item license present, identity leakage between splits, overlap with
benchmark identities.

build_training_manifest(): MediaPipe extraction + fixed-size windows + VAE latents + prompt embeddings -> the JSON
manifest read by training/dataset.py. Heavy parts are injected (VAE, prompt encoder) so the whole path is tested
on CPU with the tiny random VAE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from common.config import resolve_path

REGISTRY = "datasets/registry.yaml"
ACCEPTED = {"research": {"allowed", "research_only", "per_item", "own"}, "commercial": {"allowed", "per_item", "own"}}


def load_registry(path: str | Path = REGISTRY) -> dict[str, Any]:
    return (yaml.safe_load(resolve_path(path).read_text(encoding="utf-8")) or {}).get("sources", {})


def validate_sources(manifest: str | Path, *, registry: dict | None = None, min_frames: int = 17,
                     forbid_identities: set[str] | None = None, decode: bool = True) -> dict[str, Any]:
    from common.video_io import probe_video

    path = resolve_path(manifest)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    reg = load_registry() if registry is None else registry
    problems: list[str] = []
    if data.get("schema_version") != 1:
        problems.append("schema_version must be 1")
    use = data.get("intended_use")
    if use not in ACCEPTED:
        problems.append("intended_use must be 'research' or 'commercial'")
    clips = data.get("clips") or []
    if not clips:
        problems.append("no clips")
    seen, splits_by_identity = set(), {}
    root = path.parent
    for c in clips:
        cid = c.get("id", "?")
        for key in ("id", "identity_id", "split", "motion", "references", "source", "license"):
            if not c.get(key):
                problems.append(f"{cid}: missing {key}")
        if cid in seen:
            problems.append(f"{cid}: duplicate id")
        seen.add(cid)
        if c.get("split") not in ("train", "val"):
            problems.append(f"{cid}: split must be train or val (test data lives in benchmark/)")
        src = c.get("source")
        status = "own" if src == "own" else (reg.get(src) or {}).get("status")
        if src and status is None:
            problems.append(f"{cid}: source {src!r} is not in {REGISTRY}")
        elif use in ACCEPTED and status not in ACCEPTED[use]:
            problems.append(f"{cid}: source {src!r} is {status!r}, not allowed for {use} use")
        if src != "own" and not c.get("source_url"):
            problems.append(f"{cid}: source_url required for third-party material")
        if forbid_identities and c.get("identity_id") in forbid_identities:
            problems.append(f"{cid}: identity {c['identity_id']} belongs to the benchmark")
        splits_by_identity.setdefault(c.get("identity_id"), set()).add(c.get("split"))
        files = [c["motion"]] if c.get("motion") else []
        files += list(c.get("references") or [])
        for f in files:
            if not (root / f).is_file():
                problems.append(f"{cid}: file not found {f}")
        if decode and c.get("motion") and (root / c["motion"]).is_file():
            try:
                meta = probe_video(root / c["motion"])
                if meta.num_frames < min_frames:
                    problems.append(f"{cid}: {meta.num_frames} frames < {min_frames}")
            except Exception as e:  # unreadable video: reported, not fatal for the other clips
                problems.append(f"{cid}: cannot decode motion video ({type(e).__name__})")
    for identity, splits in splits_by_identity.items():
        if identity and len(splits) > 1:
            problems.append(f"identity {identity} appears in {sorted(splits)}: split leakage")
    return {"manifest": str(path), "clips": len(clips), "intended_use": use, "ok": not problems, "problems": problems}


def build_training_manifest(sources: str | Path, out_dir: str | Path, *, num_frames: int, size: int | None = None,
                            vae, encode_prompt: Callable[[str], Any], prompt: str = "a person moving",
                            fps: float = 16.0, stride: int | None = None, validate: bool = True,
                            forbid_identities: set[str] | None = None, height: int | None = None,
                            width: int | None = None, min_body_detection: float = 0.0,
                            log: Callable[[str], None] | None = None) -> Path:
    """Extract motion, cut windows of `num_frames` (4k+1), encode latents at height x width (default size x size),
    write manifest.json. Clips whose body detection rate is below `min_body_detection` are skipped and listed in
    the manifest under `skipped` (nothing is dropped silently)."""
    import torch
    from PIL import Image

    from common.video_io import read_video
    from inference.conditioning import letterbox
    from preprocessing.pipeline import ExtractionConfig, extract_motion

    from .precompute import encode_latents

    height, width = height or size, width or size
    if not height or not width:
        raise ValueError("give size, or height and width")
    if (num_frames - 1) % 4 or height % 16 or width % 16:
        raise ValueError("num_frames must be 4k+1 and height/width multiples of 16")
    src_path = resolve_path(sources)
    if validate:
        report = validate_sources(src_path, min_frames=num_frames, forbid_identities=forbid_identities)
        if not report["ok"]:
            raise ValueError("Source manifest is invalid:\n  " + "\n  ".join(report["problems"]))
    data = yaml.safe_load(src_path.read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    embeds = out / "prompt.pt"
    torch.save(torch.as_tensor(encode_prompt(prompt)).float().cpu(), embeds)
    stride = stride or num_frames
    samples, skipped = [], []
    for i, c in enumerate(data["clips"]):
        clip_dir = out / c["id"]
        motion_dir = clip_dir / "motion"
        video = src_path.parent / c["motion"]
        summary = extract_motion(video, motion_dir, ExtractionConfig(target_fps=fps, write_previews=False,
                                                                     write_openpose=False))
        rate = float(((summary or {}).get("body") or {}).get("detection_rate", 1.0))
        if rate < min_body_detection:
            skipped.append({"id": c["id"], "reason": f"body detection {rate:.2f} < {min_body_detection}"})
            if log:
                log(f"[{i + 1}/{len(data['clips'])}] skip {c['id']}: body detection {rate:.2f}")
            continue
        frames = [np.asarray(letterbox(Image.fromarray(f), width, height, fill=(0, 0, 0)))
                  for f in read_video(video, target_fps=fps)]
        refs = []
        for j, r in enumerate(c["references"]):
            with Image.open(src_path.parent / r) as im:
                dst = clip_dir / f"reference_{j}.png"
                im.convert("RGB").save(dst)
            refs.append(dst.relative_to(out).as_posix())
        for start in range(0, len(frames) - num_frames + 1, stride):
            lat = clip_dir / f"latents_{start:05d}.pt"
            torch.save(encode_latents(vae, frames[start:start + num_frames]).cpu(), lat)
            samples.append({"id": f"{c['id']}@{start}", "split": c["split"], "identity_id": c["identity_id"],
                            "latents": lat.relative_to(out).as_posix(), "prompt_embeds": "prompt.pt",
                            "motion_dir": motion_dir.relative_to(out).as_posix(), "start_frame": start,
                            "references": refs, "source": c["source"], "license": c["license"],
                            "source_url": c.get("source_url")})
        if log:
            log(f"[{i + 1}/{len(data['clips'])}] {c['id']}: body {rate:.2f}, {len(samples)} windows total")
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "num_frames": num_frames, "height": height, "width": width,
                                    "fps": fps, "prompt": prompt, "built_from": str(src_path),
                                    "samples": samples, "skipped": skipped}, indent=2), encoding="utf-8")
    return manifest
