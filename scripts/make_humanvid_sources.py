"""Source manifest (training/data_sources.py schema) for the downloaded HumanVid synthetic clips.

Reads datasets/raw/humanvid_synthetic/selection.json (the pinned file list), writes one reference image per clip
and datasets/raw/humanvid_synthetic/sources.yaml. Each clip is its own identity (the dataset does not say which
character appears in which clip), so train/val never share a clip. Validation = the *_video_10 folders.

Pilot limitation: the reference is a frame from the END of the same clip (training windows start at frame 0), not
an independent photo of the identity. Documented in docs/experiments; revisit when real multi-shot data exists.

Usage: python scripts/make_humanvid_sources.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "datasets/raw/humanvid_synthetic"
LICENSE = "Apache-2.0 (HF dataset card) / CC-BY-4.0 (GitHub README)"


def main() -> int:
    sel = json.loads((ROOT / "selection.json").read_text(encoding="utf-8"))
    refs = ROOT / "references"
    refs.mkdir(exist_ok=True)
    clips, missing = [], []
    for f in sel["files"]:
        video = ROOT / f["path"]
        if not video.is_file():
            missing.append(f["path"])
            continue
        folder, name = f["path"].split("/")[1], Path(f["path"]).stem
        cid = f"hv-{folder}-{name}"
        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 5))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            missing.append(f["path"])
            continue
        ref = refs / f"{cid}.png"
        cv2.imwrite(str(ref), frame)
        clips.append({"id": cid, "identity_id": cid, "split": "val" if folder.endswith("_10") else "train",
                      "motion": Path(f["path"]).as_posix(), "references": [ref.relative_to(ROOT).as_posix()],
                      "source": "humanvid_synthetic", "license": LICENSE,
                      "source_url": f"https://huggingface.co/datasets/{sel['repo']}/blob/{sel['revision']}/{f['path']}"})
    out = ROOT / "sources.yaml"
    out.write_text(yaml.safe_dump({"schema_version": 1, "intended_use": "research", "clips": clips},
                                  sort_keys=False), encoding="utf-8")
    n_val = sum(c["split"] == "val" for c in clips)
    print(f"{out}: {len(clips)} clips ({len(clips) - n_val} train, {n_val} val); missing {len(missing)}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
