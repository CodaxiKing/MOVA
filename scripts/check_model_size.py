"""Print download size and local cache completeness of Hugging Face repos WITHOUT downloading weights.

Also saves the file manifest (configs/model_manifests/) so later cache checks work offline.

Usage:
  python scripts/check_model_size.py Wan-AI/Wan2.1-VACE-1.3B-diffusers
  python scripts/check_model_size.py Wan-AI/Wan2.1-VACE-1.3B-diffusers --revision <sha> --verify-hashes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import constants as hf_constants  # noqa: E402

from common.hf_utils import load_or_fetch_manifest, repo_size, verify_cache  # noqa: E402
from common.resources import DOWNLOAD_MARGIN_GB, disk_free_gb  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--revision", default=None, help="commit SHA (default: main)")
    ap.add_argument("--verify-hashes", action="store_true", help="SHA-256 every cached file (slow)")
    args = ap.parse_args()
    free_gb = disk_free_gb(hf_constants.HF_HUB_CACHE)
    for repo in args.repos:
        manifest = load_or_fetch_manifest(repo, args.revision)
        cache = verify_cache(manifest, check_hashes=args.verify_hashes)
        print(f"== {repo}")
        print(f"   revision (exact)            {manifest['revision']}")
        for k, v in repo_size(repo, manifest["revision"]).items():
            print(f"   {k:28s} {v:8.2f} GB")
        print(f"   required by pipeline        {cache.expected_bytes / 1e9:8.2f} GB in {cache.files_expected} files")
        print(f"   cache                       {cache.summary()}")
        for path in cache.missing[:10]:
            print(f"     missing: {path}")
        need = cache.missing_bytes / 1e9 * 1.05 + (DOWNLOAD_MARGIN_GB if cache.missing_bytes else 0)
        status = "OK" if free_gb >= need else "INSUFFICIENT"
        print(f"   free disk (HF cache drive)  {free_gb:8.2f} GB, needed {need:.2f} GB -> {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
