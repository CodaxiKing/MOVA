"""Print the download size of Hugging Face repos WITHOUT downloading them.

Usage: python scripts/check_model_size.py Wan-AI/Wan2.1-VACE-1.3B-diffusers [more/repos ...]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.hf_utils import is_cached, repo_size  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    args = ap.parse_args()
    free_gb = shutil.disk_usage(Path.home()).free / 1e9
    for repo in args.repos:
        sizes = repo_size(repo)
        print(f"== {repo}  (cached locally: {is_cached(repo)})")
        for k, v in sizes.items():
            print(f"   {k:28s} {v:8.2f} GB")
        status = "OK" if free_gb > sizes["TOTAL"] * 1.2 else "INSUFFICIENT"
        print(f"   free disk (home drive): {free_gb:.1f} GB -> {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
