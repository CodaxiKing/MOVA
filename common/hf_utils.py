"""Hugging Face helpers: report repo size before any download, check local cache."""

from __future__ import annotations

from collections import defaultdict

from huggingface_hub import HfApi, try_to_load_from_cache


def repo_size(repo_id: str, revision: str | None = None) -> dict[str, float]:
    """Return {top_level_folder: GB, ..., 'TOTAL': GB} without downloading anything."""
    by_dir: dict[str, int] = defaultdict(int)
    for entry in HfApi().list_repo_tree(repo_id, recursive=True, revision=revision):
        size = getattr(entry, "size", None)
        if not size:
            continue
        top = entry.path.split("/")[0] if "/" in entry.path else "(root)"
        by_dir[top] += size
    out = {k: round(v / 1e9, 2) for k, v in sorted(by_dir.items(), key=lambda kv: -kv[1])}
    out["TOTAL"] = round(sum(by_dir.values()) / 1e9, 2)
    return out


def is_cached(repo_id: str, marker_file: str = "model_index.json") -> bool:
    return isinstance(try_to_load_from_cache(repo_id, marker_file), str)
