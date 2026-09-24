"""Hugging Face helpers: sizes, pinned revisions, and complete cache verification.

A cache is "complete" only when every required file of the pinned revision is present with the
expected size (and, optionally, the expected LFS SHA-256). The file list is stored in a local
manifest so verification also works offline after the first online check.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from huggingface_hub import HfApi, try_to_load_from_cache

from .config import PROJECT_ROOT

MANIFEST_DIR = PROJECT_ROOT / "configs" / "model_manifests"  # versioned: pinned file lists
# Components a Diffusers pipeline actually loads; README/assets/examples are not required.
DIFFUSERS_REQUIRED = ("model_index.json", "scheduler/", "transformer/", "vae/", "tokenizer/", "text_encoder/")
FULL_SHA = re.compile(r"[0-9a-f]{40}")


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


def _is_required(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == p or (p.endswith("/") and path.startswith(p)) for p in prefixes)


def fetch_manifest(repo_id: str, revision: str | None, prefixes: tuple[str, ...] = DIFFUSERS_REQUIRED) -> dict:
    """Ask the Hub for the exact file list (size + LFS sha256) of a revision. Network, no download."""
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    files = []
    for s in info.siblings:
        if not _is_required(s.rfilename, prefixes):
            continue
        lfs = s.lfs
        sha = getattr(lfs, "sha256", None) if lfs is not None and not isinstance(lfs, dict) else (lfs or {}).get("sha256")
        # Small non-LFS files have no sha256; the git blob SHA-1 still identifies their exact bytes.
        files.append({"path": s.rfilename, "size": s.size, "sha256": sha,
                      "git_sha1": None if sha else getattr(s, "blob_id", None)})
    if not files:
        raise ValueError(f"No required files found in {repo_id}@{info.sha} for prefixes {prefixes}")
    return {"repo_id": repo_id, "revision": info.sha, "requested_revision": revision,
            "required_prefixes": list(prefixes), "files": sorted(files, key=lambda f: f["path"])}


def manifest_path(repo_id: str, revision: str, root: Path = MANIFEST_DIR) -> Path:
    return root / f"{repo_id.replace('/', '--')}@{revision}.json"


def load_or_fetch_manifest(repo_id: str, revision: str | None, prefixes: tuple[str, ...] = DIFFUSERS_REQUIRED,
                           root: Path = MANIFEST_DIR) -> dict:
    """Offline when the revision is a full commit SHA with a saved manifest; otherwise query the Hub once."""
    if revision and FULL_SHA.fullmatch(revision):
        path = manifest_path(repo_id, revision, root)
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("revision") == revision and manifest.get("required_prefixes") == list(prefixes):
                return manifest
    manifest = fetch_manifest(repo_id, revision, prefixes)
    if revision and FULL_SHA.fullmatch(revision) and manifest["revision"] != revision:
        raise ValueError(f"Hub returned revision {manifest['revision']} for pinned {revision}")
    path = manifest_path(repo_id, manifest["revision"], root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@dataclass
class CacheReport:
    repo_id: str
    revision: str
    complete: bool
    files_expected: int
    files_ok: int
    expected_bytes: int
    missing_bytes: int
    missing: list[str] = field(default_factory=list)
    size_mismatch: list[dict] = field(default_factory=list)
    hash_mismatch: list[str] = field(default_factory=list)
    hashes_checked: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        state = "COMPLETE" if self.complete else "INCOMPLETE"
        extra = f", {len(self.missing)} missing ({self.missing_bytes / 1e9:.2f} GB)" if self.missing else ""
        if self.size_mismatch:
            extra += f", {len(self.size_mismatch)} wrong size"
        if self.hash_mismatch:
            extra += f", {len(self.hash_mismatch)} wrong hash"
        return (f"{self.repo_id}@{self.revision[:12]}: {state} "
                f"({self.files_ok}/{self.files_expected} files{extra}; hashes checked: {self.hashes_checked})")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git_blob_sha1(path: str) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _hash_ok(local: str, f: dict) -> bool:
    if f.get("sha256"):
        return _sha256(local) == f["sha256"]
    if f.get("git_sha1"):
        return _git_blob_sha1(local) == f["git_sha1"]
    return True


def verify_cache(manifest: dict, cache_dir: str | Path | None = None, check_hashes: bool = False) -> CacheReport:
    """Check every required file of the pinned revision in the local HF cache."""
    repo, rev = manifest["repo_id"], manifest["revision"]
    rep = CacheReport(repo, rev, False, len(manifest["files"]), 0,
                      sum(f["size"] or 0 for f in manifest["files"]), 0, hashes_checked=check_hashes)
    for f in manifest["files"]:
        local = try_to_load_from_cache(repo, f["path"], revision=rev,
                                       cache_dir=str(cache_dir) if cache_dir else None)
        if not isinstance(local, str) or not os.path.isfile(local):
            rep.missing.append(f["path"])
            rep.missing_bytes += f["size"] or 0
            continue
        size = os.path.getsize(local)
        if f["size"] is not None and size != f["size"]:
            rep.size_mismatch.append({"path": f["path"], "expected": f["size"], "actual": size})
            rep.missing_bytes += f["size"]
            continue
        if check_hashes and not _hash_ok(local, f):
            rep.hash_mismatch.append(f["path"])
            rep.missing_bytes += f["size"] or 0
            continue
        rep.files_ok += 1
    rep.complete = rep.files_ok == rep.files_expected
    return rep
