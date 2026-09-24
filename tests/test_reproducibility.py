"""Pinned revisions, complete cache verification, resource preflight and provenance."""

import hashlib
import json

import pytest

from common import hf_utils
from common.hf_utils import load_or_fetch_manifest, verify_cache
from common.provenance import environment_fingerprint, lock_mismatches, package_versions
from common.resources import InsufficientResources, baseline_checks, enforce

REPO, SHA = "org/model", "a" * 40
REV = "r1"  # short snapshot name: Windows MAX_PATH in deep temp dirs


def _fake_cache(tmp_path, files: dict[str, bytes]):
    snap = tmp_path / "hub" / "models--org--model" / "snapshots" / REV
    (tmp_path / "hub" / "models--org--model" / "refs").mkdir(parents=True)
    for rel, data in files.items():
        (snap / rel).parent.mkdir(parents=True, exist_ok=True)
        (snap / rel).write_bytes(data)
    return tmp_path / "hub"


def _manifest(files: dict[str, bytes], lfs=("transformer/w.safetensors",)):
    out = []
    for rel, data in files.items():
        if rel in lfs:
            out.append({"path": rel, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "git_sha1": None})
        else:
            blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
            out.append({"path": rel, "size": len(data), "sha256": None, "git_sha1": blob})
    return {"repo_id": REPO, "revision": REV, "required_prefixes": ["x"], "files": out}


FILES = {"model_index.json": b'{"a": 1}', "transformer/w.safetensors": b"weights" * 100}


def test_cache_complete_only_when_every_file_matches(tmp_path):
    cache = _fake_cache(tmp_path, FILES)
    rep = verify_cache(_manifest(FILES), cache_dir=cache, check_hashes=True)
    assert rep.complete and rep.files_ok == 2 and rep.missing_bytes == 0


def test_model_index_alone_is_not_a_complete_cache(tmp_path):
    cache = _fake_cache(tmp_path, {"model_index.json": FILES["model_index.json"]})
    rep = verify_cache(_manifest(FILES), cache_dir=cache)
    assert not rep.complete
    assert rep.missing == ["transformer/w.safetensors"]
    assert rep.missing_bytes == len(FILES["transformer/w.safetensors"])


def test_truncated_and_corrupted_files_are_detected(tmp_path):
    truncated = {**FILES, "transformer/w.safetensors": b"weights"}
    rep = verify_cache(_manifest(FILES), cache_dir=_fake_cache(tmp_path / "a", truncated))
    assert not rep.complete and rep.size_mismatch[0]["path"] == "transformer/w.safetensors"

    same_size = {"model_index.json": b'{"b": 1}', "transformer/w.safetensors": b"WEIGHTS" * 100}
    cache = _fake_cache(tmp_path / "b", same_size)
    assert verify_cache(_manifest(FILES), cache_dir=cache).complete  # size-only check cannot see it
    rep = verify_cache(_manifest(FILES), cache_dir=cache, check_hashes=True)
    assert not rep.complete and set(rep.hash_mismatch) == set(FILES)  # LFS sha256 and git blob sha1


def test_pinned_manifest_is_reused_offline(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(repo, revision, prefixes):
        calls.append(revision)
        return {"repo_id": repo, "revision": SHA, "requested_revision": revision,
                "required_prefixes": list(prefixes), "files": [{"path": "model_index.json", "size": 1}]}

    monkeypatch.setattr(hf_utils, "fetch_manifest", fake_fetch)
    first = load_or_fetch_manifest(REPO, SHA, root=tmp_path)
    monkeypatch.setattr(hf_utils, "fetch_manifest", lambda *a: pytest.fail("network used for a pinned manifest"))
    assert load_or_fetch_manifest(REPO, SHA, root=tmp_path) == first
    assert calls == [SHA]


def test_pinned_revision_mismatch_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_utils, "fetch_manifest", lambda repo, rev, prefixes: {
        "repo_id": repo, "revision": "b" * 40, "required_prefixes": list(prefixes), "files": [{"path": "x", "size": 1}]})
    with pytest.raises(ValueError, match="pinned"):
        load_or_fetch_manifest(REPO, SHA, root=tmp_path)


def test_resource_preflight_refuses_known_shortfalls():
    kw = dict(hf_cache_dir=".", output_dir=".", offload="model", cuda=True)
    ok = baseline_checks(missing_download_bytes=0, need_text_encoder=False, ram_gb=8, vram_gb=6,
                         output_disk_gb=50, **kw)
    assert all(c.ok for c in ok) and not any("HF cache" in c.name for c in ok)
    enforce(ok)

    bad = baseline_checks(missing_download_bytes=19_040_000_000, need_text_encoder=True, ram_gb=1.0, vram_gb=6,
                          cache_disk_gb=17.8, output_disk_gb=17.8, **kw)
    failed = {c.name for c in bad if not c.ok}
    assert failed == {"disk (HF cache)", "RAM (text encoder)", "RAM (pipeline, offload=model)"}
    with pytest.raises(InsufficientResources, match="nothing was started"):
        enforce(bad)
    enforce(bad, override=True)


def test_provenance_and_lock_comparison(tmp_path):
    versions = package_versions()
    assert versions["torch"] and versions["numpy"]
    lock = tmp_path / "lock.txt"
    lock.write_text("torch==2.14.0\nnumpy==0.0.1  # wrong on purpose\n", encoding="utf-8")
    diff = lock_mismatches({"torch": "2.14.0+cu128", "numpy": "2.5.3", "diffusers": "0.40.0"}, lock)
    assert list(diff) == ["numpy"]  # build tag (+cu128/+cpu) is not a version difference
    env = environment_fingerprint()
    assert env["git"]["commit"] and "dirty" in env["git"]
    json.dumps(env)
