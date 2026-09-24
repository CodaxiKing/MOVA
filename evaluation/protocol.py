"""Versioned benchmark inputs. A draft is not a populated benchmark."""

import hashlib
import json
import re
from pathlib import Path

import yaml

from common.config import resolve_path
from evaluation.video import validate_video


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_manifest(path):
    manifest = yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))
    return validate_manifest(manifest)


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Manifest schema_version must be 1")
    if not isinstance(manifest.get("version"), str) or not manifest["version"]:
        raise ValueError("Benchmark version is required")
    p = manifest["protocol"]
    for key in ("width", "height", "frames", "fps"):
        if type(p[key]) is not int or p[key] <= 0:
            raise ValueError(f"protocol.{key} must be a positive integer")
    if p["frames"] < 5 or p["frames"] % 4 != 1 or p["width"] % 16 or p["height"] % 16:
        raise ValueError("Protocol must match Wan dimensions (4k+1 frames, multiples of 16)")
    if type(p["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if not 0 < p["pck_threshold"] <= 1 or not 0 <= p["visibility"] <= 1:
        raise ValueError("Invalid evaluation thresholds")
    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark needs cases")
    ids = set()
    for case in cases:
        cid = case["id"]
        if not re.fullmatch(r"[a-z0-9_-]+", cid) or cid in ids:
            raise ValueError(f"Invalid or duplicate case ID: {cid}")
        ids.add(cid)
        if case.get("split") != "test":
            raise ValueError("Benchmark cases must be held-out test data")
    return manifest


def inspect_manifest(manifest):
    issues = []
    for case in manifest["cases"]:
        for key in ("category", "identity_id", "prompt", "reference_source", "motion_source",
                    "reference_license", "motion_license"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                issues.append(f"{case['id']}: missing {key}")
        for key in ("reference", "motion"):
            if not case.get(key) or not resolve_path(case[key]).is_file():
                issues.append(f"{case['id']}: missing file {key}")
    return issues


def freeze_manifest(manifest, destination):
    from PIL import Image

    validate_manifest(manifest)
    issues = inspect_manifest(manifest)
    if issues:
        raise ValueError("Cannot freeze incomplete benchmark:\n" + "\n".join(issues))
    p = manifest["protocol"]
    locked = json.loads(json.dumps(manifest))
    for case in locked["cases"]:
        with Image.open(resolve_path(case["reference"])) as image:
            image.verify()
        validate_video(resolve_path(case["motion"]), count=p["frames"], width=p["width"],
                       height=p["height"], fps=p["fps"])
        case["reference_sha256"] = sha256(resolve_path(case["reference"]))
        case["motion_sha256"] = sha256(resolve_path(case["motion"]))
    locked["benchmark_sha256"] = fingerprint(locked)
    # Exclusive creation: updating a frozen benchmark requires a new file/version.
    with Path(destination).open("x", encoding="utf-8") as stream:
        json.dump(locked, stream, indent=2, allow_nan=False)
    return locked


def verify_lock(path):
    locked = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    expected = locked.pop("benchmark_sha256")
    if fingerprint(locked) != expected:
        raise ValueError("Benchmark lock was modified")
    validate_manifest(locked)
    issues = inspect_manifest(locked)
    if issues:
        raise ValueError("\n".join(issues))
    for case in locked["cases"]:
        for key in ("reference", "motion"):
            if sha256(resolve_path(case[key])) != case[f"{key}_sha256"]:
                raise ValueError(f"{case['id']}: {key} bytes changed since freeze")
    locked["benchmark_sha256"] = expected
    return locked
