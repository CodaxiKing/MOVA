"""Sequential, bounded benchmark evaluation and comparison."""

import importlib.metadata
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from common.config import PROJECT_ROOT, resolve_path
from common.env import detect_hardware
from common.experiment import ExperimentRun
from evaluation.motion import METRIC_VERSION, compare_tracks, load_tracks
from evaluation.protocol import fingerprint, sha256, verify_lock
from evaluation.video import validate_video
from preprocessing.mp_models import MODEL_DIR
from preprocessing.pipeline import ExtractionConfig, extract_motion


def evaluator_signature(protocol):
    models = {name: sha256(MODEL_DIR / f"{name}.task")
              for name in ("pose_landmarker_full", "face_landmarker", "hand_landmarker")}
    code = {str(path.relative_to(PROJECT_ROOT)): sha256(path)
            for folder in ("preprocessing", "evaluation")
            for path in sorted((PROJECT_ROOT / folder).rglob("*.py"))}
    return fingerprint({"models": models, "code": code, "metric_version": METRIC_VERSION,
                        "protocol": protocol, "packages": {
                            name: importlib.metadata.version(name)
                            for name in ("mediapipe", "numpy", "opencv-python", "torch")}})


def evaluate_benchmark(lock_path, outputs, *, label, generation_contract, artifacts_root="outputs/evaluation",
                       generation_index=None):
    """Evaluate every case; missing/invalid cases cannot silently disappear."""
    locked = verify_lock(lock_path)
    p = locked["protocol"]
    if not isinstance(generation_contract, dict) or not generation_contract:
        raise ValueError("A non-empty generation contract is required")
    if generation_index is not None:
        if (generation_index["benchmark_sha256"] != locked["benchmark_sha256"] or
                generation_index["generation_contract"] != generation_contract):
            raise ValueError("Generation provenance differs from evaluation inputs")
    signature = evaluator_signature(p)  # Requires local extractor weights; never downloads here.
    run = ExperimentRun("benchmark", artifacts_root=resolve_path(artifacts_root),
                        run_id=f"benchmark-{uuid.uuid4().hex[:12]}")
    cfg = ExtractionConfig(write_previews=False, write_openpose=False)
    report = {"schema_version": 1, "label": label, "benchmark_sha256": locked["benchmark_sha256"],
              "evaluator_sha256": signature, "generation_contract": generation_contract,
              "protocol": p, "cases": [], "quality_status": "NOT_ESTABLISHED"}
    run.log(dataset=locked["version"], benchmark_sha256=locked["benchmark_sha256"],
            hardware=detect_hardware().to_dict(), extraction_config=asdict(cfg),
            evaluator_sha256=signature, generation_contract=generation_contract)
    try:
        for case in locked["cases"]:
            cid = case["id"]
            item = {"id": cid, "category": case["category"], "status": "failed"}
            t0 = time.perf_counter()
            try:
                video = resolve_path(outputs) / f"{cid}.mp4"
                item["integrity"] = validate_video(video, count=p["frames"], width=p["width"],
                                                   height=p["height"], fps=p["fps"])
                item["output_sha256"] = sha256(video)
                if generation_index is not None:
                    provenance = generation_index["cases"][cid]
                    if provenance["output_sha256"] != item["output_sha256"]:
                        raise ValueError("Generated video changed since generation record")
                    item["generation"] = provenance
                else:
                    item["generation"] = {"provenance": "user_declared", "generation_time_s": None,
                                          "seconds_per_frame": None, "vram_peak_gb": None}
                root = run.artifacts_dir / cid
                extract_motion(resolve_path(case["motion"]), root / "driver", cfg)
                extract_motion(video, root / "generated", cfg)
                item["metrics"] = compare_tracks(load_tracks(root / "driver"), load_tracks(root / "generated"),
                                                 pck_threshold=p["pck_threshold"], visibility=p["visibility"])
                item["status"] = "evaluated"
            except Exception as error:
                item["error"] = f"{type(error).__name__}: {error}"
            item["evaluation_time_s"] = time.perf_counter() - t0
            report["cases"].append(item)
        report["status"] = "complete" if all(c["status"] == "evaluated" for c in report["cases"]) else "partial"
        path = run.artifacts_dir / "report.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        run.finish("success" if report["status"] == "complete" else "failed", report=str(path))
        return path, report
    except Exception as error:
        run.finish("failed", error=repr(error))
        raise


def compare_reports(baseline, candidate):
    for key in ("schema_version", "benchmark_sha256", "evaluator_sha256", "protocol", "generation_contract"):
        if baseline.get(key) != candidate.get(key) or key not in baseline:
            raise ValueError(f"Non-comparable reports: {key}")
    if baseline.get("status") != "complete" or candidate.get("status") != "complete":
        raise ValueError("Both reports must contain every evaluated case")
    before = {c["id"]: c for c in baseline["cases"]}
    after = {c["id"]: c for c in candidate["cases"]}
    if before.keys() != after.keys() or not before or len(before) != len(baseline["cases"]) or len(after) != len(candidate["cases"]):
        raise ValueError("Case sets differ or contain duplicates")
    deltas = []
    for cid in sorted(before):
        if before[cid]["status"] != "evaluated" or after[cid]["status"] != "evaluated":
            raise ValueError("Cannot compare failed cases")
        changes = {}
        for group, fields in {"body": ("pck", "paired_coverage", "mean_error_paired", "acceleration_error_paired"),
                              "hands": ("pck", "paired_coverage", "mean_error_paired", "acceleration_error_paired"),
                              "face": ("paired_coverage", "blendshape_mae_paired")}.items():
            for field in fields:
                a, b = before[cid]["metrics"][group][field], after[cid]["metrics"][group][field]
                changes[f"{group}.{field}"] = None if a is None or b is None else b - a
        deltas.append({"id": cid, "candidate_minus_baseline": changes})
    return {"status": "REQUIRES_REVIEW", "cases": deltas,
            "note": "No automatic promotion: inspect coverage, identity and visual quality; lower acceleration alone is not better."}
