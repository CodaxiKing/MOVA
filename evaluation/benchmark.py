"""Sequential, bounded benchmark evaluation and comparison."""

import importlib.metadata
import json
import os
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


def _write_json_atomic(path, data):
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _load_resume(resume_from, report):
    """Previously evaluated cases that may be reused; refuses if the conditions differ."""
    folder = resolve_path(resume_from)
    for name in ("report.partial.json", "report.json"):
        if (folder / name).is_file():
            previous = json.loads((folder / name).read_text(encoding="utf-8"))
            break
    else:
        raise FileNotFoundError(f"No report.partial.json or report.json in {folder}")
    for key in ("schema_version", "benchmark_sha256", "evaluator_sha256", "protocol", "generation_contract"):
        if previous.get(key) != report[key]:
            raise ValueError(f"Resume refused: {key} differs from the interrupted evaluation")
    return folder, {c["id"]: c for c in previous["cases"] if c["status"] == "evaluated"}


def evaluate_benchmark(lock_path, outputs, *, label, generation_contract, artifacts_root="outputs/evaluation",
                       generation_index=None, resume_from=None, build_review_page=True):
    """Evaluate every case; missing/invalid cases cannot silently disappear.

    State is saved to report.partial.json after each case; `resume_from` reuses cases already evaluated
    under identical conditions whose generated video is byte-identical.
    """
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
              "protocol": p, "cases": [], "quality_status": "NOT_ESTABLISHED", "status": "in_progress"}
    reusable, resumed_dir = {}, None
    if resume_from is not None:
        resumed_dir, reusable = _load_resume(resume_from, report)
        report["resumed_from"] = str(resumed_dir)
    run.log(dataset=locked["version"], benchmark_sha256=locked["benchmark_sha256"],
            hardware=detect_hardware().to_dict(), extraction_config=asdict(cfg),
            evaluator_sha256=signature, generation_contract=generation_contract,
            resumed_from=report.get("resumed_from"))
    partial = run.artifacts_dir / "report.partial.json"
    try:
        for case in locked["cases"]:
            cid = case["id"]
            video = resolve_path(outputs) / f"{cid}.mp4"
            previous = reusable.get(cid)
            if previous is not None and video.is_file() and sha256(video) == previous.get("output_sha256"):
                report["cases"].append({**previous, "reused_from": str(resumed_dir)})
                _write_json_atomic(partial, report)
                continue
            root = run.artifacts_dir / cid
            item = {"id": cid, "category": case["category"], "status": "failed", "artifacts": str(root),
                    "inputs": {"reference": case["reference"], "motion": case["motion"], "output": str(video)}}
            t0 = time.perf_counter()
            try:
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
                extract_motion(resolve_path(case["motion"]), root / "driver", cfg)
                extract_motion(video, root / "generated", cfg)
                item["metrics"] = compare_tracks(load_tracks(root / "driver"), load_tracks(root / "generated"),
                                                 pck_threshold=p["pck_threshold"], visibility=p["visibility"])
                item["status"] = "evaluated"
            except Exception as error:
                item["error"] = f"{type(error).__name__}: {error}"
            item["evaluation_time_s"] = time.perf_counter() - t0
            report["cases"].append(item)
            _write_json_atomic(partial, report)
        report["status"] = "complete" if all(c["status"] == "evaluated" for c in report["cases"]) else "partial"
        path = run.artifacts_dir / "report.json"
        _write_json_atomic(path, report)
        partial.unlink(missing_ok=True)
        review = None
        if build_review_page:
            from evaluation.review import build_review

            try:
                review = str(build_review(path))
            except Exception as error:  # the report is the source of truth; a review failure must not lose it
                review = f"FAILED: {type(error).__name__}: {error}"
        run.finish("success" if report["status"] == "complete" else "failed", report=str(path), review=review)
        return path, report
    except BaseException as error:  # includes KeyboardInterrupt: record where to resume from
        run.finish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=repr(error),
                   resume_from=str(run.artifacts_dir))
        raise


COMPARED_FIELDS = {
    "body": ("pck", "paired_coverage", "mean_error_paired", "acceleration_error_paired"),
    "hands": ("pck", "paired_coverage", "mean_error_paired", "acceleration_error_paired"),
    "face": ("paired_coverage", "blendshape_mae_paired"),
    "trajectory": ("paired_coverage", "trajectory_error", "final_displacement_error", "scale_log_error",
                   "absolute_root_error"),
    "head_rotation": ("paired_coverage", "geodesic_error_deg", "relative_geodesic_error_deg"),
    "body_orientation": ("paired_coverage", "yaw_error_deg", "relative_yaw_error_deg"),
}


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
        for group, fields in COMPARED_FIELDS.items():
            for field in fields:
                a = before[cid]["metrics"].get(group, {}).get(field)
                b = after[cid]["metrics"].get(group, {}).get(field)
                changes[f"{group}.{field}"] = None if a is None or b is None else b - a
        deltas.append({"id": cid, "candidate_minus_baseline": changes})
    return {"status": "REQUIRES_REVIEW", "cases": deltas,
            "note": "No automatic promotion: inspect coverage, identity and visual quality; lower acceleration alone is not better."}
