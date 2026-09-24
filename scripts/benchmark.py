"""Prepare, freeze, generate, evaluate and compare a bounded benchmark."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
from PIL import Image

from common.config import PROJECT_ROOT, load_config, resolve_path
from common.env import detect_hardware
from common.provenance import package_versions
from common.video_io import iter_frames, probe_video, write_video
from evaluation.protocol import (fingerprint, freeze_manifest, inspect_manifest, read_manifest, sha256, update_case,
                                 verify_lock, write_manifest)
from evaluation.video import validate_video
from inference.conditioning import letterbox


def prepare_video(source, destination, protocol):
    """First N sampled frames, letterboxed; refuse padding or FPS upsampling."""
    p = protocol
    if probe_video(source).fps + 0.01 < p["fps"]:
        raise ValueError("Source FPS below protocol; choose another clip or protocol")
    frames = [np.asarray(letterbox(Image.fromarray(rgb), p["width"], p["height"], fill=(0, 0, 0)))
              for _, _, rgb in iter_frames(source, target_fps=p["fps"], max_frames=p["frames"])]
    if len(frames) != p["frames"]:
        raise ValueError("Source too short; benchmark does not repeat frames")
    if Path(destination).exists():
        raise FileExistsError(destination)
    write_video(destination, frames, p["fps"])
    return validate_video(destination, count=p["frames"], width=p["width"], height=p["height"], fps=p["fps"])


def intake_case(manifest_path, case_id, *, reference, motion, **fields):
    """Put one case's media in place (reference as PNG, motion prepared to the protocol) and fill its metadata.
    Existing media are never overwritten; the manifest is rewritten only after both files are in place."""
    manifest = read_manifest(manifest_path)
    case = update_case(manifest, case_id, **fields)
    ref_dst, mot_dst = resolve_path(case["reference"]), resolve_path(case["motion"])
    for dst in (ref_dst, mot_dst):
        if dst.exists():
            raise FileExistsError(f"{dst} exists; remove it deliberately to replace this case")
    ref_dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(resolve_path(reference)) as im:
        im.verify()
    with Image.open(resolve_path(reference)) as im:
        im.convert("RGB").save(ref_dst)
    try:
        report = prepare_video(resolve_path(motion), mot_dst, manifest["protocol"])
    except Exception:
        ref_dst.unlink(missing_ok=True)
        raise
    write_manifest(manifest_path, manifest)
    return {"case": case_id, "reference": str(ref_dst), "motion": str(mot_dst), "motion_validation": report,
            "remaining_issues": [i for i in inspect_manifest(manifest) if i.startswith(case_id + ":")]}


def generation_config(locked, case):
    cfg = load_config("configs/baseline.yaml")
    p = locked["protocol"]
    cfg["inputs"] = {"reference": case["reference"], "motion": case["motion"],
                     "motion_is_control": False, "motion_fps": p["fps"], "frame_stride": 1}
    cfg["generation"].update(height=p["height"], width=p["width"], num_frames=p["frames"],
                             fps=p["fps"], seed=p["seed"], keep_reference_aspect=False, prompt=case["prompt"])
    return cfg


CODE_ROOT = Path(__file__).resolve().parent.parent
# Files whose content can change a generated video; resuming across a change would mix two systems.
GENERATION_CODE = ("inference", "common", "preprocessing", "runtime", "models", "core", "mova",
                   "scripts/inference_baseline.py", "evaluation/video.py", "configs/baseline.yaml",
                   "configs/runtime.yaml")


def generation_code_fingerprint(root=CODE_ROOT):
    files = {}
    for entry in GENERATION_CODE:
        path = root / entry
        for f in sorted(path.rglob("*.py")) if path.is_dir() else [path]:
            if f.is_file():
                files[f.relative_to(root).as_posix()] = sha256(f)
    return fingerprint(files)


def generation_environment():
    return {"packages": package_versions(), "generation_code_sha256": generation_code_fingerprint()}


def write_json_atomic(path, data):
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # an interruption leaves either the old or the new state, never half a file


def _open_run(locked, lock_path, resume):
    env = generation_environment()
    if resume is None:
        root = PROJECT_ROOT / "outputs" / "benchmark" / uuid.uuid4().hex[:12]
        (root / "videos").mkdir(parents=True)
        index = {"schema_version": 2, "status": "in_progress", "lock": str(lock_path),
                 "benchmark_sha256": locked["benchmark_sha256"], "environment": env, "cases": {}, "attempts": {}}
        write_json_atomic(root / "generation.json", index)
        return root, index
    root = resolve_path(resume)
    index = json.loads((root / "generation.json").read_text(encoding="utf-8"))
    if index.get("benchmark_sha256") != locked["benchmark_sha256"]:
        raise ValueError("Resume refused: the run was started with a different benchmark lock")
    if index.get("environment") != env:
        old = index.get("environment") or {}
        changed = [k for k in env["packages"] if old.get("packages", {}).get(k) != env["packages"][k]]
        if old.get("generation_code_sha256") != env["generation_code_sha256"]:
            changed.append("generation code")
        raise ValueError(f"Resume refused: changed since the run started: {changed}. Start a new run.")
    index.setdefault("attempts", {})
    return root, index


def generate_benchmark(locked, lock_path, *, allow_download=False, resume=None, runner=None):
    """Generate every case once, saving state after each case so an interruption can be resumed."""
    runner = runner or subprocess.run
    root, index = _open_run(locked, lock_path, resume)
    print(f"Benchmark run: {root}\nIf interrupted, resume with: "
          f"python scripts/benchmark.py generate --lock {lock_path} --resume {root}")
    if index["status"] == "complete":
        print("Run already complete; nothing to do.")
        return root
    contract = index.get("request_contract")
    for case in locked["cases"]:
        cid = case["id"]
        done = index["cases"].get(cid)
        if done:
            video = root / "videos" / f"{cid}.mp4"
            if not video.is_file() or sha256(video) != done["output_sha256"]:
                raise ValueError(f"Resume refused: {video} is missing or changed since it was generated")
            print(f"[skip] {cid}: already generated and unchanged")
            continue
        cfg = generation_config(locked, case)
        current_contract = {"generation": {k: v for k, v in cfg["generation"].items() if k != "prompt"},
                            "input_protocol": "prepared-driver-v1", "model_settings": cfg["model"]}
        if contract is not None and contract != current_contract:
            raise ValueError("Generation config changed during benchmark")
        contract = index["request_contract"] = current_contract
        index["attempts"][cid] = index["attempts"].get(cid, 0) + 1
        write_json_atomic(root / "generation.json", index)

        case_dir = root / cid
        case_dir.mkdir(exist_ok=True)
        cfg["output"]["dir"] = str(case_dir)
        config_path = root / f"{cid}.yaml"
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        before = {d.name for d in case_dir.iterdir() if d.is_dir()}
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts/inference_baseline.py"), "--config", str(config_path)]
        if allow_download:
            cmd.append("--allow-download")
        runner(cmd, cwd=PROJECT_ROOT, check=True)
        # Earlier interrupted attempts may have left folders; only this attempt's new output counts.
        new = [d for d in case_dir.iterdir() if d.is_dir() and d.name not in before and (d / "output.mp4").is_file()]
        if len(new) != 1:
            raise ValueError(f"Expected exactly one new output for {cid}, found {len(new)}")
        output = new[0] / "output.mp4"
        shutil.copyfile(output, root / "videos" / f"{cid}.mp4")
        record_path = PROJECT_ROOT / "experiments" / "runs" / new[0].name / "run.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        resolved_contract = {"input_protocol": "prepared-driver-v1",
                             "settings": {k: v for k, v in record["settings"].items()
                                          if k not in ("prompt", "embed_cache_dir")}}
        if "generation_contract" in index and index["generation_contract"] != resolved_contract:
            raise ValueError("Resolved generation settings changed between cases")
        index["generation_contract"] = resolved_contract
        write_json_atomic(root / "contract.json", resolved_contract)
        elapsed = record["stats"]["generation_time_s"]
        index["cases"][cid] = {
            "run_record": str(record_path), "run_record_sha256": sha256(record_path),
            "output_sha256": sha256(output), "generation_time_s": elapsed,
            "seconds_per_frame": elapsed / locked["protocol"]["frames"],
            "vram_peak_gb": record["stats"]["vram_peak_gb"], "hardware": record["hardware"],
            "settings": record["settings"], "git_commit": record["git_commit"],
            "model_revision": record.get("model_revision"), "attempts": index["attempts"][cid]}
        write_json_atomic(root / "generation.json", index)
    index["status"] = "complete"
    write_json_atomic(root / "generation.json", index)
    print(f"Generated videos: {root / 'videos'}\nEvaluation contract: {root / 'contract.json'}")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "freeze", "prepare"):
        ap = sub.add_parser(command)
        ap.add_argument("--manifest", default="benchmark/v1.draft.yaml")
        if command == "freeze":
            ap.add_argument("--out", required=True)
        if command == "prepare":
            ap.add_argument("--video", required=True)
            ap.add_argument("--out", required=True)
    ap = sub.add_parser("intake", help="put one case's media in place and fill its source/license/identity")
    ap.add_argument("--manifest", default="benchmark/v1.draft.yaml")
    ap.add_argument("--case", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--motion", required=True, help="raw clip already trimmed to the moment of interest")
    for f in ("identity_id", "reference_source", "motion_source", "reference_license", "motion_license", "prompt"):
        ap.add_argument("--" + f.replace("_", "-"), dest=f, default=None)
    for command in ("generate", "evaluate"):
        ap = sub.add_parser(command)
        ap.add_argument("--lock", required=True)
        if command == "generate":
            ap.add_argument("--allow-download", action="store_true")
            ap.add_argument("--resume", help="outputs/benchmark/<run> of an interrupted generate")
        else:
            ap.add_argument("--outputs", required=True, help="Directory with <case_id>.mp4")
            ap.add_argument("--label", required=True)
            ap.add_argument("--contract", required=True, help="JSON with matching generation conditions")
            ap.add_argument("--generation-index", help="Optional generation.json produced by generate")
            ap.add_argument("--resume", help="outputs/evaluation/<run> of an interrupted evaluate")
            ap.add_argument("--identity-embedder", default=None,
                            help="optional cached embedding model, e.g. hf:facebook/dinov2-small (never downloaded)")
    ap = sub.add_parser("review", help="(re)build the visual review page of an evaluation report")
    ap.add_argument("--report", required=True)
    ap = sub.add_parser("gsb", help="build a blind pairwise GSB rating session from two evaluation reports")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap = sub.add_parser("gsb-score", help="score a filled ratings.csv against its key.json")
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--key", required=True)
    ap = sub.add_parser("compare")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "intake":
        fields = {f: getattr(args, f) for f in ("identity_id", "reference_source", "motion_source",
                                                 "reference_license", "motion_license", "prompt")}
        print(json.dumps(intake_case(args.manifest, args.case, reference=args.reference, motion=args.motion, **fields),
                         indent=2))
        return 0
    if args.command in ("check", "freeze", "prepare"):
        manifest = read_manifest(args.manifest)
        if args.command == "check":
            issues = inspect_manifest(manifest)
            print(json.dumps({"status": "BLOCKED" if issues else "READY_TO_FREEZE", "cases": len(manifest["cases"]),
                              "issues": issues}, indent=2))
            return 2 if issues else 0
        if args.command == "freeze":
            freeze_manifest(manifest, resolve_path(args.out))
            print(f"Frozen benchmark: {args.out}")
        else:
            print(prepare_video(resolve_path(args.video), resolve_path(args.out), manifest["protocol"]))
        return 0
    if args.command == "review":
        from evaluation.review import build_review

        print(build_review(resolve_path(args.report)))
        return 0
    if args.command == "gsb":
        from evaluation.gsb import build_gsb_session

        paths = build_gsb_session(resolve_path(args.baseline), resolve_path(args.candidate), resolve_path(args.out),
                                  seed=args.seed)
        print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
        print("Give raters index.html + ratings.csv only; keep key.json hidden until scoring.")
        return 0
    if args.command == "gsb-score":
        from evaluation.gsb import score_gsb

        print(json.dumps(score_gsb(resolve_path(args.ratings), resolve_path(args.key)), indent=2))
        return 0
    if args.command == "compare":
        from evaluation.benchmark import compare_reports

        report = compare_reports(json.loads(resolve_path(args.baseline).read_text()),
                                 json.loads(resolve_path(args.candidate).read_text()))
        with resolve_path(args.out).open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        return 0
    if args.command == "evaluate":
        from evaluation.benchmark import evaluate_benchmark

        path, report = evaluate_benchmark(args.lock, args.outputs, label=args.label,
                                         generation_contract=json.loads(resolve_path(args.contract).read_text()),
                                         generation_index=json.loads(resolve_path(args.generation_index).read_text())
                                         if args.generation_index else None, resume_from=args.resume,
                                         identity_embedder=args.identity_embedder)
        print(path)
        return 0 if report["status"] == "complete" else 2
    locked = verify_lock(args.lock)
    if not detect_hardware().cuda_available:
        raise ValueError("Baseline generation requires CUDA. No models were downloaded.")
    generate_benchmark(locked, args.lock, allow_download=args.allow_download, resume=args.resume)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
