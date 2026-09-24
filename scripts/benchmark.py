"""Prepare, freeze, generate, evaluate and compare a bounded benchmark."""

import argparse
import json
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
from common.video_io import iter_frames, probe_video, write_video
from evaluation.protocol import freeze_manifest, inspect_manifest, read_manifest, sha256, verify_lock
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


def generation_config(locked, case):
    cfg = load_config("configs/baseline.yaml")
    p = locked["protocol"]
    cfg["inputs"] = {"reference": case["reference"], "motion": case["motion"],
                     "motion_is_control": False, "motion_fps": p["fps"], "frame_stride": 1}
    cfg["generation"].update(height=p["height"], width=p["width"], num_frames=p["frames"],
                             fps=p["fps"], seed=p["seed"], keep_reference_aspect=False, prompt=case["prompt"])
    return cfg


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
    for command in ("generate", "evaluate"):
        ap = sub.add_parser(command)
        ap.add_argument("--lock", required=True)
        if command == "generate":
            ap.add_argument("--allow-download", action="store_true")
        else:
            ap.add_argument("--outputs", required=True, help="Directory with <case_id>.mp4")
            ap.add_argument("--label", required=True)
            ap.add_argument("--contract", required=True, help="JSON with matching generation conditions")
            ap.add_argument("--generation-index", help="Optional generation.json produced by generate")
    ap = sub.add_parser("compare")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    args = parser.parse_args()
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
                                         if args.generation_index else None)
        print(path)
        return 0 if report["status"] == "complete" else 2
    locked = verify_lock(args.lock)
    if not detect_hardware().cuda_available:
        raise ValueError("Baseline generation requires CUDA. No models were downloaded.")
    root = PROJECT_ROOT / "outputs" / "benchmark" / uuid.uuid4().hex[:12]
    root.mkdir(parents=True)
    (root / "videos").mkdir()
    contract = None
    index = {"benchmark_sha256": locked["benchmark_sha256"], "cases": {}}
    for case in locked["cases"]:
        cfg = generation_config(locked, case)
        current_contract = {"generation": {k: v for k, v in cfg["generation"].items() if k != "prompt"},
                            "input_protocol": "prepared-driver-v1", "model_settings": cfg["model"]}
        if contract is not None and contract != current_contract:
            raise ValueError("Generation config changed during benchmark")
        contract = current_contract
        cfg["output"]["dir"] = str(root / case["id"])
        config_path = root / f"{case['id']}.yaml"
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts/inference_baseline.py"), "--config", str(config_path)]
        if args.allow_download:
            cmd.append("--allow-download")
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        videos = list((root / case["id"]).glob("*/output.mp4"))
        if len(videos) != 1:
            raise ValueError(f"Expected exactly one output for {case['id']}")
        shutil.copyfile(videos[0], root / "videos" / f"{case['id']}.mp4")
        record_path = PROJECT_ROOT / "experiments" / "runs" / videos[0].parent.name / "run.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        resolved_contract = {"input_protocol": "prepared-driver-v1",
                             "settings": {k: v for k, v in record["settings"].items()
                                          if k not in ("prompt", "embed_cache_dir")}}
        if "generation_contract" in index and index["generation_contract"] != resolved_contract:
            raise ValueError("Resolved generation settings changed between cases")
        index["generation_contract"] = resolved_contract
        (root / "contract.json").write_text(json.dumps(resolved_contract, indent=2), encoding="utf-8")
        elapsed = record["stats"]["generation_time_s"]
        index["cases"][case["id"]] = {
            "run_record": str(record_path), "run_record_sha256": sha256(record_path),
            "output_sha256": sha256(videos[0]), "generation_time_s": elapsed,
            "seconds_per_frame": elapsed / locked["protocol"]["frames"],
            "vram_peak_gb": record["stats"]["vram_peak_gb"], "hardware": record["hardware"],
            "settings": record["settings"], "git_commit": record["git_commit"],
            "model_revision": None, "note": "Model revision not pinned by legacy baseline loader"}
        (root / "generation.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Generated videos: {root / 'videos'}\nEvaluation contract: {root / 'contract.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
