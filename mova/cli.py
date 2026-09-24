"""`mova` command line. Thin layer: parses arguments and calls MOVA Core; no inference logic here.

  mova info [--model NAME] [--json]
  mova infer [--model M] [--runtime R] [--device D] [--precision P] --reference ref.png --motion dance.mp4 --output out.mp4
  mova preprocess --video dance.mp4 --out outputs/motion/dance
  mova benchmark <check|prepare|freeze|generate|evaluate|compare> ...   (scripts/benchmark.py)
  mova evaluate ...                                                     (= mova benchmark evaluate ...)
  mova test [pytest args]
  mova train [--smoke] [--config configs/train_adapter.yaml] [--steps N] [--resume ckpt]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(ROOT))

from common.errors import MovaError  # noqa: E402


def _add_infer_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None, help="registered model name or alias (mova info lists them)")
    p.add_argument("--runtime", default=None, help="pytorch (default from configs/runtime.yaml)")
    p.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:<index>")
    p.add_argument("--precision", default=None, help="auto | fp32 | fp16 | bf16")
    p.add_argument("--offload", default=None, help="auto | none | model | sequential")
    p.add_argument("--config", default=None, help="run config (default: the model's default config)")
    p.add_argument("--runtime-config", default="configs/runtime.yaml")
    p.add_argument("--reference", default=None)
    p.add_argument("--motion", default=None)
    p.add_argument("--output", default=None, help="also copy the validated output.mp4 here")
    p.add_argument("--overwrite", action="store_true", help="allow replacing an existing --output file")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override (repeatable)")
    p.add_argument("--allow-download", action="store_true", help="download missing weights of the pinned manifest")
    p.add_argument("--allow-cpu", action="store_true", help="allow models that are impractical on CPU to run there")
    p.add_argument("--verify-hashes", action="store_true", help="also SHA-256 every cached weight file (slow)")
    p.add_argument("--skip-resource-check", action="store_true", help="start even if the preflight says no")
    p.add_argument("--json", action="store_true", help="print the result as JSON")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mova", description="MOVA character motion control")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="versions, devices, runtimes and models")
    p.add_argument("--model", default=None, help="also show spec and compatibility of this model")
    p.add_argument("--json", action="store_true")

    _add_infer_args(sub.add_parser("infer", help="reference image + motion video -> video"))

    p = sub.add_parser("preprocess", help="extract body/face/hand motion from a video (CPU)")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/extraction.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--retarget-to", default=None, metavar="REFERENCE_IMAGE",
                   help="also write <out>/retargeted/: motion with this character's body proportions")

    # Listed for --help only; main() forwards them before parsing (arguments go to the underlying tool).
    sub.add_parser("benchmark", help="benchmark workflow (scripts/benchmark.py ...)")
    sub.add_parser("evaluate", help="evaluate benchmark outputs (= scripts/benchmark.py evaluate ...)")
    sub.add_parser("test", help="run the test suite (pytest ...)")
    p = sub.add_parser("train", help="train the MOVA adapter on a frozen backbone (--smoke: tiny CPU run)")
    p.add_argument("--config", default="configs/train_adapter.yaml")
    p.add_argument("--smoke", action="store_true", help="tiny random backbone + synthetic data; proves the loop")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--resume", default=None, help="adapter checkpoint to continue from")
    p.add_argument("--runtime", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--precision", default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return ap


def _forward(cmd: list[str]) -> int:
    return subprocess.run(cmd, cwd=ROOT).returncode


PASSTHROUGH = {
    "benchmark": lambda rest: [sys.executable, str(ROOT / "scripts/benchmark.py"), *rest],
    "evaluate": lambda rest: [sys.executable, str(ROOT / "scripts/benchmark.py"), "evaluate", *rest],
    "test": lambda rest: [sys.executable, "-m", "pytest", *rest],
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in PASSTHROUGH:  # argparse.REMAINDER drops leading flags such as `mova test -q`
        return _forward(PASSTHROUGH[argv[0]](argv[1:]))
    args = build_parser().parse_args(argv)
    try:
        if args.command == "info":
            from core.info import format_info, model_info, system_info

            info = system_info()
            model = model_info(args.model) if args.model else None
            print(json.dumps({"system": info, "model": model}, indent=2, default=str) if args.json
                  else format_info(info, model))
            return 0

        if args.command == "infer":
            from core.inference import InferenceRequest, run_inference

            req = InferenceRequest(
                reference=args.reference, motion=args.motion, output=args.output, model=args.model,
                config=args.config, overrides=args.set, runtime=args.runtime, device=args.device,
                precision=args.precision, offload=args.offload, runtime_config=args.runtime_config,
                allow_download=args.allow_download, allow_cpu=args.allow_cpu, verify_hashes=args.verify_hashes,
                skip_resource_check=args.skip_resource_check, overwrite=args.overwrite)
            result = run_inference(req, echo=(lambda _: None) if args.json else print)
            if args.json:
                print(json.dumps({"run_id": result.run_id, "output": str(result.output), "record": str(result.record),
                                  "runtime": result.context, "stats": result.stats}, indent=2, default=str))
            return 0

        if args.command == "preprocess":
            from core.preprocess import run_extraction

            _, record = run_extraction(args.video, args.out, args.config, args.set, retarget_to=args.retarget_to)
            print(f"Run record: {record}")
            return 0

        if args.command == "train":
            from core.train import TrainRequest, run_training

            result = run_training(TrainRequest(config=args.config, smoke=args.smoke, steps=args.steps,
                                               resume=args.resume, runtime=args.runtime, device=args.device,
                                               precision=args.precision, overrides=args.set))
            print(json.dumps(result, indent=2))
            return 0
    except MovaError as e:
        print(f"ERROR [{e.code}]: {e}", file=sys.stderr)
        return e.exit_code
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
