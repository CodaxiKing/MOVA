"""Local MOVA web server. The browser is a thin client of core services."""
from __future__ import annotations

import base64
import json
import mimetypes
import re
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
UPLOADS = ROOT / "assets" / "web_uploads"
RUNS = ROOT / "experiments" / "runs"
JOBS: dict[str, dict] = {}
JOB_LOCK = threading.Lock()


def runs() -> list[dict]:
    result = []
    for path in sorted(RUNS.glob("*/run.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            record["record_url"] = f"/api/runs/{path.parent.name}"
            artifacts = Path(record.get("artifacts_dir") or path.parent).resolve()
            if artifacts.is_relative_to(ROOT / "outputs"):
                for name in ("output.mp4", "control.mp4", "pose_openpose.mp4", "side_by_side.mp4"):
                    if (artifacts / name).is_file():
                        record.setdefault("media", {})[name] = f"/api/media/{artifacts.relative_to(ROOT).as_posix()}/{name}"
            extraction = Path(record.get("output_dir") or path.parent).resolve()
            if extraction.is_relative_to(ROOT / "outputs"):
                for name in ("pose_openpose.mp4", "body_preview.mp4", "face_preview.mp4", "hands_preview.mp4"):
                    if (extraction / name).is_file():
                        record.setdefault("media", {})[name] = f"/api/media/{extraction.relative_to(ROOT).as_posix()}/{name}"
            result.append(record)
            review = path.parent / "web_review.json"
            if review.is_file():
                try:
                    record["web_review"] = json.loads(review.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
            inputs = record.get("inputs") or {}
            for kind in ("reference", "motion"):
                source = (ROOT / str(inputs.get(kind, ""))).resolve()
                if source.is_file() and (source.is_relative_to(ROOT / "assets") or source.is_relative_to(ROOT / "outputs")):
                    record.setdefault("input_media", {})[kind] = f"/api/input/{path.parent.name}/{kind}"
        except (OSError, ValueError, TypeError):
            continue
    return result


def save_upload(data: dict, key: str, suffixes: set[str]) -> Path:
    item = data.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"Arquivo obrigatório: {key}")
    suffix = Path(str(item.get("name", ""))).suffix.lower()
    if suffix not in suffixes:
        raise ValueError(f"Formato inválido para {key}: {suffix}")
    raw = base64.b64decode(item.get("data", ""), validate=True)
    if not raw or len(raw) > 200 * 1024 * 1024:
        raise ValueError(f"{key}: arquivo vazio ou maior que 200 MB")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(raw)
    return path


def execute(job_id: str, data: dict) -> None:
    job = JOBS[job_id]
    try:
        operation = data.get("operation")
        if operation == "preprocess":
            from core.preprocess import run_extraction

            video = save_upload(data, "motion", {".mp4", ".mov", ".avi", ".webm"})
            output = ROOT / "outputs" / "web" / job_id
            summary, record = run_extraction(str(video), str(output))
            job.update(status="success", run_id=record.parent.name, summary=summary)
        elif operation == "infer":
            from core.inference import InferenceRequest, run_inference

            reference = save_upload(data, "reference", {".png", ".jpg", ".jpeg", ".webp"})
            motion = save_upload(data, "motion", {".mp4", ".mov", ".avi", ".webm"})
            model = data.get("model", "tiny")
            if model not in {"tiny", "wan"}:
                raise ValueError("Modelo inválido")
            frames = int(data.get("frames", 5 if model == "tiny" else 17))
            steps = int(data.get("steps", 2 if model == "tiny" else 30))
            if frames < 5 or frames > 33 or (frames - 1) % 4 or steps < 1 or steps > 50:
                raise ValueError("Quadros ou passos fora do intervalo permitido")
            overrides = [f"generation.num_frames={frames}", f"generation.num_inference_steps={steps}"]
            prompt = data.get("prompt")
            if prompt is not None:
                if not isinstance(prompt, str) or len(prompt) > 1000:
                    raise ValueError("Descrição inválida")
                if prompt.strip():
                    overrides.append("generation.prompt=" + json.dumps(prompt.strip()))
            for field, config_key, minimum, maximum in (
                ("guidance", "guidance_scale", 1, 10), ("conditioning", "conditioning_scale", 0, 2),
                ("seed", "seed", 0, 2147483647)):
                if field in data:
                    value = float(data[field]) if field != "seed" else int(data[field])
                    if not minimum <= value <= maximum:
                        raise ValueError(f"{field} fora do intervalo permitido")
                    overrides.append(f"generation.{config_key}={value}")
            device = data.get("device", "auto")
            precision = data.get("precision", "auto")
            if device not in {"auto", "cpu", "cuda:0"} or precision not in {"auto", "fp32", "fp16", "bf16"}:
                raise ValueError("Dispositivo ou precisão inválida")
            result = run_inference(InferenceRequest(reference=str(reference), motion=str(motion), model=model,
                overrides=overrides, device=device, precision=precision, allow_download=False, allow_cpu=model == "tiny"),
                echo=lambda line: job["log"].append(str(line)))
            job.update(status="success", run_id=result.run_id, output=f"/api/media/{result.output.resolve().relative_to(ROOT).as_posix()}")
        else:
            raise ValueError("Operação inválida")
    except Exception as exc:
        job.update(status="failed", error=str(exc))
    finally:
        JOB_LOCK.release()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def json(self, value, status=200):
        body = json.dumps(value, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/info":
            from core.info import model_info, system_info
            from common.env import detect_hardware, select_profile
            from core.config import resolve_run_config
            from models.registry import create_model
            from shutil import disk_usage

            info = system_info()
            info["model_details"] = {name: model_info(name) for name in ("tiny", "wan")}
            info["disk_free_gb"] = round(disk_usage(ROOT).free / 1e9, 2)
            info["profile"] = select_profile(detect_hardware()).to_dict()
            try:
                _, config = resolve_run_config(model="wan")
                cache = create_model("wan", config).weights_status()
                info["wan_weights"] = {"complete": cache.complete, "summary": cache.summary()}
            except Exception as exc:
                info["wan_weights"] = {"complete": False, "summary": str(exc)}
            return self.json(info)
        if path == "/api/runs":
            return self.json(runs())
        if path.startswith("/api/runs/"):
            run_id = path.removeprefix("/api/runs/")
            record = RUNS / run_id / "run.json"
            if re.fullmatch(r"[A-Za-z0-9_-]+", run_id) and record.is_file():
                return self.json(json.loads(record.read_text(encoding="utf-8")))
            return self.json({"error": "Registro não encontrado"}, 404)
        if path.startswith("/api/jobs/"):
            return self.json(JOBS.get(path.rsplit("/", 1)[-1], {"error": "Job não encontrado"}))
        if path.startswith("/api/sample/"):
            samples = {"reference": ROOT / "assets/reference/maya.png", "motion": ROOT / "assets/motion/dance.mp4.mp4"}
            target = samples.get(path.rsplit("/", 1)[-1])
            if not target or not target.is_file():
                return self.json({"error": "Exemplo indisponível"}, 404)
            return self.file(target)
        if path.startswith("/api/input/"):
            parts = path.split("/")
            if len(parts) != 5 or not re.fullmatch(r"[A-Za-z0-9_-]+", parts[3]) or parts[4] not in {"reference", "motion"}:
                return self.json({"error": "Entrada inválida"}, 404)
            record = RUNS / parts[3] / "run.json"
            if not record.is_file():
                return self.json({"error": "Registro não encontrado"}, 404)
            source = (ROOT / str((json.loads(record.read_text(encoding="utf-8")).get("inputs") or {}).get(parts[4], ""))).resolve()
            if not source.is_file() or not (source.is_relative_to(ROOT / "assets") or source.is_relative_to(ROOT / "outputs")):
                return self.json({"error": "Entrada indisponível"}, 404)
            return self.file(source)
        if path.startswith("/api/media/"):
            target = (ROOT / unquote(path.removeprefix("/api/media/"))).resolve()
            if not target.is_relative_to(ROOT / "outputs") or not target.is_file() or target.suffix.lower() != ".mp4":
                return self.json({"error": "Mídia não encontrada"}, 404)
            return self.file(target)
        return super().do_GET()

    def file(self, target: Path):
        content = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        path = urlparse(self.path).path
        if path.startswith("/api/runs/") and path.endswith("/review"):
            run_id = path.split("/")[3]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or not (RUNS / run_id / "run.json").is_file():
                return self.json({"error": "Registro não encontrado"}, 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 8192:
                    return self.json({"error": "Revisão muito grande"}, 413)
                data = json.loads(self.rfile.read(length))
                if data.get("verdict") not in {"ok", "bad", None} or not isinstance(data.get("notes"), str) or len(data["notes"]) > 4000:
                    return self.json({"error": "Revisão inválida"}, 400)
                target = RUNS / run_id / "web_review.json"
                target.write_text(json.dumps({"verdict": data.get("verdict"), "notes": data["notes"]}, ensure_ascii=False, indent=2), encoding="utf-8")
                return self.json({"saved": True})
            except (ValueError, TypeError):
                return self.json({"error": "JSON inválido"}, 400)
        if path != "/api/jobs":
            return self.json({"error": "Rota não encontrada"}, 404)
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > 275 * 1024 * 1024:
            return self.json({"error": "Payload vazio ou muito grande"}, 413)
        try:
            data = json.loads(self.rfile.read(length))
        except ValueError:
            return self.json({"error": "JSON inválido"}, 400)
        if not JOB_LOCK.acquire(blocking=False):
            return self.json({"error": "Já existe uma tarefa em execução"}, 409)
        job_id = uuid.uuid4().hex
        JOBS[job_id] = {"id": job_id, "status": "running", "log": []}
        threading.Thread(target=execute, args=(job_id, data), daemon=True).start()
        return self.json(JOBS[job_id], 202)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interface local do MOVA")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"MOVA: http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
