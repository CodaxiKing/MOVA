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
                for name in ("output.mp4", "pose_openpose.mp4", "side_by_side.mp4"):
                    if (artifacts / name).is_file():
                        record.setdefault("media", {})[name] = f"/api/media/{artifacts.relative_to(ROOT).as_posix()}/{name}"
            result.append(record)
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
            result = run_inference(InferenceRequest(reference=str(reference), motion=str(motion), model=model,
                overrides=overrides, allow_download=False, allow_cpu=model == "tiny"),
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
            from shutil import disk_usage

            info = system_info()
            info["model_details"] = {name: model_info(name) for name in ("tiny", "wan")}
            info["disk_free_gb"] = round(disk_usage(ROOT).free / 1e9, 2)
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
        if path.startswith("/api/media/"):
            target = (ROOT / unquote(path.removeprefix("/api/media/"))).resolve()
            if not target.is_relative_to(ROOT / "outputs") or not target.is_file() or target.suffix.lower() != ".mp4":
                return self.json({"error": "Mídia não encontrada"}, 404)
            content = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "video/mp4")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/jobs":
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
