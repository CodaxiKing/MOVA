"""Diagnostic information for `mova info` (and the future API): versions, devices, runtimes, models."""

from __future__ import annotations

import platform
import sys
from typing import Any

from models.registry import get_spec, list_specs
from runtime import default_device_manager, describe_runtimes
from runtime.manager import framework_versions

from .capabilities import compatibility_matrix, missing_requirements


def mova_version() -> str:
    from mova import __version__

    return __version__


def system_info() -> dict[str, Any]:
    dm = default_device_manager()
    devices = []
    for d in dm.list_devices():
        row = d.to_dict()
        row["memory"] = dm.memory_info(d)
        devices.append(row)
    return {
        "mova": mova_version(),
        "python": sys.version.split()[0],
        "os": platform.platform(),
        **framework_versions(),
        "devices": devices,
        "runtimes": describe_runtimes(),
        "models": [{"name": s.name, "aliases": list(s.aliases), "task": s.task,
                    "missing_requirements": missing_requirements(s)} for s in list_specs()],
    }


def model_info(name: str) -> dict[str, Any]:
    spec = get_spec(name)
    return {"spec": spec.to_dict(), "missing_requirements": missing_requirements(spec),
            "compatibility": [r.to_dict() for r in compatibility_matrix(spec)]}


def format_info(info: dict[str, Any], model: dict[str, Any] | None = None) -> str:
    lines = [f"MOVA          : {info['mova']}",
             f"Python        : {info['python']}",
             f"OS            : {info['os']}",
             f"PyTorch       : {info['pytorch']} (CUDA build: {info['cuda_build']}, ROCm build: {info['rocm_build']})",
             "Devices       :"]
    for d in info["devices"]:
        mem = d["memory"]
        kind = "VRAM" if d["type"] != "cpu" else "RAM"
        extra = f", cc {d['compute_capability']}" if d["compute_capability"] else ""
        prec = ["fp32"] + (["fp16"] if d["fp16"] else []) + (["bf16"] if d["bf16"] else [])
        lines.append(f"  {d['id']:<8} {d['name']} [{d['backend']}{extra}] {kind} {mem['free_gb']} free / "
                     f"{mem['total_gb']} GB | precisions: {', '.join(prec)}")
        lines += [f"           note: {n}" for n in d["notes"]]
    lines.append("Runtimes      :")
    for r in info["runtimes"]:
        lines.append(f"  {r['name']:<9} {r['status']}{' ' + r['version'] if r['version'] else ''}"
                     f"{' - ' + r['reason'] if r['reason'] else ''}")
    lines.append("Models        :")
    for m in info["models"]:
        miss = f" (missing packages: {', '.join(m['missing_requirements'])})" if m["missing_requirements"] else ""
        lines.append(f"  {m['name']:<26} aliases: {', '.join(m['aliases']) or '-':<14} {m['task']}{miss}")
    if model:
        s = model["spec"]
        size = f" ({s['weights_size_gb']} GB)" if s["weights_size_gb"] else ""
        missing = model["missing_requirements"]
        lines += ["", f"Model {s['name']} ({s['version']})",
                  f"  {s['description']}",
                  f"  family: {s['family']} | license: {s['license']} | weights: {s['weights'] or 'none'}{size}",
                  f"  runtimes: {', '.join(s['runtimes'])} | devices: {', '.join(s['devices'])} | "
                  f"precisions: {', '.join(s['precisions'])}",
                  f"  capabilities: {', '.join(s['capabilities'])}",
                  f"  requirements: {', '.join(s['requirements']) or '-'}"
                  + (f" (missing: {', '.join(missing)})" if missing else ""),
                  f"  default config: {s['default_config']}",
                  "  verification:"]
        lines += [f"    {k}: {v}" for k, v in s["verification"].items()]
        lines.append("  compatibility on this machine:")
        for r in model["compatibility"]:
            chosen = f" -> {r['precision']}, offload {r['offload']}" if r["precision"] else ""
            lines.append(f"    {r['runtime']:<9} {r['device']:<7} {r['status']:<12}{chosen} | {r['detail']}")
    return "\n".join(lines)
