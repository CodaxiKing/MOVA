"""Visual review page for an evaluation report: side-by-side videos, metrics and per-case flags.

Flags are REVIEW HINTS that tell a human where to look first. They are not quality verdicts and must not
be used to promote a model automatically (see ADR-007).
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from common.config import resolve_path
from common.video_io import read_video, write_video
from inference.conditioning import letterbox

# Thresholds only decide which cases get highlighted; tune them as reviewed cases accumulate.
REVIEW_HINTS = {
    "min_paired_coverage": 0.8,
    "min_pck": 0.5,
    "max_trajectory_error": 0.5,      # torso lengths
    "max_scale_log_error": 0.25,      # |log size ratio| ~ 28 %
    "max_head_relative_error_deg": 15.0,
    "max_yaw_relative_error_deg": 20.0,
    "max_blendshape_mae": 0.15,
    # identity-v1 / temporal-v1: first guesses from synthetic fixtures (same person 0.009, face 29 % wider 0.055;
    # recoloured face ΔE 35 vs 1.5). NOT calibrated on generated videos yet.
    "max_face_geometry_error": 0.035,
    "max_face_color_delta_e": 10.0,
    "max_torso_color_delta_e": 12.0,
    "max_warp_error_ratio": 2.0,       # generated texture changes 2x more than the real driver video
    "min_flow_ratio": 0.3,             # generated moves < 30 % of the driver: frozen / under-animated
}


def _get(metrics: dict, group: str, field: str):
    node = metrics.get(group, {})
    for part in field.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node if isinstance(node, (int, float)) and not isinstance(node, bool) else None
BODY_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
              (23, 25), (25, 27), (24, 26), (26, 28), (0, 11), (0, 12)]
DRIVER_COLOR, GENERATED_COLOR = (0, 220, 255), (255, 60, 200)
LABEL_H = 24


def case_flags(item: dict, hints: dict = REVIEW_HINTS) -> list[dict]:
    """[{'level': 'error'|'warn'|'info', 'message': str}] for one report case."""
    flags: list[dict] = []

    def add(level, message):
        flags.append({"level": level, "message": message})

    if item.get("status") != "evaluated":
        add("error", f"Not evaluated: {item.get('error', 'unknown error')}")
        return flags
    m = item.get("metrics", {})
    ident = m.get("identity", {})
    if ident.get("metric_version") and not ident.get("reference_face_detected"):
        add("info", "identity: no face detected in the reference image, face identity metrics are null")
    for group in ("body", "hands", "face", "trajectory", "head_rotation", "body_orientation"):
        g = m.get(group, {})
        if g.get("status") == "UNAVAILABLE":
            add("info", f"{group}: unavailable ({g.get('reason')})")
            continue
        cov = g.get("paired_coverage")
        if cov is None:
            add("info", f"{group}: driver has no usable detections, metric is null")
        elif cov < hints["min_paired_coverage"]:
            add("warn", f"{group}: generated output matched only {cov:.0%} of driver detections")
    checks = [
        ("body", "pck", "min_pck", "body pose PCK {v:.2f} < {t}"),
        ("hands", "pck", "min_pck", "hand PCK {v:.2f} < {t}"),
        ("trajectory", "trajectory_error", "max_trajectory_error", "root path off by {v:.2f} torso lengths"),
        ("trajectory", "scale_log_error", "max_scale_log_error", "apparent body size diverges (|log| {v:.2f})"),
        ("head_rotation", "relative_geodesic_error_deg", "max_head_relative_error_deg",
         "head rotation differs by {v:.1f}° (relative to first frame)"),
        ("body_orientation", "relative_yaw_error_deg", "max_yaw_relative_error_deg",
         "torso turning differs by {v:.1f}°"),
        ("face", "blendshape_mae_paired", "max_blendshape_mae", "expression MAE {v:.3f}"),
        ("identity", "face_geometry_error.mean", "max_face_geometry_error", "face proportions drift ({v:.3f} log-ratio)"),
        ("identity", "face_color_delta_e.mean", "max_face_color_delta_e", "face colour drift ΔE {v:.1f}"),
        ("identity", "torso_color_delta_e.mean", "max_torso_color_delta_e", "outfit colour drift ΔE {v:.1f}"),
        ("temporal", "warp_error_ratio", "max_warp_error_ratio", "texture flicker {v:.1f}x the driver's (warp error)"),
        ("temporal", "mean_flow_px_ratio", "min_flow_ratio", "moves only {v:.0%} as much as the driver (frozen?)"),
    ]
    for group, field, key, text in checks:
        v = _get(m, group, field)
        if v is None:
            continue
        bad = v < hints[key] if key.startswith("min") else v > hints[key]
        if bad:
            add("warn", text.format(v=v, t=hints[key]))
    head = m.get("head_rotation", {})
    if head.get("geodesic_error_deg") is not None and head.get("relative_geodesic_error_deg") is not None:
        if head["geodesic_error_deg"] - head["relative_geodesic_error_deg"] > hints["max_head_relative_error_deg"]:
            add("info", f"constant head offset ~{head['geodesic_error_deg']:.0f}° (character may face differently)")
    return flags


def _load_body(folder: Path):
    path = folder / "body_motion.pt"
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def _draw_skeleton(img: np.ndarray, kp2d, frame: int, color, visibility=0.5) -> None:
    if kp2d is None or frame >= len(kp2d["kp2d"]) or not bool(kp2d["present"][frame]):
        return
    h, w = img.shape[:2]
    kp = np.asarray(kp2d["kp2d"][frame], dtype=float)
    ok = kp[:, 3] >= visibility
    pts = [(int(round(x * w)), int(round(y * h))) for x, y in kp[:, :2]]
    for a, b in BODY_EDGES:
        if ok[a] and ok[b]:
            cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i in {i for e in BODY_EDGES for i in e}:
        if ok[i]:
            cv2.circle(img, pts[i], 3, color, -1, cv2.LINE_AA)


def _tile(frame: np.ndarray, label: str) -> np.ndarray:
    bar = np.full((LABEL_H, frame.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([bar, frame], axis=0)


def compose_case_video(item: dict, protocol: dict, destination: Path) -> Path | None:
    inputs = item.get("inputs") or {}
    w, h = protocol["width"], protocol["height"]
    output = Path(inputs.get("output", ""))
    if not output.is_file():
        return None
    generated = read_video(output)
    driver = read_video(resolve_path(inputs["motion"])) if inputs.get("motion") else []
    ref_path = resolve_path(inputs["reference"]) if inputs.get("reference") else None
    reference = (np.asarray(letterbox(Image.open(ref_path), w, h)) if ref_path and ref_path.is_file()
                 else np.zeros((h, w, 3), np.uint8))
    artifacts = Path(item.get("artifacts", ""))
    drv_body, gen_body = _load_body(artifacts / "driver"), _load_body(artifacts / "generated")
    n = len(generated) if not driver else min(len(generated), len(driver))
    frames = []
    for i in range(n):
        gen = np.ascontiguousarray(generated[i])
        overlay = (gen * 0.55).astype(np.uint8)
        _draw_skeleton(overlay, drv_body, i, DRIVER_COLOR)
        _draw_skeleton(overlay, gen_body, i, GENERATED_COLOR)
        tiles = [_tile(reference, "reference"),
                 _tile(driver[i] if driver else np.zeros_like(gen), "driver"),
                 _tile(gen, "generated"),
                 _tile(overlay, "pose: driver / generated")]
        cv2.putText(tiles[3], f"{i + 1}/{n}", (w - 44, LABEL_H + h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (235, 235, 235), 1, cv2.LINE_AA)
        frames.append(np.concatenate(tiles, axis=1))
    return write_video(destination, frames, protocol["fps"])


def _fmt(v) -> str:
    if v is None:
        return "<span class='null'>null</span>"
    if isinstance(v, float):
        return f"{v:.3f}"
    return html.escape(str(v))


METRIC_TABLE = {
    "Body pose": ("body", ["pck", "paired_coverage", "mean_error_paired", "acceleration_error_paired"]),
    "Hands": ("hands", ["pck", "paired_coverage", "mean_error_paired"]),
    "Face expression": ("face", ["paired_coverage", "blendshape_mae_paired"]),
    "Global trajectory": ("trajectory", ["trajectory_error", "final_displacement_error", "scale_log_error",
                                         "absolute_root_error", "reference_path_length", "paired_coverage"]),
    "Head rotation": ("head_rotation", ["geodesic_error_deg", "relative_geodesic_error_deg",
                                        "reference_rotation_range_deg", "paired_coverage"]),
    "Body orientation": ("body_orientation", ["yaw_error_deg", "relative_yaw_error_deg",
                                              "reference_yaw_range_deg", "paired_coverage"]),
    "Identity": ("identity", ["face_coverage", "face_geometry_error.mean", "face_color_delta_e.mean",
                              "face_color_hist_intersection.mean", "torso_coverage", "torso_color_delta_e.mean",
                              "embedding.mean"]),
    "Temporal (dynamic quality)": ("temporal", ["generated.warp_error", "warp_error_ratio", "generated.static_flicker",
                                                "generated.luma_flicker", "generated.mean_flow_px",
                                                "mean_flow_px_ratio"]),
}

CSS = """
:root{--bg:#f6f7fb;--card:#fff;--fg:#1d2230;--muted:#667085;--line:#e4e7ec;--warn:#b54708;--warnbg:#fffaeb;
--err:#b42318;--errbg:#fef3f2;--info:#175cd3;--infobg:#eff8ff;--ok:#067647;--okbg:#ecfdf3}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--card:#161b22;--fg:#e6edf3;--muted:#8b949e;--line:#30363d;
--warn:#f5b041;--warnbg:#2d2208;--err:#ff7b72;--errbg:#2d1214;--info:#79c0ff;--infobg:#0c2136;--ok:#56d364;--okbg:#0f2a17}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px}h1{margin:0 0 4px}h2{margin:0}.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:16px 0}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600}td.num{font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;margin:2px 4px 2px 0}
.error{color:var(--err);background:var(--errbg)}.warn{color:var(--warn);background:var(--warnbg)}
.info{color:var(--info);background:var(--infobg)}.ok{color:var(--ok);background:var(--okbg)}
video{width:100%;border-radius:8px;background:#000}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.null{color:var(--muted);font-style:italic}a{color:var(--info)}.scroll{overflow-x:auto}
pre{white-space:pre-wrap;background:var(--bg);padding:8px;border-radius:8px}
"""


def build_review(report_path: str | Path, hints: dict = REVIEW_HINTS) -> Path:
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    out = report_path.parent / "review"
    (out / "videos").mkdir(parents=True, exist_ok=True)
    protocol = report["protocol"]
    rows, sections, all_flags = [], [], {}
    for item in report["cases"]:
        cid = item["id"]
        flags = case_flags(item, hints)
        all_flags[cid] = flags
        video_rel = None
        try:
            if compose_case_video(item, protocol, out / "videos" / f"{cid}.mp4"):
                video_rel = f"videos/{cid}.mp4"
        except Exception as error:
            flags.append({"level": "error", "message": f"review video failed: {type(error).__name__}: {error}"})
        worst = next((lvl for lvl in ("error", "warn") if any(f["level"] == lvl for f in flags)), None)
        state = f"<span class='badge {worst or 'ok'}'>{worst.upper() if worst else 'no flags'}</span>"
        body = item.get("metrics", {}).get("body", {})
        traj = item.get("metrics", {}).get("trajectory", {})
        head = item.get("metrics", {}).get("head_rotation", {})
        ident_geo = _get(item.get("metrics", {}), "identity", "face_geometry_error.mean")
        warp_ratio = _get(item.get("metrics", {}), "temporal", "warp_error_ratio")
        rows.append(f"<tr><td><a href='#{html.escape(cid)}'>{html.escape(cid)}</a></td>"
                    f"<td>{html.escape(str(item.get('category', '')))}</td><td>{html.escape(item['status'])}</td>"
                    f"<td>{state} {len([f for f in flags if f['level'] != 'info'])}</td>"
                    f"<td class='num'>{_fmt(body.get('pck'))}</td><td class='num'>{_fmt(traj.get('trajectory_error'))}</td>"
                    f"<td class='num'>{_fmt(head.get('relative_geodesic_error_deg'))}</td>"
                    f"<td class='num'>{_fmt(ident_geo)}</td><td class='num'>{_fmt(warp_ratio)}</td></tr>")
        flag_html = "".join(f"<span class='badge {f['level']}'>{html.escape(f['message'])}</span>" for f in flags) \
            or "<span class='badge ok'>no flags</span>"
        tables = []
        for title, (group, fields) in METRIC_TABLE.items():
            g = item.get("metrics", {}).get(group)
            if not g:
                continue
            if g.get("status") == "UNAVAILABLE":
                tables.append(f"<div><b>{title}</b><p class='muted'>unavailable: {html.escape(str(g.get('reason')))}</p></div>")
                continue
            trs = "".join(f"<tr><td>{f}</td><td class='num'>{_fmt(_get(item['metrics'], group, f))}</td></tr>"
                          for f in fields)
            tables.append(f"<div><b>{title}</b><table>{trs}</table></div>")
        gen = item.get("generation") or {}
        integ = item.get("integrity") or {}
        extra = (f"<p class='muted'>integrity: {_fmt(integ.get('status'))} · codec {_fmt(integ.get('codec'))} · "
                 f"generation {_fmt(gen.get('generation_time_s'))} s · VRAM peak {_fmt(gen.get('vram_peak_gb'))} GB · "
                 f"model revision {_fmt((gen.get('model_revision') or '')[:12] or None)}"
                 f"{' · reused from ' + html.escape(item['reused_from']) if item.get('reused_from') else ''}</p>")
        err = f"<pre>{html.escape(item['error'])}</pre>" if item.get("error") else ""
        video = (f"<video src='{video_rel}' controls loop muted autoplay playsinline></video>" if video_rel
                 else "<p class='muted'>No video available for this case.</p>")
        sections.append(f"<section class='card' id='{html.escape(cid)}'><h2>{html.escape(cid)} "
                        f"<span class='muted'>· {html.escape(str(item.get('category', '')))}</span></h2>"
                        f"<p>{flag_html}</p>{video}{extra}{err}<div class='grid'>{''.join(tables)}</div></section>")

    n_eval = sum(c["status"] == "evaluated" for c in report["cases"])
    n_flag = sum(any(f["level"] in ("warn", "error") for f in fl) for fl in all_flags.values())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MOVA review — {html.escape(report.get('label', ''))}</title>
<style>{CSS}</style></head><body><main>
<h1>MOVA benchmark review</h1>
<p class="muted">label <b>{html.escape(str(report.get('label')))}</b> · status <b>{html.escape(str(report.get('status')))}</b> ·
{n_eval}/{len(report['cases'])} evaluated · {n_flag} flagged · benchmark {html.escape(report['benchmark_sha256'][:12])} ·
evaluator {html.escape(report['evaluator_sha256'][:12])} · quality status {html.escape(str(report.get('quality_status')))}</p>
<div class="card"><b>How to read this page.</b> Each video shows <i>reference | driver | generated | pose overlay</i>
(driver skeleton in cyan, generated in magenta). Flags are hints for where to look, not verdicts; <span class='null'>null</span>
means the metric could not be computed and is never treated as zero. Record your judgement in <code>review.csv</code>.</div>
<div class="card scroll"><table><tr><th>case</th><th>category</th><th>status</th><th>flags</th><th>body PCK</th>
<th>trajectory err</th><th>head rel. err °</th><th>face geometry err</th><th>warp ratio</th></tr>{''.join(rows)}</table></div>
{''.join(sections)}
</main></body></html>"""
    index = out / "index.html"
    index.write_text(page, encoding="utf-8")
    (out / "flags.json").write_text(json.dumps({"hints": hints, "cases": all_flags}, indent=2), encoding="utf-8")
    with (out / "review.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["benchmark_sha256", "case_id", "auto_flags", "reviewer", "identity", "motion", "face", "hands",
                         "temporal", "overall", "notes"])
        for item in report["cases"]:
            msgs = "; ".join(f["message"] for f in all_flags[item["id"]] if f["level"] != "info")
            writer.writerow([report["benchmark_sha256"], item["id"], msgs, "", "", "", "", "", "", "", ""])
    return index
