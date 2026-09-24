"""Blind pairwise GSB (Good / Same / Bad) human evaluation, per axis — the protocol of the Kling-MotionControl
report (arXiv 2603.03160): score = (G + S) / (B + S), from the CANDIDATE's point of view against a BASELINE.

Workflow:
  build_gsb_session(baseline_report, candidate_report, out)  -> index.html + ratings.csv (+ key.json, keep hidden)
  raters fill ratings.csv with L / S / R per axis (Left better / Same / Right better), one row per case and rater
  score_gsb(ratings.csv, key.json)                           -> per-axis G/S/B counts, GSB, win rate

Blinding: the rater never sees which side is baseline or candidate; the side is drawn per case with a seeded RNG
and stored only in key.json. Axes follow Kling plus hands (MOVA separates hands from expression).
GSB > 1 means the candidate is preferred; it is a human preference, not a proof of quality.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from common.config import resolve_path
from common.video_io import read_video, write_video

AXES = ("visual_quality", "dynamic_quality", "identity_preservation", "motion_accuracy", "expression_accuracy",
        "hands_accuracy", "overall")
AXIS_HELP = {
    "visual_quality": "sharpness, artifacts, realism of each frame",
    "dynamic_quality": "temporal stability: no flicker, boiling, popping or frozen parts",
    "identity_preservation": "same face, hair, outfit and proportions as the reference image",
    "motion_accuracy": "body pose, trajectory and timing follow the driver video",
    "expression_accuracy": "facial expression and head motion follow the driver",
    "hands_accuracy": "hand shapes and finger poses follow the driver; plausible hands",
    "overall": "which one would you ship",
}
VALID = {"L", "S", "R", ""}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fit(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2

    return cv2.resize(np.ascontiguousarray(frame), (w, h), interpolation=cv2.INTER_AREA)


def build_gsb_session(baseline_report: str | Path, candidate_report: str | Path, out_dir: str | Path, *,
                      seed: int = 0, title: str = "MOVA GSB session") -> dict[str, Path]:
    a_path, b_path = Path(baseline_report), Path(candidate_report)
    a, b = json.loads(a_path.read_text(encoding="utf-8")), json.loads(b_path.read_text(encoding="utf-8"))
    for key in ("benchmark_sha256", "protocol"):
        if a.get(key) != b.get(key):
            raise ValueError(f"Reports are not comparable: {key} differs")
    ca = {c["id"]: c for c in a["cases"] if c.get("status") == "evaluated"}
    cb = {c["id"]: c for c in b["cases"] if c.get("status") == "evaluated"}
    ids = sorted(ca.keys() & cb.keys())
    if not ids:
        raise ValueError("No case evaluated in both reports")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; a GSB session is never overwritten")
    (out / "videos").mkdir(parents=True)
    p = a["protocol"]
    w, h = p["width"], p["height"]
    rng = random.Random(seed)
    key = {"schema_version": 1, "seed": seed, "baseline": {"report": str(a_path), "sha256": _sha(a_path),
           "label": a.get("label")}, "candidate": {"report": str(b_path), "sha256": _sha(b_path),
           "label": b.get("label")}, "benchmark_sha256": a["benchmark_sha256"], "cases": {}}
    sections = []
    for cid in ids:
        candidate_left = rng.random() < 0.5
        key["cases"][cid] = {"left": "candidate" if candidate_left else "baseline"}
        left, right = (cb[cid], ca[cid]) if candidate_left else (ca[cid], cb[cid])
        lv, rv = read_video(Path(left["inputs"]["output"])), read_video(Path(right["inputs"]["output"]))
        motion = left["inputs"].get("motion")
        dv = read_video(resolve_path(motion)) if motion and resolve_path(motion).is_file() else []
        ref = left["inputs"].get("reference")
        ref_img = None
        if ref and resolve_path(ref).is_file():
            from PIL import Image

            with Image.open(resolve_path(ref)) as im:
                ref_img = _fit(np.asarray(im.convert("RGB")), w, h)
        n = min(len(lv), len(rv), len(dv) if dv else 10**9)
        frames = []
        for i in range(n):
            tiles = [ref_img if ref_img is not None else np.zeros((h, w, 3), np.uint8),
                     _fit(dv[i], w, h) if dv else np.zeros((h, w, 3), np.uint8), _fit(lv[i], w, h), _fit(rv[i], w, h)]
            frames.append(np.concatenate(tiles, axis=1))
        write_video(out / "videos" / f"{cid}.mp4", frames, p["fps"])
        sections.append(f"<section class='card'><h2>{html.escape(cid)}</h2>"
                        f"<p class='muted'>reference | driver | <b>LEFT</b> | <b>RIGHT</b></p>"
                        f"<video src='videos/{html.escape(cid)}.mp4' controls loop muted autoplay playsinline></video>"
                        f"</section>")
    with (out / "ratings.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["case_id", "rater", *AXES, "notes"])
        for cid in ids:
            wr.writerow([cid, "", *([""] * len(AXES)), ""])
    (out / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    axes_html = "".join(f"<li><b>{a_}</b>: {html.escape(t)}</li>" for a_, t in AXIS_HELP.items())
    (out / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;margin:0;background:#f6f7fb;color:#1d2230}}main{{max-width:1180px;margin:auto;padding:16px}}
.card{{background:#fff;border:1px solid #e4e7ec;border-radius:12px;padding:16px;margin:16px 0}}video{{width:100%;border-radius:8px;background:#000}}
.muted{{color:#667085}}@media (prefers-color-scheme:dark){{body{{background:#0d1117;color:#e6edf3}}.card{{background:#161b22;border-color:#30363d}}.muted{{color:#8b949e}}}}</style>
</head><body><main><h1>{html.escape(title)}</h1>
<div class="card"><p>For each case, compare LEFT and RIGHT against the reference image and the driver video. In
<code>ratings.csv</code> write <b>L</b> (left better), <b>S</b> (same) or <b>R</b> (right better) for each axis, leave
blank if you cannot judge. Add one row per rater. Do not open <code>key.json</code> before rating.</p><ul>{axes_html}</ul></div>
{''.join(sections)}</main></body></html>""", encoding="utf-8")
    return {"index": out / "index.html", "ratings": out / "ratings.csv", "key": out / "key.json"}


def gsb_score(g: int, s: int, b: int) -> float | None:
    """Kling's definition (G + S) / (B + S); None when undefined (no Same and no Bad)."""
    return None if b + s == 0 else (g + s) / (b + s)


def score_gsb(ratings_csv: str | Path, key_json: str | Path) -> dict[str, Any]:
    key = json.loads(Path(key_json).read_text(encoding="utf-8"))
    counts = {axis: {"G": 0, "S": 0, "B": 0, "unrated": 0} for axis in AXES}
    raters, rows = set(), 0
    with Path(ratings_csv).open(newline="", encoding="utf-8") as fh:
        for line, row in enumerate(csv.DictReader(fh), start=2):
            cid = row["case_id"]
            if cid not in key["cases"]:
                raise ValueError(f"ratings.csv line {line}: unknown case {cid!r}")
            rows += 1
            raters.add(row.get("rater", ""))
            candidate_left = key["cases"][cid]["left"] == "candidate"
            for axis in AXES:
                v = (row.get(axis) or "").strip().upper()
                if v not in VALID:
                    raise ValueError(f"ratings.csv line {line}, {axis}: {v!r} is not L, S, R or blank")
                if v == "":
                    counts[axis]["unrated"] += 1
                elif v == "S":
                    counts[axis]["S"] += 1
                else:  # L/R -> Good/Bad for the candidate
                    counts[axis]["G" if (v == "L") == candidate_left else "B"] += 1
    result = {"baseline": key["baseline"]["label"], "candidate": key["candidate"]["label"],
              "benchmark_sha256": key["benchmark_sha256"], "rows": rows, "raters": sorted(raters), "axes": {},
              "definition": "GSB = (G + S) / (B + S) from the candidate's point of view; > 1 favours the candidate",
              "note": "Human preference on this benchmark only; not an objective quality measure."}
    for axis, c in counts.items():
        judged = c["G"] + c["S"] + c["B"]
        result["axes"][axis] = {**c, "judged": judged, "gsb": gsb_score(c["G"], c["S"], c["B"]),
                                "win_rate": c["G"] / (c["G"] + c["B"]) if c["G"] + c["B"] else None}
    return result
