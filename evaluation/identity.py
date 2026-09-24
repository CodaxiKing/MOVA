"""Identity preservation diagnostics (reference image vs generated frames). No downloads, no face-recognition weights.

identity-v1 measures three things that can drift independently:

1. Face geometry: 3D MediaPipe landmarks of rigid facial structure (eye corners, nose, face width, forehead).
   Pairwise distances are invariant to rotation, translation and scale (normalised by their median); the error is
   the mean |log ratio| against the reference. Mouth, lips, eyebrows and chin are excluded because expressions move
   them. This catches morphology drift (face gets wider, eyes move apart). It is NOT a face-recognition embedding:
   two different people with similar proportions can score close.
2. Face appearance: CIELAB colour of the face region (mean ΔE76 and histogram intersection).
3. Body appearance: CIELAB colour of the torso polygon (shoulders + hips): outfit colour drift.

Optional 4. Embedding cosine similarity through a pluggable `embedder` (e.g. DINOv2, Apache-2.0). InsightFace/ArcFace
weights are non-commercial and are deliberately not used. Without an embedder the field is UNAVAILABLE, never 0.

Missing detections are reported as coverage and never counted as perfect matches.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import cv2
import numpy as np

IDENTITY_VERSION = "identity-v1"

# MediaPipe Face Mesh indices of rigid structure (both eyes' corners, nose bridge/tip/base/alae, cheekbones,
# face sides at the ears, forehead). Expression-driven points (lips, mouth corners, brows, chin) are excluded.
RIGID_FACE_POINTS = (33, 133, 263, 362, 168, 6, 197, 195, 5, 4, 1, 2, 98, 327, 234, 454, 127, 356, 10, 151,
                     116, 345, 123, 352)
TORSO_POINTS = (11, 12, 24, 23)  # left shoulder, right shoulder, right hip, left hip (polygon order)


def _pairwise(points: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(points[:, None] - points[None], axis=-1)
    return d[np.triu_indices(len(points), k=1)]


def face_geometry_signature(landmarks: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """(478, 3) normalised MediaPipe landmarks -> scale-free vector of rigid pairwise distances (log)."""
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.shape[0] <= max(RIGID_FACE_POINTS) or not np.isfinite(lm).all():
        return None
    # x, y are normalised by width/height; MediaPipe's z uses roughly the x scale.
    pts = lm[list(RIGID_FACE_POINTS)] * np.array([width, height, width], dtype=np.float64)
    d = _pairwise(pts)
    if np.any(d <= 1e-9):
        return None
    return np.log(d / np.median(d))


def geometry_error(reference_sig: np.ndarray, generated_sig: np.ndarray) -> float:
    """Mean |log distance ratio| after removing global scale: 0 = identical proportions; 0.05 ~ 5 % average."""
    diff = generated_sig - reference_sig
    return float(np.mean(np.abs(diff - np.median(diff))))


def _polygon_mask(shape: tuple[int, int], points_xy: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(points_xy).astype(np.int32)], 1)
    return mask.astype(bool)


def face_region_mask(landmarks: np.ndarray, height: int, width: int) -> np.ndarray | None:
    lm = np.asarray(landmarks, dtype=np.float64)
    if not np.isfinite(lm).all():
        return None
    xy = lm[:, :2] * np.array([width, height])
    hull = cv2.convexHull(np.round(xy).astype(np.int32))
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    mask = mask.astype(bool)
    return mask if mask.sum() >= 16 else None


def torso_region_mask(body_kp: np.ndarray, height: int, width: int, visibility: float = 0.5) -> np.ndarray | None:
    kp = np.asarray(body_kp, dtype=np.float64)
    corners = kp[list(TORSO_POINTS)]
    if not np.isfinite(corners).all() or (corners[:, 3] < visibility).any():
        return None
    mask = _polygon_mask((height, width), corners[:, :2] * np.array([width, height]))
    return mask if mask.sum() >= 16 else None


def _lab(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)
    # OpenCV 8-bit Lab: L*255/100, a+128, b+128 -> back to CIE units.
    return np.stack([lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128], axis=-1)


def color_stats(rgb: np.ndarray, mask: np.ndarray, bins: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """(mean Lab colour, normalised joint a*b* histogram) of the masked pixels."""
    lab = _lab(rgb)[mask]
    hist, _, _ = np.histogram2d(lab[:, 1], lab[:, 2], bins=bins, range=[[-64, 64], [-64, 64]])
    return lab.mean(0), hist.ravel() / max(hist.sum(), 1)


def color_distance(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
    """(ΔE76 of mean colours, histogram intersection in [0, 1])."""
    return float(np.linalg.norm(a[0] - b[0])), float(np.minimum(a[1], b[1]).sum())


class Embedder(Protocol):
    name: str

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray: ...  # (N, D) float


def _crop(rgb: np.ndarray, mask: np.ndarray, pad: float = 0.15) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    py, px = int((y1 - y0) * pad), int((x1 - x0) * pad)
    h, w = mask.shape
    return rgb[max(0, y0 - py):min(h, y1 + py), max(0, x0 - px):min(w, x1 + px)]


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    a = np.asarray(values)
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


def identity_metrics(reference_rgb: np.ndarray, reference_face: np.ndarray | None, reference_body: np.ndarray | None,
                     frames: list[np.ndarray], face_track: dict[str, Any] | None, body_track: dict[str, Any] | None,
                     *, embedder: Embedder | None = None, visibility: float = 0.5) -> dict[str, Any]:
    """Compare the reference identity with every generated frame.

    reference_face: (478, 3) landmarks of the reference image or None; reference_body: (33, 4) or None.
    face_track / body_track: generated tracks as saved by preprocessing (landmarks/present, kp2d/present).
    """
    rh, rw = reference_rgb.shape[:2]
    n = len(frames)
    out: dict[str, Any] = {"metric_version": IDENTITY_VERSION, "frames": n,
                           "reference_face_detected": reference_face is not None,
                           "reference_torso_detected": False}
    face_ok = np.zeros(n, bool)
    if face_track is not None and n:
        face_ok = np.asarray(face_track["present"], bool)[:n]
    ref_sig = face_geometry_signature(reference_face, rw, rh) if reference_face is not None else None
    ref_face_mask = face_region_mask(reference_face, rh, rw) if reference_face is not None else None
    ref_face_col = color_stats(reference_rgb, ref_face_mask) if ref_face_mask is not None else None
    ref_torso_mask = torso_region_mask(reference_body, rh, rw, visibility) if reference_body is not None else None
    ref_torso_col = color_stats(reference_rgb, ref_torso_mask) if ref_torso_mask is not None else None
    out["reference_torso_detected"] = ref_torso_col is not None

    geo, face_de, face_hist, torso_de, torso_hist, gen_face_crops = [], [], [], [], [], []
    torso_frames = 0
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        if face_ok[i]:
            lm = np.asarray(face_track["landmarks"][i], dtype=np.float64)
            if ref_sig is not None:
                sig = face_geometry_signature(lm, w, h)
                if sig is not None:
                    geo.append(geometry_error(ref_sig, sig))
            mask = face_region_mask(lm, h, w)
            if mask is not None:
                if ref_face_col is not None:
                    de, hi = color_distance(ref_face_col, color_stats(frame, mask))
                    face_de.append(de)
                    face_hist.append(hi)
                gen_face_crops.append(_crop(frame, mask))
        if body_track is not None and ref_torso_col is not None and bool(np.asarray(body_track["present"])[i]):
            mask = torso_region_mask(np.asarray(body_track["kp2d"][i]), h, w, visibility)
            if mask is not None:
                torso_frames += 1
                de, hi = color_distance(ref_torso_col, color_stats(frame, mask))
                torso_de.append(de)
                torso_hist.append(hi)

    ref_ok = reference_face is not None
    out["face_coverage"] = float(face_ok.mean()) if (n and ref_ok) else None
    out["face_geometry_error"] = _summary(geo) | {"frames": len(geo),
                                                  "units": "mean |log distance ratio| of rigid face landmarks"}
    out["face_color_delta_e"] = _summary(face_de) | {"units": "CIE76 ΔE of mean Lab colour"}
    out["face_color_hist_intersection"] = _summary(face_hist)
    out["torso_coverage"] = (torso_frames / n) if (n and ref_torso_col is not None) else None
    out["torso_color_delta_e"] = _summary(torso_de)
    out["torso_color_hist_intersection"] = _summary(torso_hist)
    out["embedding"] = _embedding_similarity(embedder, reference_rgb, ref_face_mask, gen_face_crops)
    return out


def _embedding_similarity(embedder, reference_rgb, ref_mask, crops) -> dict[str, Any]:
    if embedder is None:
        return {"status": "UNAVAILABLE", "reason": "no embedding backend configured (see evaluation/identity.py)"}
    if ref_mask is None or not crops:
        return {"status": "UNAVAILABLE", "reason": "face not detected in reference or generated frames",
                "backend": embedder.name}
    emb = np.asarray(embedder([_crop(reference_rgb, ref_mask)] + crops), dtype=np.float64)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    cos = emb[1:] @ emb[0]
    return {"backend": embedder.name, "frames": int(len(cos)), **_summary(cos.tolist()),
            "units": "cosine similarity of face-crop embeddings"}


class HFImageEmbedder:
    """Mean-pooled last hidden state of a Hugging Face vision backbone (e.g. facebook/dinov2-small, Apache-2.0).

    Never downloads: use `from_pretrained(..., local_files_only=True)`; if the weights are not cached it raises and
    the caller reports UNAVAILABLE. `model` may also be any module (tests use a tiny random Dinov2Model).
    """

    def __init__(self, model, image_size: int = 224, name: str = "hf-image-embedder",
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.model, self.image_size, self.name = model.eval(), image_size, name
        self.mean, self.std = np.asarray(mean), np.asarray(std)

    @classmethod
    def from_pretrained(cls, repo: str = "facebook/dinov2-small", revision: str | None = None) -> "HFImageEmbedder":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(repo, revision=revision, local_files_only=True)
        return cls(model, name=f"{repo}@{revision or 'cached'}")

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        import torch

        batch = []
        for c in crops:
            x = cv2.resize(np.ascontiguousarray(c), (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            batch.append(((x / 255.0 - self.mean) / self.std).transpose(2, 0, 1))
        with torch.no_grad():
            hidden = self.model(pixel_values=torch.tensor(np.stack(batch), dtype=torch.float32)).last_hidden_state
        return hidden.mean(1).numpy()


def load_embedder(spec: str | None) -> tuple[Embedder | None, str | None]:
    """`None` -> no embedder. `hf:<repo>[@rev]` -> cached HF model or (None, reason)."""
    if not spec:
        return None, None
    if not spec.startswith("hf:"):
        return None, f"unknown embedder spec {spec!r}"
    repo, _, rev = spec[3:].partition("@")
    try:
        return HFImageEmbedder.from_pretrained(repo, rev or None), None
    except Exception as e:  # not cached / offline: reported, never downloaded
        return None, f"{repo} not available locally ({type(e).__name__})"


def analyze_reference(rgb: np.ndarray, min_confidence: float = 0.5) -> tuple[np.ndarray | None, np.ndarray | None]:
    """MediaPipe face (478,3) and body (33,4) landmarks of one image; None when not detected."""
    from preprocessing.face.extract_face import FaceExtractor
    from preprocessing.pose.extract_body import BodyExtractor

    rgb = np.ascontiguousarray(rgb)
    face = FaceExtractor(min_confidence)
    body = BodyExtractor("pose_landmarker_full", min_confidence)
    try:
        f = face.process(rgb, 0)
        body.process(rgb, 0)
        b = body.result()
    finally:
        face.close()
        body.close()
    body_kp = b["kp2d"][0] if len(b["present"]) and b["present"][0] else None
    return (np.asarray(f) if f is not None else None), body_kp


EmbedderFactory = Callable[[], Embedder]
