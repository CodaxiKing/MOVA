"""Smoke tests that exercise real code paths without large weights.

- Extraction: MediaPipe on a synthetic video (no person) must run and write all outputs.
- Baseline: a tiny randomly-initialised WanVACE pipeline must accept our conditioning on CPU.
"""

import numpy as np
import pytest
import torch

from common.video_io import write_video
from preprocessing.mp_models import MODEL_DIR


def _mp_models_present() -> bool:
    return all((MODEL_DIR / f"{n}.task").exists() for n in ("pose_landmarker_full", "face_landmarker", "hand_landmarker"))


@pytest.mark.skipif(not _mp_models_present(), reason="MediaPipe .task models not downloaded")
def test_extraction_on_synthetic_video(tmp_path):
    from preprocessing.pipeline import ExtractionConfig, extract_motion

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (128, 96, 3), dtype=np.uint8) for _ in range(6)]
    video = write_video(tmp_path / "noise.mp4", frames, fps=12)
    summary = extract_motion(video, tmp_path / "out", ExtractionConfig())
    for name in ("body_motion.pt", "face_motion.pt", "hand_motion.pt", "body_preview.mp4",
                 "face_preview.mp4", "hands_preview.mp4", "pose_openpose.mp4", "summary.json"):
        assert (tmp_path / "out" / name).exists(), name
    body = torch.load(tmp_path / "out" / "body_motion.pt", weights_only=False)
    assert body["kp2d"].shape == (6, 33, 4) and summary["num_frames"] == 6
    assert summary["body"]["detection_rate"] == 0.0  # noise has no person


@pytest.mark.skipif(not _mp_models_present(), reason="MediaPipe .task models not downloaded")
def test_extraction_detects_person(tmp_path):
    """Positive detection on scikit-image's bundled public-domain 'astronaut' photo, moved over time."""
    skdata = pytest.importorskip("skimage.data")
    import cv2

    from preprocessing.pipeline import ExtractionConfig, extract_motion

    img = skdata.astronaut()
    frames = []
    for i in range(8):
        M = cv2.getRotationMatrix2D((256, 256), 6 * np.sin(i / 2), 1.0)
        frames.append(cv2.warpAffine(img, M, (512, 512), borderMode=cv2.BORDER_REFLECT))
    video = write_video(tmp_path / "astronaut.mp4", frames, fps=8)
    summary = extract_motion(video, tmp_path / "out", ExtractionConfig(write_previews=False))
    assert summary["body"]["detection_rate"] >= 0.75
    assert summary["face"]["detection_rate"] >= 0.75
    face = torch.load(tmp_path / "out" / "face_motion.pt", weights_only=False)
    smile = face["blendshape_names"].index("mouthSmileLeft")
    assert float(face["blendshapes"][:, smile].mean()) > 0.3  # the subject is smiling


def _tiny_vace_pipe():
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanVACEPipeline, WanVACETransformer3DModel

    torch.manual_seed(0)
    vae = AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                           temperal_downsample=[False, True, True])
    transformer = WanVACETransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=16, out_channels=16,
        text_dim=32, freq_dim=32, ffn_dim=32, num_layers=3, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
        rope_max_seq_len=32, vace_layers=[0, 2], vace_in_channels=96,
    )
    return WanVACEPipeline(tokenizer=None, text_encoder=None, vae=vae,
                           scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0), transformer=transformer)


def test_baseline_generate_with_tiny_vace():
    pytest.importorskip("diffusers")
    from PIL import Image

    from inference.baseline_vace import BaselineSettings, generate
    from inference.conditioning import letterbox, prepare_control_frames

    s = BaselineSettings(height=32, width=32, num_frames=5, num_inference_steps=2, dtype="float32")
    control = prepare_control_frames([np.zeros((48, 48, 3), np.uint8)] * 8, 32, 32, 5)
    ref = letterbox(Image.new("RGB", (40, 60), (200, 100, 50)), 32, 32)
    pe, ne = torch.randn(1, 8, 32), torch.zeros(1, 8, 32)
    frames, stats = generate(_tiny_vace_pipe(), s, ref, control, pe, ne)
    assert np.asarray(frames).shape == (5, 32, 32, 3)
    assert stats["generation_time_s"] >= 0
