import numpy as np
import pytest

from common.video_io import write_video
from evaluation.video import prepare_generated_frames, validate_video


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -0.1, 1.1])
def test_reject_invalid_generated_pixels(bad):
    frames = np.zeros((2, 16, 16, 3), dtype=np.float32)
    frames[0, 0, 0, 0] = bad
    with pytest.raises(ValueError):
        prepare_generated_frames(frames, count=2, width=16, height=16)


def test_generation_shape_and_count():
    frames = np.zeros((2, 16, 16, 3), dtype=np.float32)
    for count, width in [(3, 16), (2, 32)]:
        with pytest.raises(ValueError):
            prepare_generated_frames(frames, count=count, width=width, height=16)
    assert prepare_generated_frames(frames, count=2, width=16, height=16)[0].dtype == np.uint8


def test_encoded_video_contract(tmp_path):
    frames = np.zeros((3, 16, 32, 3), dtype=np.uint8)
    path = write_video(tmp_path / 'valid.mp4', frames, 12)
    expected = dict(count=3, width=32, height=16, fps=12)
    report = validate_video(path, **expected)
    assert report['status'] == 'PASS'
    assert report['duration_s'] == pytest.approx(0.25)
    assert report['codec']
    for field, value in [('count', 4), ('width', 16), ('fps', 24)]:
        with pytest.raises(ValueError):
            validate_video(path, **{**expected, field: value})
    path.write_bytes(b'not a video')
    with pytest.raises(ValueError):
        validate_video(path, **expected)
