import numpy as np

from core.motion_quality import full_body_coverage


def test_cropped_ankles_do_not_count_as_full_body():
    kp = np.zeros((3, 33, 4), dtype=np.float32)
    kp[:, [27, 28], :2] = 0.5
    kp[:, [27, 28], 3] = 1
    kp[1, 27, 1] = 1.2
    kp[2, 28, 3] = 0.1
    result = full_body_coverage({"kp2d": kp, "present": np.ones(3, dtype=bool)})
    assert result == {"frames": 3, "full_body_frames": 1, "full_body_fraction": 0.333}
